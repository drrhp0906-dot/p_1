/* ============================================================================
   p_1 · docs/model.js — the transformer twin on the tensor.js engine
   ----------------------------------------------------------------------------
   Decoder-only pre-LN transformer, pure JavaScript, zero dependencies.
   This is the browser sibling of colab/03_model.py: SAME architecture,
   SAME op set, SAME canonical parameter order, SAME seed derivation.
   Built ONLY from docs/tensor.js ops (embed, add, slice, reshape, transpose,
   matmul, matmulT, bmm, scale, softmax, layerNorm, gelu, crossEntropy).

   ARCHITECTURE (identical to colab/03_model.py — read that for the prose)
     x = wte[ids] + wpe[0:T]                  embed + broadcast add
     per block:  x = x + attn(ln1(x));  x = x + mlp(ln2(x))
     final:      x = lnf(x);  logits = x @ wte^T   (weight-tied head)
     attn: fused qkv [d,3d] -> heads -> q@k^T * (1/sqrt(d_head)) per
           (batch,head) via bmm, causal mask ADDED as a [T,T] zeros/-1e9
           leaf BEFORE rowwise softmax, probs@v, merge heads, proj [d,d].
     mlp:  fc [d,d_ff] -> gelu -> proj [d_ff,d]. LayerNorm: last axis,
           eps=1e-5, biased variance.

   SEED DERIVATION (the cross-language contract — mirrored in 03_model.py)
     Every parameter owns seed = baseSeed + its canonical index:
       0 'wte'  1 'wpe'  then per layer i (block base = 2 + 12*i):
         +0 blk{i}.ln1.g      +1 blk{i}.ln1.b
         +2 blk{i}.attn.qkv.w +3 blk{i}.attn.qkv.b
         +4 blk{i}.attn.proj.w +5 blk{i}.attn.proj.b
         +6 blk{i}.ln2.g      +7 blk{i}.ln2.b
         +8 blk{i}.mlp.fc.w   +9 blk{i}.mlp.fc.b
         +10 blk{i}.mlp.proj.w +11 blk{i}.mlp.proj.b
       then lnf.g (= 2 + 12*n_layer), lnf.b.
     heInit draws from a FRESH mulberry32(seed) per tensor; ones/zeros
     params consume no randomness but still occupy an index.

   WEIGHT LAYOUT: linear weights [in, out]; wte [vocab, d_model]; the head
   is matmulT(wte) so wte never needs a transpose copy.

   UMD: window.P1.Transformer (+ CONFIG_A/CONFIG_A) / module.exports.
   ============================================================================ */

(function (root, factory) {
  let P1 = root.P1;
  if (typeof module === 'object' && module.exports) {
    if (!P1) P1 = require('./tensor.js');
    module.exports = factory(P1);
  } else {
    if (!P1) throw new Error('model.js needs tensor.js loaded first (window.P1)');
    const api = factory(P1);
    root.P1.Transformer = api.Transformer;
    root.P1.MODEL = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (P1) {
  'use strict';

  const Tensor = P1.Tensor, heInit = P1.heInit, mulberry32 = P1.mulberry32;

  const VERSION = '1.0.0';

  // The two contract configs (identical keys in colab/03_model.py).
  const CONFIG_A = { n_layer: 2, n_head: 4, d_model: 128, d_ff: 512,
                     ctx: 256, vocab_size: 96 };
  const CONFIG_B = { n_layer: 4, n_head: 4, d_model: 256, d_ff: 1024,
                     ctx: 512, vocab_size: 4096 };

  const NEG_INF = -1e9;   // causal-mask constant — MUST match colab/03_model.py
  const LN_EPS = 1e-5;    // layerNorm eps — MUST match colab/03_model.py

  function paramNames(cfg) {
    const names = ['wte', 'wpe'];
    for (let i = 0; i < cfg.n_layer; i++) {
      names.push('blk' + i + '.ln1.g', 'blk' + i + '.ln1.b',
                 'blk' + i + '.attn.qkv.w', 'blk' + i + '.attn.qkv.b',
                 'blk' + i + '.attn.proj.w', 'blk' + i + '.attn.proj.b',
                 'blk' + i + '.ln2.g', 'blk' + i + '.ln2.b',
                 'blk' + i + '.mlp.fc.w', 'blk' + i + '.mlp.fc.b',
                 'blk' + i + '.mlp.proj.w', 'blk' + i + '.mlp.proj.b');
    }
    names.push('lnf.g', 'lnf.b');
    return names;
  }

  /* Deterministic token ids shared with Python: mulberry32(seed) uniforms,
     ids[k] = floor(u * vocab). Same stream -> same ids (bit-exact PRNG). */
  function randomIds(n, vocab, seed) {
    const r = mulberry32(seed >>> 0);
    const out = new Array(n);
    for (let k = 0; k < n; k++) out[k] = Math.floor(r() * vocab);
    return out;
  }

  /* Next-token targets for the smoke: within each row targets[t] = ids[t+1],
     last position wraps to ids[0]. Identical to Python shifted_targets. */
  function shiftedTargets(ids, B, T) {
    const out = new Array(B * T);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        out[b * T + t] = ids[b * T + ((t + 1) % T)];
      }
    }
    return out;
  }

  class Transformer {
    constructor(config, seed = 0) {
      const cfg = {
        n_layer: config.n_layer, n_head: config.n_head,
        d_model: config.d_model, d_ff: config.d_ff,
        ctx: config.ctx, vocab_size: config.vocab_size
      };
      if (cfg.d_model % cfg.n_head !== 0) {
        throw new Error('d_model must be divisible by n_head');
      }
      this.config = cfg;
      this.seed = seed;
      this.d_head = cfg.d_model / cfg.n_head;

      const d = cfg.d_model, ff = cfg.d_ff, V = cfg.vocab_size, C = cfg.ctx;
      const names = paramNames(cfg);
      const P = {};
      let idx = 0;
      const nxt = () => seed + (idx++);   // seed = baseSeed + canonical index

      P.wte = heInit(V, d, nxt(), { requiresGrad: true, name: 'wte' });
      P.wpe = heInit(C, d, nxt(), { requiresGrad: true, name: 'wpe' });
      for (let i = 0; i < cfg.n_layer; i++) {
        const b = 'blk' + i + '.';
        P[b + 'ln1.g'] = Tensor.ones([d], { requiresGrad: true, name: b + 'ln1.g' });
        P[b + 'ln1.b'] = Tensor.zeros([d], { requiresGrad: true, name: b + 'ln1.b' });
        P[b + 'attn.qkv.w'] = heInit(d, 3 * d, nxt(), { requiresGrad: true, name: b + 'attn.qkv.w' });
        P[b + 'attn.qkv.b'] = Tensor.zeros([3 * d], { requiresGrad: true, name: b + 'attn.qkv.b' });
        P[b + 'attn.proj.w'] = heInit(d, d, nxt(), { requiresGrad: true, name: b + 'attn.proj.w' });
        P[b + 'attn.proj.b'] = Tensor.zeros([d], { requiresGrad: true, name: b + 'attn.proj.b' });
        P[b + 'ln2.g'] = Tensor.ones([d], { requiresGrad: true, name: b + 'ln2.g' });
        P[b + 'ln2.b'] = Tensor.zeros([d], { requiresGrad: true, name: b + 'ln2.b' });
        P[b + 'mlp.fc.w'] = heInit(d, ff, nxt(), { requiresGrad: true, name: b + 'mlp.fc.w' });
        P[b + 'mlp.fc.b'] = Tensor.zeros([ff], { requiresGrad: true, name: b + 'mlp.fc.b' });
        P[b + 'mlp.proj.w'] = heInit(ff, d, nxt(), { requiresGrad: true, name: b + 'mlp.proj.w' });
        P[b + 'mlp.proj.b'] = Tensor.zeros([d], { requiresGrad: true, name: b + 'mlp.proj.b' });
      }
      P['lnf.g'] = Tensor.ones([d], { requiresGrad: true, name: 'lnf.g' });
      P['lnf.b'] = Tensor.zeros([d], { requiresGrad: true, name: 'lnf.b' });

      // order sanity — the canonical list must match key insertion order
      if (Object.keys(P).join(',') !== names.join(',')) {
        throw new Error('param order drift');
      }
      this.P = P;
      this._names = names;
      this._maskCache = {};
    }

    paramNames() { return this._names.slice(); }
    params() { return this._names.map(n => this.P[n]); }
    numParams() { return this.params().reduce((s, p) => s + p.size, 0); }
    zeroGrad() { for (const p of this.params()) p.zeroGrad(); }

    /* Task 7 bridge: copy weights into params BY NAME. `named` maps param
       name -> array-like (Float32Array, Array, or Map) — e.g. the tensors
       parsed from web_model.bin by weights.js. The name set must match
       paramNames() exactly and each array must hold exactly p.size elements
       — hard errors otherwise. Mirrored in colab/03_model.py. */
    setParams(named) {
      const obj = (named instanceof Map) ? Object.fromEntries(named) : named;
      const got = Object.keys(obj);
      const want = this._names;
      const missing = want.filter(n => !(n in obj));
      const extra = got.filter(n => want.indexOf(n) === -1);
      if (missing.length || extra.length) {
        throw new Error('setParams name mismatch: missing=[' +
          missing.join(',') + '] extra=[' + extra.join(',') + ']');
      }
      for (const name of want) {
        const p = this.P[name];
        const src = obj[name];
        const f32 = (src instanceof Float32Array) ? src : Float32Array.from(src);
        if (f32.length !== p.size) {
          throw new Error('setParams ' + name + ': ' + f32.length +
            ' elements, model wants ' + p.size);
        }
        p.data.set(f32);
      }
      return this;
    }

    /* [T, T] leaf: 0 where j <= i, NEG_INF where j > i. Non-trainable. */
    causalMask(T) {
      if (!this._maskCache[T]) {
        const data = new Float32Array(T * T);
        for (let i = 0; i < T; i++) {
          for (let j = i + 1; j < T; j++) data[i * T + j] = NEG_INF;
        }
        this._maskCache[T] = new Tensor([T, T], data,
          { requiresGrad: false, name: 'causal_mask' + T });
      }
      return this._maskCache[T];
    }

    /* ids: flat array of B*T ints (row-major b*T + t). Returns logits
       Tensor [B*T, vocab] ready for crossEntropy. */
    forward(ids, B) {
      const cfg = this.config;
      const d = cfg.d_model, H = cfg.n_head, hd = this.d_head;
      const C = cfg.ctx;
      const N = ids.length;
      if (B <= 0 || N % B !== 0) {
        throw new Error('forward: len(ids)=' + N + ' not divisible by B=' + B);
      }
      const T = N / B;
      if (T < 1 || T > C) {
        throw new Error('forward: T=' + T + ' outside [1, ctx=' + C + ']');
      }
      const P = this.P;

      // embeddings: wte lookup + position slice, [B,T,d] + [T,d] broadcast
      let x = P.wte.embed(ids).reshape([B, T, d])
        .add(P.wpe.slice([[0, T], null]));

      const mask = this.causalMask(T);                 // [T, T], no grad
      for (let i = 0; i < cfg.n_layer; i++) {
        const b = 'blk' + i + '.';

        // ---- attention block: x = x + attn(ln1(x))
        const ln1 = x.layerNorm(P[b + 'ln1.g'], P[b + 'ln1.b'], LN_EPS);
        const x2 = ln1.reshape([B * T, d]);
        const qkv = x2.matmul(P[b + 'attn.qkv.w']).add(P[b + 'attn.qkv.b']);
        const qkv5 = qkv.reshape([B, T, 3, H, hd]);
        const q = qkv5.slice([null, null, [0, 1], null, null])
          .reshape([B, T, H, hd]).transpose([0, 2, 1, 3])
          .reshape([B * H, T, hd]);
        const k = qkv5.slice([null, null, [1, 2], null, null])
          .reshape([B, T, H, hd]).transpose([0, 2, 1, 3])
          .transpose([0, 1, 3, 2]).reshape([B * H, hd, T]);
        const v = qkv5.slice([null, null, [2, 3], null, null])
          .reshape([B, T, H, hd]).transpose([0, 2, 1, 3])
          .reshape([B * H, T, hd]);
        const scores = q.bmm(k).scale(1 / Math.sqrt(hd));  // [B*H, T, T]
        const masked = scores.add(mask);               // causal, BEFORE softmax
        const probs = masked.reshape([B * H * T, T]).softmax()
          .reshape([B * H, T, T]);
        const att = probs.bmm(v)                       // [B*H, T, hd]
          .reshape([B, H, T, hd]).transpose([0, 2, 1, 3])
          .reshape([B * T, d]);
        const attout = att.matmul(P[b + 'attn.proj.w']).add(P[b + 'attn.proj.b']);
        x = x.add(attout.reshape([B, T, d]));

        // ---- mlp block: x = x + mlp(ln2(x))
        const ln2 = x.layerNorm(P[b + 'ln2.g'], P[b + 'ln2.b'], LN_EPS);
        const x3 = ln2.reshape([B * T, d]);
        const fc = x3.matmul(P[b + 'mlp.fc.w']).add(P[b + 'mlp.fc.b']);
        const mo = fc.gelu().matmul(P[b + 'mlp.proj.w']).add(P[b + 'mlp.proj.b']);
        x = x.add(mo.reshape([B, T, d]));
      }

      const xf = x.layerNorm(P['lnf.g'], P['lnf.b'], LN_EPS);
      return xf.reshape([B * T, d]).matmulT(P.wte);    // weight-tied head
    }

    /* Plain SGD used only by smokes (AdamW arrives in Task 5). */
    sgdStep(lr) {
      for (const p of this.params()) {
        for (let i = 0; i < p.size; i++) p.data[i] -= lr * p.grad[i];
      }
      return this;
    }
  }

  /* Smoke: forward+backward + 10 SGD steps, finite-grad + no-NaN asserts.
     Used by scripts/test_task4_model.js and docs/model_test.html. */
  function smoke(cfg, label, log, opts = {}) {
    const B = opts.B || 2, T = opts.T || 32;
    const STEPS = opts.steps || 10, LR = opts.lr || 1e-3;
    const SEED = opts.seed || 42;
    log('--- smoke ' + label + ': ' + JSON.stringify(cfg));
    const model = new Transformer(cfg, SEED);
    log('    numParams = ' + model.numParams());
    const ids = randomIds(B * T, cfg.vocab_size, SEED);
    const tg = shiftedTargets(ids, B, T);
    let lastLoss = NaN;
    for (let step = 1; step <= STEPS; step++) {
      const logits = model.forward(ids, B);
      const loss = logits.crossEntropy(tg);
      model.zeroGrad();
      loss.backward();
      for (const p of model.params()) {
        for (let i = 0; i < p.size; i++) {
          if (!isFinite(p.grad[i])) {
            throw new Error('non-finite grad in ' + p.name + '[' + i + '] at step ' + step);
          }
        }
      }
      lastLoss = loss.item();
      if (step === 1 || step === STEPS || step % 3 === 0) {
        log('    step ' + String(step).padStart(2) + ' · loss ' + lastLoss.toFixed(6));
      }
      model.sgdStep(LR);
    }
    const finalLogits = model.forward(ids, B);
    for (let i = 0; i < finalLogits.size; i++) {
      if (!isFinite(finalLogits.data[i])) throw new Error('NaN in final logits');
    }
    const finalLoss = model.forward(ids, B).crossEntropy(tg).item();
    log('    after ' + STEPS + ' steps: loss ' + finalLoss.toFixed(6) +
        ' · all params/grads finite');
    return { model: model, loss: finalLoss };
  }

  return {
    VERSION: VERSION,
    Transformer: Transformer,
    CONFIG_A: CONFIG_A,
    CONFIG_B: CONFIG_B,
    paramNames: paramNames,
    randomIds: randomIds,
    shiftedTargets: shiftedTargets,
    smoke: smoke,
    NEG_INF: NEG_INF,
    LN_EPS: LN_EPS
  };
});
