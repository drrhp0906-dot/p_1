/* ============================================================================
   p_1 · docs/train.js — the in-browser live trainer (Task 6)
   ----------------------------------------------------------------------------
   Character-level Config A trained LIVE on docs/corpus_mini.js. Zero
   dependencies: built purely on docs/tensor.js (autograd) + docs/model.js
   (the transformer). UMD: window.P1.train / module.exports.

   PUBLIC API
     P1.train.buildVocab(text)
        -> { chars: sorted unique chars, stoi: Map, itos: string[] }
        Deterministic (Array.sort on code units). vocab_size = chars.length.
     new P1.train.Trainer({
          pairs,      // string[] from window.P1_CORPUS_MINI.pairs  (required)
          model,      // optional P1.Transformer — reuse weights (see below)
          lr = 1e-3, ctx = 256, batchSize = 8, seed = 0,
          warmupSteps = 200, clipNorm = 1.0,
          onStep, onGenerate                       // UI hooks (optional)
        })
        If `model` is given it is trained in place (its config supplies
        vocab_size + ctx); otherwise the Trainer BUILDS ITS OWN Config A
        Transformer: {n_layer:2, n_head:4, d_model:128, d_ff:512,
        ctx, vocab_size: vocab.length} — note vocab_size is the REAL char
        count (~75), not the CONFIG_A placeholder 96 (96 was a ceiling;
        the embedding table is sized to the actual vocab).
     trainer.step() -> { step, loss, tokensProcessed, stepMs, tokensPerSec }
        One batch: sample windows -> forward -> fused CE -> backward ->
        global-norm clip (1.0) -> AdamW update. Fire onStep hook.
     trainer.sample(prompt, maxNew=220, temperature=0.8) -> string
        Blocking greedy-ish ancestral sample (temp divides the logits
        BEFORE softmax). Stops at maxNew, 300 decoded chars, or when the
        text ends with a sentence end + "\n\n" (pair boundary).
     trainer.beginSample(prompt, maxNew, temperature) -> sampler
        CHUNKED sampler for the page: s.step(k) generates up to k tokens
        and returns; s.done / s.text let the UI drive it across frames so
        generation never blocks the tab.
     trainer.reset(seed?)   fresh weights + zeroed optimizer, same data.
     trainer.vocab, trainer.model, trainer.stepCount, trainer.numParams()

   DATA PIPELINE
     text = pairs.join("\n\n") + "\n\n"   (trailing separator: the model
                                          also sees how pairs END)
     -> encoded once into an Int32Array stream (~200k tokens).
     Every step draws batchSize random windows of ctx tokens (seeded
     mulberry32, same stream as sampling), ids = stream[s..s+ctx),
     targets = stream[s+1..s+ctx]. model.forward(ids, B) returns logits
     [B*ctx, vocab] ALREADY 2-D (model.js contract), so the fused
     crossEntropy consumes them directly; a defensive reshape([−1, vocab])
     is applied if a future forward ever returns 3-D.

   ADAMW (from scratch, over model.params() in canonical order)
     m, v Float32Arrays per param, beta1 0.9, beta2 0.95, eps 1e-8,
     wd 0.01 DECOUPLED and applied ONLY to 2-D params (weights incl. the
     tied wte/wpe) — biases and layerNorm g/b (1-D) are skipped.
     Bias correction uses the standard 1/(1-β^t). LR schedule: LINEAR
     WARMUP over the first `warmupSteps` (lr * step/warmupSteps) then
     CONSTANT. Why no cosine: a browser run is hundreds-to-a-few-thousand
     steps; over that horizon a cosine tail changes nothing measurable and
     makes the loss curve harder to read for a human watching it live.

   GRADIENT CLIPPING: global norm over ALL params, scaled to max 1.0.

   MEMORY / GC NOTES
     Preallocated and reused: batch id + target Int32Arrays, AdamW m/v,
     the sampler's probability buffer, per-T causal masks (cached inside
     model.js, bounded by ctx). The engine is immutable-graph by design —
     each step's intermediate tensors are short-lived garbage that V8
     collects cheaply; pooling them would mean rewriting tensor.js.

   SAMPLING RNG: one mulberry32(seed) stream drives BOTH window selection
     and token sampling — a page session is reproducible from the seed.
     Pass { rng } to sample()/beginSample() to override.
   ============================================================================ */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    const P1 = require('./tensor.js');
    const Transformer = require('./model.js').Transformer;
    module.exports = factory(P1, Transformer);
  } else {
    if (!root.P1 || !root.P1.Transformer) {
      throw new Error('train.js needs tensor.js + model.js loaded first (window.P1)');
    }
    root.P1.train = factory(root.P1, root.P1.Transformer);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (P1, Transformer) {
  'use strict';

  const mulberry32 = P1.mulberry32;
  const VERSION = '1.0.0';

  const now = () =>
    (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();

  /* ------------------------------------------------------------ vocab */

  function buildVocab(text) {
    const set = new Set();
    for (let i = 0; i < text.length; i++) set.add(text[i]);
    const chars = Array.from(set).sort();   // UTF-16 code-unit order — deterministic
    const stoi = new Map();
    const itos = chars.slice();
    for (let i = 0; i < chars.length; i++) stoi.set(chars[i], i);
    return { chars: itos, stoi, itos };
  }

  /* ------------------------------------------------------------ AdamW */

  class AdamW {
    constructor(params, opt = {}) {
      this.beta1 = opt.beta1 !== undefined ? opt.beta1 : 0.9;
      this.beta2 = opt.beta2 !== undefined ? opt.beta2 : 0.95;
      this.eps = opt.eps !== undefined ? opt.eps : 1e-8;
      this.wd = opt.wd !== undefined ? opt.wd : 0.01;
      this.params = params;
      this.m = params.map(p => new Float32Array(p.size));
      this.v = params.map(p => new Float32Array(p.size));
      this.t = 0;
    }
    zeroState() {
      for (const a of this.m) a.fill(0);
      for (const a of this.v) a.fill(0);
      this.t = 0;
    }
    /* lr here is the SCHEDULED lr for this step (warmup applied by caller). */
    update(lr) {
      this.t++;
      const b1 = this.beta1, b2 = this.beta2, eps = this.eps, wd = this.wd;
      const bc1 = 1 - Math.pow(b1, this.t);
      const bc2 = 1 - Math.pow(b2, this.t);
      for (let pi = 0; pi < this.params.length; pi++) {
        const p = this.params[pi];
        const m = this.m[pi], v = this.v[pi], g = p.grad, d = p.data;
        const decay = p.ndim === 2;   // wd ONLY on weight matrices, not biases/ln
        for (let i = 0; i < p.size; i++) {
          const gi = g[i];
          m[i] = b1 * m[i] + (1 - b1) * gi;
          v[i] = b2 * v[i] + (1 - b2) * gi * gi;
          let upd = (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
          if (decay) upd += wd * d[i];          // decoupled weight decay
          d[i] -= lr * upd;
        }
      }
      return this;
    }
  }

  /* ---------------------------------------------------------- Trainer */

  const DEFAULTS = {
    lr: 1e-3, ctx: 256, batchSize: 8, seed: 0,
    warmupSteps: 200, clipNorm: 1.0,
    beta1: 0.9, beta2: 0.95, eps: 1e-8, wd: 0.01
  };

  class Trainer {
    constructor(opts) {
      if (!opts || !Array.isArray(opts.pairs) || opts.pairs.length === 0) {
        throw new Error('Trainer: opts.pairs (array of "Q: ...\\nA: ..." strings) is required');
      }
      const o = Object.assign({}, DEFAULTS, opts);
      this.pairs = o.pairs.slice();
      this.lr = o.lr;
      this.ctx = o.ctx;
      this.batchSize = o.batchSize;
      this.seed = o.seed >>> 0;
      this.warmupSteps = o.warmupSteps;
      this.clipNorm = o.clipNorm;
      this.beta1 = o.beta1; this.beta2 = o.beta2;
      this.eps = o.eps; this.wd = o.wd;
      this.onStep = typeof o.onStep === 'function' ? o.onStep : null;
      this.onGenerate = typeof o.onGenerate === 'function' ? o.onGenerate : null;

      // ---- vocab + token stream (built once; reset() keeps them) ----
      const text = this.pairs.join('\n\n') + '\n\n';
      this.vocab = buildVocab(text);
      const V = this.vocab.itos.length;
      const stoi = this.vocab.stoi;
      this.stream = new Int32Array(text.length);
      for (let i = 0; i < text.length; i++) {
        const id = stoi.get(text[i]);
        if (id === undefined) throw new Error('Trainer: char not in vocab (impossible) at ' + i);
        this.stream[i] = id;
      }

      // ---- model: accept one, or build Config A ourselves ----
      if (o.model) {
        this.model = o.model;
        if (this.model.config.vocab_size !== V) {
          throw new Error('Trainer: model vocab_size ' + this.model.config.vocab_size +
            ' != corpus vocab ' + V);
        }
        if (this.model.config.ctx < this.ctx) {
          throw new Error('Trainer: model ctx ' + this.model.config.ctx + ' < requested ' + this.ctx);
        }
        if (this.model.config.ctx !== this.ctx) this.ctx = this.model.config.ctx;
      } else {
        this.model = new Transformer({
          n_layer: 2, n_head: 4, d_model: 128, d_ff: 512,
          ctx: this.ctx, vocab_size: V
        }, this.seed);
      }

      this._buildOptimizer();
      this._buildSamplerBuffers();
    }

    _buildOptimizer() {
      this.opt = new AdamW(this.model.params(), {
        beta1: this.beta1, beta2: this.beta2, eps: this.eps, wd: this.wd
      });
      this.stepCount = 0;
      this.tokensTotal = 0;
      this._tps = new Float64Array(64);  // ring of recent step durations (ms)
      this._tpsTok = new Float64Array(64);
      this._tpsHead = 0;
      this._tokensPerSec = 0;
    }

    _buildSamplerBuffers() {
      // reused per-step batch buffers (the obvious preallocations; the
      // forward graph itself is engine-owned and short-lived)
      this._ids = new Int32Array(this.batchSize * this.ctx);
      this._targets = new Int32Array(this.batchSize * this.ctx);
      this._rng = mulberry32(this.seed);
    }

    /* Fresh weights + zeroed optimizer; data/vocab stay. Optionally reseed. */
    reset(seed) {
      if (seed !== undefined) this.seed = seed >>> 0;
      const cfg = this.model.config;
      this.model = new Transformer(cfg, this.seed);
      this._buildOptimizer();
      this._buildSamplerBuffers();
      return this;
    }

    get numParams() { return this.model.numParams(); }

    encodeText(text, skipUnknown) {
      const out = [];
      for (const ch of text) {
        const id = this.vocab.stoi.get(ch);
        if (id !== undefined) out.push(id);
        else if (!skipUnknown) throw new Error('encodeText: unknown char ' + JSON.stringify(ch));
      }
      return out;
    }

    decodeIds(ids) {
      const itos = this.vocab.itos;
      let s = '';
      for (let i = 0; i < ids.length; i++) s += itos[ids[i]];
      return s;
    }

    /* ------------------------------------------------------- one step */

    step() {
      const t0 = now();
      const B = this.batchSize, C = this.ctx, N = this.stream.length;
      const hi = N - C - 1;                     // last valid window start
      if (hi < 1) throw new Error('Trainer: stream too short for ctx=' + C);
      const rng = this._rng, stream = this.stream;
      const ids = this._ids, targets = this._targets;

      for (let b = 0; b < B; b++) {
        const s = Math.floor(rng() * hi);       // random ctx-window, seeded
        const base = b * C;
        for (let t = 0; t < C; t++) {
          ids[base + t] = stream[s + t];
          targets[base + t] = stream[s + t + 1];
        }
      }

      this.model.zeroGrad();
      const logits = this.model.forward(ids, B);      // [B*ctx, vocab] (2-D)
      const ce = (logits.ndim === 3 ? logits.reshape([-1, logits.shape[2]]) : logits)
        .crossEntropy(targets);
      const lossVal = ce.item();
      if (!isFinite(lossVal)) throw new Error('Trainer.step: non-finite loss at step ' + (this.stepCount + 1));
      ce.backward();

      // ---- global-norm gradient clipping (max 1.0) ----
      const params = this.model.params();
      let sq = 0;
      for (let pi = 0; pi < params.length; pi++) {
        const g = params[pi].grad;
        for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
      }
      const norm = Math.sqrt(sq);
      if (norm > this.clipNorm) {
        const coef = this.clipNorm / (norm + 1e-9);
        for (let pi = 0; pi < params.length; pi++) {
          const g = params[pi].grad;
          for (let i = 0; i < g.length; i++) g[i] *= coef;
        }
      }

      // ---- AdamW, linear warmup -> constant lr ----
      this.stepCount++;
      const lrT = this.lr * Math.min(1, this.stepCount / this.warmupSteps);
      this.opt.update(lrT);

      const tokens = B * C;
      this.tokensTotal += tokens;
      const dt = now() - t0;
      this._tps[this._tpsHead] = Math.max(dt, 0.01);
      this._tpsTok[this._tpsHead] = tokens;
      this._tpsHead = (this._tpsHead + 1) % this._tps.length;
      let ms = 0, tok = 0;
      for (let i = 0; i < this._tps.length; i++) { ms += this._tps[i]; tok += this._tpsTok[i]; }
      this._tokensPerSec = tok / (ms / 1000);

      const rec = {
        step: this.stepCount, loss: lossVal,
        tokensProcessed: tokens, stepMs: dt,
        tokensPerSec: this._tokensPerSec, gradNorm: norm, lr: lrT
      };
      if (this.onStep) this.onStep(rec);
      return rec;
    }

    /* ------------------------------------------------------- sampling */

    /* Chunked sampler: call s.step(k) each frame until s.done. Keeps [1,T]
       single-row forwards (fast) and the tab responsive. */
    beginSample(prompt, maxNew = 220, temperature = 0.8, opts = {}) {
      const rng = opts.rng || this._rng;
      const itos = this.vocab.itos, V = itos.length, C = this.model.config.ctx;
      const ids = [];
      if (prompt) {
        for (const ch of prompt) {
          const id = this.vocab.stoi.get(ch);
          if (id !== undefined) ids.push(id);   // unknown prompt chars: skipped
        }
      }
      if (ids.length === 0) ids.push(Math.floor(rng() * V));  // seed char
      const probs = new Float32Array(V);       // reused across tokens
      const st = {
        ids, probs, rng, itos, V, C,
        temp: Math.max(temperature, 1e-4),
        maxNew: Math.max(1, maxNew),
        newCount: 0,
        text: this.decodeIds(ids),
        done: false
      };
      const finish = () => { st.done = true; if (this.onGenerate) this.onGenerate(st.text); };
      st.step = (k = 16) => {
        let n = 0;
        while (!st.done && n < k) { this._sampleToken(st); n++; }
        if (st.done) finish();
        return st;
      };
      st.finishNow = () => { if (!st.done) { st.done = true; finish(); } return st; };
      return st;
    }

    _sampleToken(st) {
      // slide a ctx-window over the growing sequence; [1, T] forward
      const L = st.ids.length;
      const T = Math.min(L, st.C);
      const win = L > st.C ? st.ids.slice(L - st.C) : st.ids;
      const logits = this.model.forward(win, 1);          // [T, V]
      const row = logits.data, off = (T - 1) * st.V;
      let m = -Infinity;
      for (let j = 0; j < st.V; j++) if (row[off + j] > m) m = row[off + j];
      let sum = 0;
      for (let j = 0; j < st.V; j++) {                    // divide by temp FIRST
        const e = Math.exp((row[off + j] - m) / st.temp);
        st.probs[j] = e; sum += e;
      }
      let r = st.rng() * sum, pick = st.V - 1;
      for (let j = 0; j < st.V; j++) { r -= st.probs[j]; if (r <= 0) { pick = j; break; } }
      st.ids.push(pick);
      st.text += st.itos[pick];
      st.newCount++;
      if (st.text.endsWith('\n\n')) st.done = true;       // sentence end + pair boundary
      else if (st.text.length >= 300) st.done = true;     // hard 300-char cap
      else if (st.newCount >= st.maxNew) st.done = true;
    }

    /* Blocking convenience wrapper (Node tests, quick checks). */
    sample(prompt, maxNew = 220, temperature = 0.8, opts = {}) {
      const st = this.beginSample(prompt, maxNew, temperature, opts);
      while (!st.done) st.step(4096);
      return st.text;
    }
  }

  return {
    VERSION: VERSION,
    buildVocab: buildVocab,
    AdamW: AdamW,
    Trainer: Trainer
  };
});
