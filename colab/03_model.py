"""
p_1 · colab/03_model.py — decoder-only transformer on the NumPy autograd twin
--------------------------------------------------------------------------------
The Task 4 Colab sibling of docs/model.js: SAME architecture, SAME op set,
SAME parameter order, SAME seeds. Built ONLY from colab/tensor_py.py ops
(embed, add, slice, reshape, transpose, matmul, matmulT, bmm, scale, softmax,
layerNorm, gelu, crossEntropy) — no ML libraries.

ARCHITECTURE (pre-LN decoder-only, identical in both languages)
  x = wte[ids] + wpe[0:T]                    (embed + broadcast add)
  per block:  x = x + attn(ln1(x));  x = x + mlp(ln2(x))
  final:      x = lnf(x);  logits = x @ wte^T   (weight-tied head = matmulT)
  attn: fused qkv [d,d*3] -> split heads -> per-(batch,head) scores
        q@k^T * (1/sqrt(d_head)), causal mask via ADDING a [T,T] leaf of
        zeros / -1e9 (SAME constant both languages) BEFORE rowwise softmax,
        probs @ v, merge heads, proj [d,d].
  mlp:  fc [d, d_ff] -> gelu -> proj [d_ff, d].
  LayerNorm: last axis, eps=1e-5, BIASED variance.

SEED DERIVATION (the cross-language contract — mirrored in docs/model.js)
  Every parameter owns seed = baseSeed + its canonical index in PARAM order:
    0 'wte'   1 'wpe'   then per layer i (block base = 2 + 12*i):
      +0  blk{i}.ln1.g        +1  blk{i}.ln1.b
      +2  blk{i}.attn.qkv.w   +3  blk{i}.attn.qkv.b
      +4  blk{i}.attn.proj.w  +5  blk{i}.attn.proj.b
      +6  blk{i}.ln2.g        +7  blk{i}.ln2.b
      +8  blk{i}.mlp.fc.w     +9  blk{i}.mlp.fc.b
      +10 blk{i}.mlp.proj.w   +11 blk{i}.mlp.proj.b
    then lnf.g (= 2 + 12*n_layer), lnf.b.
  heInit draws from a FRESH mulberry32(seed) stream per tensor, so parameters
  are independent of each other's draw counts; ones/zeros params (ln g/b,
  biases) consume no randomness but still occupy an index.

WEIGHT LAYOUT (linear weights are [in, out]; head uses matmulT)
  wte [vocab, d_model], wpe [ctx, d_model], qkv.w [d_model, 3*d_model],
  proj.w [d_model, d_model], fc.w [d_model, d_ff], mlp.proj.w [d_ff, d_model].

__main__ SMOKE (Task 4 acceptance): both configs, param counts, forward+
backward on B=2 T=32 random ids (seed 42), all grads finite, 10 SGD steps
lr 1e-3, per-step losses printed, no NaN.

Run:  python3 colab/03_model.py
"""

import math

from tensor_py import Tensor, heInit, mulberry32

# The two contract configs (identical keys in docs/model.js).
CONFIG_A = {'n_layer': 2, 'n_head': 4, 'd_model': 128, 'd_ff': 512,
            'ctx': 256, 'vocab_size': 96}
CONFIG_B = {'n_layer': 4, 'n_head': 4, 'd_model': 256, 'd_ff': 1024,
            'ctx': 512, 'vocab_size': 4096}

NEG_INF = -1e9          # causal-mask constant — MUST match docs/model.js
LN_EPS = 1e-5           # layerNorm eps — MUST match docs/model.js
SMOKE_SEED = 42         # id-draw seed for the smoke test (both languages)
SMOKE_B, SMOKE_T = 2, 32
SMOKE_STEPS, SMOKE_LR = 10, 1e-3


def param_names(cfg):
    """Canonical parameter order — mirrored EXACTLY in docs/model.js."""
    names = ['wte', 'wpe']
    for i in range(cfg['n_layer']):
        names += ['blk%d.ln1.g' % i, 'blk%d.ln1.b' % i,
                  'blk%d.attn.qkv.w' % i, 'blk%d.attn.qkv.b' % i,
                  'blk%d.attn.proj.w' % i, 'blk%d.attn.proj.b' % i,
                  'blk%d.ln2.g' % i, 'blk%d.ln2.b' % i,
                  'blk%d.mlp.fc.w' % i, 'blk%d.mlp.fc.b' % i,
                  'blk%d.mlp.proj.w' % i, 'blk%d.mlp.proj.b' % i]
    names += ['lnf.g', 'lnf.b']
    return names


def random_ids(n, vocab, seed):
    """Deterministic token ids shared with the JS side: mulberry32(seed)
    uniforms, ids[k] = floor(u * vocab). Same stream -> same ids."""
    r = mulberry32(seed & 0xFFFFFFFF)
    return [int(r() * vocab) for _ in range(n)]


class Transformer:
    """Decoder-only pre-LN transformer. forward(ids, B) returns logits
    [B*T, vocab] (2-D, row-major over (b, t)) ready for crossEntropy."""

    def __init__(self, config, seed=0):
        cfg = dict(config)
        d, ff = cfg['d_model'], cfg['d_ff']
        V, C, H = cfg['vocab_size'], cfg['ctx'], cfg['n_head']
        if d % H != 0:
            raise ValueError('d_model must be divisible by n_head')
        self.config = cfg
        self.seed = seed
        self.d_head = d // H

        names = param_names(cfg)
        P = {}
        idx = 0

        def nxt():
            nonlocal idx
            idx += 1
            return seed + (idx - 1)   # seed = baseSeed + canonical index

        P['wte'] = heInit(V, d, nxt(), requires_grad=True, name='wte')
        P['wpe'] = heInit(C, d, nxt(), requires_grad=True, name='wpe')
        for i in range(cfg['n_layer']):
            b = 'blk%d.' % i
            P[b + 'ln1.g'] = Tensor.ones([d], requires_grad=True, name=b + 'ln1.g')
            P[b + 'ln1.b'] = Tensor.zeros([d], requires_grad=True, name=b + 'ln1.b')
            P[b + 'attn.qkv.w'] = heInit(d, 3 * d, nxt(), requires_grad=True,
                                         name=b + 'attn.qkv.w')
            P[b + 'attn.qkv.b'] = Tensor.zeros([3 * d], requires_grad=True,
                                               name=b + 'attn.qkv.b')
            P[b + 'attn.proj.w'] = heInit(d, d, nxt(), requires_grad=True,
                                          name=b + 'attn.proj.w')
            P[b + 'attn.proj.b'] = Tensor.zeros([d], requires_grad=True,
                                                name=b + 'attn.proj.b')
            P[b + 'ln2.g'] = Tensor.ones([d], requires_grad=True, name=b + 'ln2.g')
            P[b + 'ln2.b'] = Tensor.zeros([d], requires_grad=True, name=b + 'ln2.b')
            P[b + 'mlp.fc.w'] = heInit(d, ff, nxt(), requires_grad=True,
                                       name=b + 'mlp.fc.w')
            P[b + 'mlp.fc.b'] = Tensor.zeros([ff], requires_grad=True,
                                             name=b + 'mlp.fc.b')
            P[b + 'mlp.proj.w'] = heInit(ff, d, nxt(), requires_grad=True,
                                         name=b + 'mlp.proj.w')
            P[b + 'mlp.proj.b'] = Tensor.zeros([d], requires_grad=True,
                                               name=b + 'mlp.proj.b')
        P['lnf.g'] = Tensor.ones([d], requires_grad=True, name='lnf.g')
        P['lnf.b'] = Tensor.zeros([d], requires_grad=True, name='lnf.b')

        if list(P.keys()) != names:
            raise AssertionError('param order drift: %r vs %r' % (list(P.keys()), names))
        self.P = P
        self._names = names
        self._mask_cache = {}

    # ------------------------------------------------------------ params

    def paramNames(self):
        return list(self._names)

    def params(self):
        return [self.P[n] for n in self._names]

    def numParams(self):
        return sum(p.size for p in self.params())

    def zeroGrad(self):
        for p in self.params():
            p.zeroGrad()

    def setParams(self, named):
        """Task 7 bridge: copy weights into params BY NAME. `named` maps param
        name -> array-like (flat or logical shape; e.g. the checkpoint
        'param:<name>' arrays or the parsed web_model.bin tensors). The name
        set must match paramNames() exactly and each array must hold exactly
        p.size elements — hard errors otherwise. Mirrored in docs/model.js."""
        import numpy as np
        got = dict(named)
        want = self._names
        missing = [n for n in want if n not in got]
        extra = [n for n in got if n not in set(want)]
        if missing or extra:
            raise KeyError('setParams name mismatch: missing=%r extra=%r'
                           % (missing, extra))
        for name in want:
            p = self.P[name]
            a = np.asarray(got[name], dtype=np.float32).reshape(-1)
            if a.size != p.size:
                raise ValueError('setParams %s: %d elements, model wants %d'
                                 % (name, a.size, p.size))
            p.data = a.copy()
        return self

    # ------------------------------------------------------------ forward

    def _causal_mask(self, T):
        """[T, T] float32 leaf: 0 where j <= i, NEG_INF where j > i.
        Non-trainable (requires_grad False); cached per T."""
        if T not in self._mask_cache:
            import numpy as np
            m = np.zeros((T, T), dtype=np.float32)
            for i in range(T):
                for j in range(i + 1, T):
                    m[i, j] = NEG_INF
            self._mask_cache[T] = Tensor([T, T], m.reshape(-1), requires_grad=False,
                                         name='causal_mask%d' % T)
        return self._mask_cache[T]

    def forward(self, ids, B):
        """ids: flat sequence of B*T ints (row-major b*T + t).
        Returns logits Tensor [B*T, vocab]."""
        cfg = self.config
        d, H, hd = cfg['d_model'], cfg['n_head'], self.d_head
        C = cfg['ctx']
        N = len(ids)
        if B <= 0 or N % B != 0:
            raise ValueError('forward: len(ids)=%d not divisible by B=%d' % (N, B))
        T = N // B
        if T < 1 or T > C:
            raise ValueError('forward: T=%d outside [1, ctx=%d]' % (T, C))
        P = self.P

        # embeddings: wte lookup + position slice, [B,T,d] + [T,d] broadcast
        x = P['wte'].embed(list(ids)).reshape([B, T, d]).add(
            P['wpe'].slice([[0, T], None]))

        mask = self._causal_mask(T)                       # [T, T], no grad
        for i in range(cfg['n_layer']):
            b = 'blk%d.' % i

            # ---- attention block: x = x + attn(ln1(x))
            ln1 = x.layerNorm(P[b + 'ln1.g'], P[b + 'ln1.b'], LN_EPS)
            x2 = ln1.reshape([B * T, d])
            qkv = x2.matmul(P[b + 'attn.qkv.w']).add(P[b + 'attn.qkv.b'])
            qkv5 = qkv.reshape([B, T, 3, H, hd])
            q = (qkv5.slice([None, None, [0, 1], None, None])
                 .reshape([B, T, H, hd]).transpose([0, 2, 1, 3])
                 .reshape([B * H, T, hd]))
            k = (qkv5.slice([None, None, [1, 2], None, None])
                 .reshape([B, T, H, hd]).transpose([0, 2, 1, 3])
                 .transpose([0, 1, 3, 2]).reshape([B * H, hd, T]))
            v = (qkv5.slice([None, None, [2, 3], None, None])
                 .reshape([B, T, H, hd]).transpose([0, 2, 1, 3])
                 .reshape([B * H, T, hd]))
            scores = q.bmm(k).scale(1.0 / math.sqrt(hd))   # [B*H, T, T]
            scores = scores.add(mask)                      # causal, BEFORE softmax
            probs = (scores.reshape([B * H * T, T]).softmax()
                     .reshape([B * H, T, T]))
            att = probs.bmm(v)                             # [B*H, T, hd]
            att = (att.reshape([B, H, T, hd]).transpose([0, 2, 1, 3])
                   .reshape([B * T, d]))
            attout = att.matmul(P[b + 'attn.proj.w']).add(P[b + 'attn.proj.b'])
            x = x.add(attout.reshape([B, T, d]))

            # ---- mlp block: x = x + mlp(ln2(x))
            ln2 = x.layerNorm(P[b + 'ln2.g'], P[b + 'ln2.b'], LN_EPS)
            x3 = ln2.reshape([B * T, d])
            fc = x3.matmul(P[b + 'mlp.fc.w']).add(P[b + 'mlp.fc.b'])
            mo = fc.gelu().matmul(P[b + 'mlp.proj.w']).add(P[b + 'mlp.proj.b'])
            x = x.add(mo.reshape([B, T, d]))

        xf = x.layerNorm(P['lnf.g'], P['lnf.b'], LN_EPS)
        logits = xf.reshape([B * T, d]).matmulT(P['wte'])  # weight-tied head
        return logits                                       # [B*T, V]

    # ------------------------------------------------------------ smoke

    def sgd_step(self, lr):
        """Plain SGD used only by the smoke test (AdamW arrives in Task 5).
        One f64 evaluation + one f32 rounding per element, like JS."""
        for p in self.params():
            p.data = (p.data.astype('float64') - lr * p.grad.astype('float64')
                      ).astype('float32')


def shifted_targets(ids, B, T, vocab):
    """Next-token targets for the smoke: within each row, targets[t] =
    ids[t+1], last position wraps to ids[0]. Deterministic in both languages.
    (Wrapping keeps every id in range without extra plumbing.)"""
    out = []
    for b in range(B):
        row = ids[b * T:(b + 1) * T]
        out.extend(row[1:] + row[:1])
    return out


def smoke(cfg, label):
    import numpy as np
    print('--- smoke %s: %s' % (label, cfg))
    model = Transformer(cfg, seed=SMOKE_SEED)
    print('    numParams = %d' % model.numParams())
    V = cfg['vocab_size']
    ids = random_ids(SMOKE_B * SMOKE_T, V, SMOKE_SEED)
    tg = shifted_targets(ids, SMOKE_B, SMOKE_T, V)
    for step in range(1, SMOKE_STEPS + 1):
        logits = model.forward(ids, SMOKE_B)
        loss = logits.crossEntropy(tg)
        model.zeroGrad()
        loss.backward()
        grads = model.params()
        bad = [p.name for p in grads
               if not np.all(np.isfinite(p.grad))]
        if bad:
            raise AssertionError('non-finite grads in %r at step %d' % (bad, step))
        print('    step %2d · loss %.6f' % (step, loss.item()))
        model.sgd_step(SMOKE_LR)
    final = model.forward(ids, SMOKE_B)
    if not np.all(np.isfinite(final.data)):
        raise AssertionError('non-finite activations after 10 steps')
    print('    after 10 steps: loss %.6f · all params/grads finite'
          % model.forward(ids, SMOKE_B).crossEntropy(tg).item())
    return model


if __name__ == '__main__':
    print('p_1 · 03_model.py — transformer smoke (engine: colab/tensor_py.py)')
    ma = smoke(CONFIG_A, 'Config A (browser baby)')
    mb = smoke(CONFIG_B, 'Config B (Colab sibling)')
    # cross-language reference line: first 8 wte values, A seed 42
    print('ConfigA wte[:8] (seed 42):',
          ' '.join('%.9f' % v for v in ma.P['wte'].data[:8]))
    print('ConfigB wte[:8] (seed 42):',
          ' '.join('%.9f' % v for v in mb.P['wte'].data[:8]))
    print('OK — both configs: backward runs, grads finite, 10 steps, no NaN')
