"""
p_1 · colab/tensor_py.py — the autograd engine, NumPy twin of docs/tensor.js
--------------------------------------------------------------------------------
Reverse-mode automatic differentiation in pure Python + NumPy. Zero ML
libraries. This is the Colab sibling of docs/tensor.js (Task 1) and must keep
the SAME op set, SAME API surface, SAME broadcasting semantics and SAME
numerics (see the parity contract at the bottom).

DESIGN NOTES (mirroring docs/tensor.js, differences called out)

- Tensor: shape[], data (np.float32, FLAT 1-D like the JS Float32Array),
  grad (np.float32 flat), requires_grad. Every op allocates a fresh output
  tensor — no in-place mutation — so graphs are immutable.

- "float32 engine" semantics: the JS engine computes each op in float64 and
  stores into a Float32Array after each op. The Python twin does exactly the
  same: inputs are upcast to float64, the op is computed, the result is
  rounded to float32 once. EXCEPTION — matmul/bmm/matmulT use NumPy matmul on
  float32 directly (BLAS, internal order not controllable). Task 4 parity
  tolerance is 1e-2; the measured cross-language diff is ~1e-5..1e-4 (see
  docs/model_fixtures.json + scripts/test_task4_model.js).

- Broadcasting: right-aligned, NumPy-native (same rule as the JS
  broadcastShapes). Backward "unbroadcasts" by summing over broadcast axes.

- matmul is 2-D only in the API (like JS). JS uses a fixed i,k,j loop with
  float32 rounding at every step; NumPy/BLAS cannot copy that order, which is
  precisely why the parity budget is 1e-2 and not 1e-7.

- Backward: iterative post-order DFS -> topological order (no recursion).
  Grads ACCUMULATE (+=) so tensors consumed multiple times (weight-tied head,
  residuals) get every contribution. _prev is an insertion-ordered tuple to
  mirror the JS Set iteration order.

RNG (the cross-language determinism linchpin)
- mulberry32(seed): the JS integer pipeline ported EXACTLY. JS Math.imul is
  32-bit signed multiplication; the low 32 bits of a product depend only on
  the low 32 bits of the operands, so unsigned masking (x * y) & 0xFFFFFFFF
  reproduces it bit-for-bit. `>>> n` is an unsigned shift = `>> n` on
  unsigned values. Same seed -> same uniforms in both languages.
- make_gaussian(rand): Marsaglia polar method, spare caching, identical
  control flow to JS. Both languages are IEEE-754 double, and sqrt/log are
  deterministic, so the gaussians match (tanh may differ by <=1 ulp across
  libms — absorbed by the 1e-2 parity budget).

TASK 4 NEW OPS (added to BOTH engines, gradient-checked on both sides —
see colab/test_tensor_py.py and scripts/test_task4_model.js):
- layerNorm(g, b, eps=1e-5): last axis, BIASED variance (GPT-2 style),
  y = g * (x - mu) / sqrt(var + eps) + b, full backward for x, g, b.
- slice(ranges): one [start, end) pair per axis (None = full axis),
  scatter-add backward.
- transpose(perm): out[k] = in[perm[k]] (NumPy convention), inverse-permute
  backward.
- bmm(w): batched [N, M, K] @ [N, K, P] -> [N, M, P] with per-batch matmul
  backward.
- matmulT(w): x @ w^T for 2-D x [M, K], w [O, K]. THE WEIGHT-TIED HEAD:
  logits = lnf_out.matmulT(wte) with wte [vocab, d_model] — forward is a
  matmul against the transpose VIEW w.T (no copy of wte, no separate head
  parameter); backward adds G @ w into dx and G^T @ x into w's grad.
  Decision documented in both engines.

PARITY CONTRACT (keep in sync with docs/tensor.js):
  same op names, same defaults (eps=1e-5, heInit std=sqrt(2/fanIn),
  [fanIn, fanOut] layout), same seed -> same weights bit-for-bit, GELU
  0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))), causal mask constant -1e9.
"""

import math

import numpy as np

VERSION = '1.1.0'
GELU_C = math.sqrt(2 / math.pi)  # tanh-approximation GELU (GPT-2 style)


# --------------------------------------------------------------------- RNG

def mulberry32(seed):
    """Exact port of the mulberry32 PRNG in docs/tensor.js.

    JS ops, bit-level: a = (a + 0x6D2B79F5) | 0;
      t = imul(a ^ (a >>> 15), 1 | a);
      t = (t + imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    Everything here is unsigned-mod-2^32; Math.imul is (x*y) & 0xFFFFFFFF
    because multiplication mod 2^32 only sees the low 32 bits. Verified
    against Node: same seed -> same first 10 uniforms (printed by both
    colab/03_model.py and scripts/test_task4_model.js).
    """
    state = seed & 0xFFFFFFFF

    def rand():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = (state ^ (state >> 15)) & 0xFFFFFFFF          # a ^ (a >>> 15)
        t = (t * (1 | state)) & 0xFFFFFFFF                # imul(t, 1 | a)
        t = ((t + ((t ^ (t >> 7)) * (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rand


def make_gaussian(rand):
    """Marsaglia polar method with spare caching — exact port of JS
    makeGaussian. Consumes the SAME uniform stream in the same order."""
    spare = None

    def gauss():
        nonlocal spare
        if spare is not None:
            v = spare
            spare = None
            return v
        while True:
            u = rand() * 2.0 - 1.0
            v = rand() * 2.0 - 1.0
            s = u * u + v * v
            if s != 0.0 and s < 1.0:
                break
        mul = math.sqrt((-2.0 * math.log(s)) / s)
        spare = v * mul
        return u * mul

    return gauss


# ------------------------------------------------------------------ utils

def _prod(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _ensure_tensor(v):
    if isinstance(v, Tensor):
        return v
    if isinstance(v, (int, float)):
        if not math.isfinite(v):
            raise ValueError('non-finite scalar: %r' % (v,))
        return Tensor([], np.array([v], dtype=np.float32))
    raise TypeError('expected a Tensor or number, got %r' % type(v))


def _unbroadcast(g, out_shape, in_shape):
    """Sum an output-shaped grad down to an input shape (right-aligned
    broadcasting reversed). Mirrors the JS index-map reduction."""
    extra = len(out_shape) - len(in_shape)
    if extra > 0:
        g = g.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(in_shape)
                 if s == 1 and out_shape[i + extra] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    return g


# ----------------------------------------------------------------- Tensor

class Tensor:
    """Flat float32 storage + autograd graph node (shape kept alongside,
    exactly like the JS Tensor backed by Float32Array)."""

    def __init__(self, shape, data=None, requires_grad=False, name=''):
        self.shape = [int(s) for s in shape]
        self.ndim = len(self.shape)
        self.size = _prod(self.shape)
        if data is None:
            self.data = np.zeros(self.size, dtype=np.float32)
        else:
            arr = np.array(data, dtype=np.float32).reshape(-1)
            if arr.size != self.size:
                raise ValueError('data size %d != tensor size %d' % (arr.size, self.size))
            self.data = arr
        self.grad = np.zeros(self.size, dtype=np.float32)
        self.requires_grad = bool(requires_grad)
        self.name = name
        self._prev = ()        # insertion-ordered parents (JS Set order)
        self._backward = None  # closure: out.grad -> parents.grad
        self._op = ''

    # ------------------------------------------------------ constructors

    @staticmethod
    def zeros(shape, **kw):
        return Tensor(shape, None, **kw)

    @staticmethod
    def ones(shape, **kw):
        """All-ones leaf (layerNorm gains). JS twin: Tensor.ones."""
        t = Tensor(shape, None, **kw)
        t.data = np.ones(t.size, dtype=np.float32)
        return t

    @staticmethod
    def randn(shape, std=1.0, seed=0, **kw):
        """Seeded, deterministic, bit-identical to JS Tensor.randn:
        data[i] = float32(gauss() * std) over mulberry32(seed)."""
        t = Tensor(shape, None, **kw)
        gauss = make_gaussian(mulberry32(seed & 0xFFFFFFFF))
        vals = [gauss() * std for _ in range(t.size)]
        t.data = np.array(vals, dtype=np.float32)
        return t

    def zeroGrad(self):
        self.grad.fill(0)
        return self

    # Pythonic alias, same behavior
    zero_grad = zeroGrad

    def item(self):
        if self.size != 1:
            raise ValueError('item(): only for size-1 tensors')
        return float(self.data[0])

    # ------------------------------------------------------- binary ops

    def add(self, other):
        b = _ensure_tensor(other)
        a = self
        out_shape = list(np.broadcast_shapes(tuple(a.shape), tuple(b.shape)))
        out = Tensor(out_shape)
        out._op = 'add'
        out.requires_grad = a.requires_grad or b.requires_grad
        A = a.data.reshape(a.shape).astype(np.float64)
        B = b.data.reshape(b.shape).astype(np.float64)
        out.data = (A + B).astype(np.float32).ravel()
        out._prev = (a, b)

        def bk():
            g = out.grad.reshape(out.shape).astype(np.float64)
            if a.requires_grad:
                a.grad += _unbroadcast(g, out.shape, a.shape).ravel()
            if b.requires_grad:
                b.grad += _unbroadcast(g, out.shape, b.shape).ravel()

        out._backward = bk
        return out

    def sub(self, other):
        b = _ensure_tensor(other)
        a = self
        out_shape = list(np.broadcast_shapes(tuple(a.shape), tuple(b.shape)))
        out = Tensor(out_shape)
        out._op = 'sub'
        out.requires_grad = a.requires_grad or b.requires_grad
        A = a.data.reshape(a.shape).astype(np.float64)
        B = b.data.reshape(b.shape).astype(np.float64)
        out.data = (A - B).astype(np.float32).ravel()
        out._prev = (a, b)

        def bk():
            g = out.grad.reshape(out.shape).astype(np.float64)
            if a.requires_grad:
                a.grad += _unbroadcast(g, out.shape, a.shape).ravel()
            if b.requires_grad:
                b.grad -= _unbroadcast(g, out.shape, b.shape).ravel()

        out._backward = bk
        return out

    def mul(self, other):
        b = _ensure_tensor(other)
        a = self
        out_shape = list(np.broadcast_shapes(tuple(a.shape), tuple(b.shape)))
        out = Tensor(out_shape)
        out._op = 'mul'
        out.requires_grad = a.requires_grad or b.requires_grad
        A = a.data.reshape(a.shape).astype(np.float64)
        B = b.data.reshape(b.shape).astype(np.float64)
        out.data = (A * B).astype(np.float32).ravel()
        out._prev = (a, b)

        def bk():
            g = out.grad.reshape(out.shape).astype(np.float64)
            # note: when a is b (x*x) both branches add into the same grad
            # array, giving the correct total 2*x*g
            if a.requires_grad:
                a.grad += _unbroadcast(g * B, out.shape, a.shape).ravel()
            if b.requires_grad:
                b.grad += _unbroadcast(g * A, out.shape, b.shape).ravel()

        out._backward = bk
        return out

    def matmul(self, other):
        """2-D matmul. JS twin uses a fixed i,k,j order with float32
        rounding per step; NumPy uses BLAS on float32 (different internal
        order — this is the accepted, MEASURED source of parity noise)."""
        a, b = self, other
        if not isinstance(b, Tensor):
            raise TypeError('matmul: needs a Tensor')
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError('matmul: 2-D only, got %s x %s' % (a.shape, b.shape))
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise ValueError('matmul: inner dims mismatch %s x %s' % (a.shape, b.shape))
        out = Tensor([M, N])
        out._op = 'matmul'
        out.requires_grad = a.requires_grad or b.requires_grad
        A = a.data.reshape(M, K)
        B = b.data.reshape(K, N)
        out.data = np.matmul(A, B).ravel()  # float32 BLAS
        out._prev = (a, b)

        def bk():
            G = out.grad.reshape(M, N)
            if a.requires_grad:
                a.grad += np.matmul(G, B.T).ravel()  # dA = G @ B^T
            if b.requires_grad:
                b.grad += np.matmul(A.T, G).ravel()  # dB = A^T @ G

        out._backward = bk
        return out

    def matmulT(self, other):
        """x @ w^T for 2-D x [M, K], w [O, K]: out[i, o] = sum_k x[i,k] * w[o,k].
        Forward = matmul against the transpose VIEW w.T (w is never copied).
        Backward: dx += G @ w, dw += G^T @ x. Used for the WEIGHT-TIED head:
        logits = lnf_out.matmulT(wte), wte stays [vocab, d_model]."""
        a, w = self, other
        if not isinstance(w, Tensor):
            raise TypeError('matmulT: needs a Tensor')
        if a.ndim != 2 or w.ndim != 2:
            raise ValueError('matmulT: 2-D only, got %s x %s' % (a.shape, w.shape))
        M, K = a.shape
        O, K2 = w.shape
        if K != K2:
            raise ValueError('matmulT: inner dims mismatch %s x %s' % (a.shape, w.shape))
        out = Tensor([M, O])
        out._op = 'matmulT'
        out.requires_grad = a.requires_grad or w.requires_grad
        A = a.data.reshape(M, K)
        W = w.data.reshape(O, K)
        out.data = np.matmul(A, W.T).ravel()  # W.T is a view; float32 BLAS
        out._prev = (a, w)

        def bk():
            G = out.grad.reshape(M, O)
            if a.requires_grad:
                a.grad += np.matmul(G, W).ravel()      # [M,O] @ [O,K]
            if w.requires_grad:
                w.grad += np.matmul(G.T, A).ravel()    # [O,M] @ [M,K]

        out._backward = bk
        return out

    # ----------------------------------------------------- task 4 ops

    def layerNorm(self, g, b, eps=1e-5):
        """LayerNorm over the LAST axis, BIASED variance (mean of squared
        deviations, GPT-2 style): y = g * (x - mu) / sqrt(var + eps) + b.
        Internal math float64, result rounded to float32 once (JS twin
        computes the identical formula row-by-row in float64)."""
        a = self
        g = _ensure_tensor(g)
        b = _ensure_tensor(b)
        if a.ndim < 1:
            raise ValueError('layerNorm: needs at least 1 axis')
        D = a.shape[-1]
        if g.size != D or b.size != D:
            raise ValueError('layerNorm: g/b must have %d elements' % D)
        R = a.size // D
        out = Tensor(list(a.shape))
        out._op = 'layerNorm'
        out.requires_grad = a.requires_grad or g.requires_grad or b.requires_grad
        x = a.data.reshape(R, D).astype(np.float64)
        g64 = g.data.reshape(D).astype(np.float64)
        b64 = b.data.reshape(D).astype(np.float64)
        mu = x.mean(axis=1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=1, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        xhat = xc * inv
        out.data = (g64 * xhat + b64).astype(np.float32).ravel()
        out._prev = (a, g, b)

        def bk():
            G = out.grad.reshape(R, D).astype(np.float64)
            if a.requires_grad:
                delta = G * g64
                m1 = delta.mean(axis=1, keepdims=True)
                m2 = (delta * xhat).mean(axis=1, keepdims=True)
                dx = (delta - m1 - xhat * m2) * inv
                a.grad += dx.ravel()
            if g.requires_grad:
                g.grad += (G * xhat).sum(axis=0)
            if b.requires_grad:
                b.grad += G.sum(axis=0)

        out._backward = bk
        return out

    def slice(self, ranges):
        """General slice: one range per axis; each range is None (full axis)
        or a [start, end) pair with 0 <= start < end <= dim. Backward
        scatter-adds the grad back into the source positions."""
        a = self
        if len(ranges) != a.ndim:
            raise ValueError('slice: need %d ranges, got %r' % (a.ndim, ranges))
        slices = []
        for d, r in enumerate(ranges):
            if r is None:
                slices.append(slice(None))
                continue
            s, e = int(r[0]), int(r[1])
            if not (0 <= s < e <= a.shape[d]):
                raise ValueError('slice: bad range [%d,%d) for axis %d (dim %d)'
                                 % (s, e, d, a.shape[d]))
            slices.append(slice(s, e))
        view = a.data.reshape(a.shape)[tuple(slices)]
        out = Tensor(list(view.shape))
        out._op = 'slice'
        out.requires_grad = a.requires_grad
        out.data = np.array(view, dtype=np.float32).ravel()  # copy
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            gfull = np.zeros(a.shape, dtype=np.float32)
            gfull[tuple(slices)] = out.grad.reshape(out.shape)
            a.grad += gfull.ravel()

        out._backward = bk
        return out

    def transpose(self, perm):
        """Axis permutation, NumPy convention: out[k] = in[perm[k]].
        Backward applies the inverse permutation."""
        a = self
        perm = [int(p) for p in perm]
        if sorted(perm) != list(range(a.ndim)):
            raise ValueError('transpose: bad perm %r for ndim %d' % (perm, a.ndim))
        view = a.data.reshape(a.shape).transpose(perm)
        out = Tensor(list(view.shape))
        out._op = 'transpose'
        out.requires_grad = a.requires_grad
        out.data = np.array(view, dtype=np.float32).ravel()  # copy
        inv = [0] * a.ndim
        for k, p in enumerate(perm):
            inv[p] = k
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            gback = out.grad.reshape(out.shape).transpose(inv)
            a.grad += np.array(gback, dtype=np.float32).ravel()

        out._backward = bk
        return out

    def embed(self, indices):
        """Embedding lookup (gather rows of this [vocab, D] matrix):
        out[r] = self[indices[r]] for a flat int array of length N ->
        out shape [N, D]. Backward SCATTER-ADDS into the (possibly repeated)
        source rows — the embedding-table sibling of slice(). Used for
        wte[token_id] in the transformer (docs/model.js mirrors this op)."""
        a = self
        if a.ndim != 2:
            raise ValueError('embed: needs a 2-D [vocab, D] tensor, got %s' % (a.shape,))
        V, D = a.shape
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError('embed: empty indices')
        if not np.all((idx >= 0) & (idx < V)):
            raise ValueError('embed: index out of range [0,%d)' % V)
        N = idx.size
        flat = (idx * D)[:, None] + np.arange(D, dtype=np.int64)[None, :]
        flat = flat.reshape(-1)  # source position of every output element
        out = Tensor([N, D])
        out._op = 'embed'
        out.requires_grad = a.requires_grad
        out.data = a.data[flat].astype(np.float32)  # gather copy
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            np.add.at(a.grad, flat, out.grad.astype(np.float64))

        out._backward = bk
        return out

    def bmm(self, other):
        """Batched matmul: [N, M, K] @ [N, K, P] -> [N, M, P]. JS twin uses
        fixed n,i,k,j orders (float32 rounding per step); NumPy batched BLAS
        — same accepted order-of-accumulation parity noise as matmul."""
        a, b = self, other
        if not isinstance(b, Tensor):
            raise TypeError('bmm: needs a Tensor')
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError('bmm: 3-D only, got %s x %s' % (a.shape, b.shape))
        N, M, K = a.shape
        N2, K2, P = b.shape
        if N != N2 or K != K2:
            raise ValueError('bmm: shape mismatch %s x %s' % (a.shape, b.shape))
        out = Tensor([N, M, P])
        out._op = 'bmm'
        out.requires_grad = a.requires_grad or b.requires_grad
        A = a.data.reshape(N, M, K)
        B = b.data.reshape(N, K, P)
        out.data = np.matmul(A, B).ravel()  # float32 batched BLAS
        out._prev = (a, b)

        def bk():
            G = out.grad.reshape(N, M, P)
            if a.requires_grad:
                a.grad += np.matmul(G, B.transpose(0, 2, 1)).ravel()
            if b.requires_grad:
                b.grad += np.matmul(A.transpose(0, 2, 1), G).ravel()

        out._backward = bk
        return out

    # ------------------------------------------------------ unary ops

    def relu(self):
        a = self
        out = Tensor(list(a.shape))
        out._op = 'relu'
        out.requires_grad = a.requires_grad
        x = a.data.reshape(a.shape).astype(np.float64)
        out.data = np.where(x > 0, x, 0.0).astype(np.float32).ravel()
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            g = out.grad.reshape(a.shape).astype(np.float64)
            a.grad += np.where(x > 0, g, 0.0).ravel()

        out._backward = bk
        return out

    def tanh(self):
        a = self
        out = Tensor(list(a.shape))
        out._op = 'tanh'
        out.requires_grad = a.requires_grad
        x = a.data.reshape(a.shape).astype(np.float64)
        t = np.tanh(x)
        out.data = t.astype(np.float32).ravel()
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            g = out.grad.reshape(a.shape).astype(np.float64)
            # JS reads the STORED float32 tanh values here (t = out.data)
            t32 = out.data.reshape(a.shape).astype(np.float64)
            a.grad += (g * (1.0 - t32 * t32)).ravel()

        out._backward = bk
        return out

    def gelu(self):
        """GELU tanh approximation, identical constants to the JS engine:
        0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
        a = self
        out = Tensor(list(a.shape))
        out._op = 'gelu'
        out.requires_grad = a.requires_grad
        x = a.data.reshape(a.shape).astype(np.float64)
        inner = GELU_C * (x + 0.044715 * x * x * x)
        t = np.tanh(inner)
        out.data = (0.5 * x * (1.0 + t)).astype(np.float32).ravel()
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            g = out.grad.reshape(a.shape).astype(np.float64)
            d_inner = GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
            a.grad += (g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner)).ravel()

        out._backward = bk
        return out

    def softmax(self):
        """Rowwise softmax over [N, D]. Max-subtracted. Backward reads the
        STORED float32 probs (like the JS engine): gx = s * (g - sum(g*s))."""
        a = self
        if a.ndim != 2:
            raise ValueError('softmax: expects [N, D], got %s' % (a.shape,))
        N, D = a.shape
        out = Tensor([N, D])
        out._op = 'softmax'
        out.requires_grad = a.requires_grad
        x = a.data.reshape(N, D).astype(np.float64)
        m = x.max(axis=1, keepdims=True)
        e = np.exp(x - m)
        se = e.sum(axis=1, keepdims=True)
        # JS stores raw e into the f32 output FIRST, then o *= inv — mirror
        # that double rounding exactly.
        e32 = e.astype(np.float32)
        inv = 1.0 / se                              # JS: inv = 1/se
        o = e32.astype(np.float64) * inv            # JS: o[b+j] *= inv
        out.data = o.astype(np.float32).ravel()
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            g = out.grad.reshape(N, D).astype(np.float64)
            oo = out.data.reshape(N, D).astype(np.float64)  # stored f32 probs
            dot = (g * oo).sum(axis=1, keepdims=True)
            a.grad += (oo * (g - dot)).ravel()

        out._backward = bk
        return out

    def crossEntropy(self, targets):
        """FUSED softmax + cross-entropy over logits [N, C]. Forward: mean
        over rows of (logsumexp - logit[target]), row max subtracted first
        so log() never sees 0. Backward: (p - onehot) / N with p = the
        STORED NORMALIZED float32 probs (the Task 1 bug class)."""
        a = self
        if a.ndim != 2:
            raise ValueError('crossEntropy: expects [N, C], got %s' % (a.shape,))
        N, C = a.shape
        if targets is None or len(targets) != N:
            raise ValueError('crossEntropy: targets must have length %d' % N)
        tgt = np.asarray(targets, dtype=np.int64).reshape(-1)
        if not np.all((tgt >= 0) & (tgt < C)):
            raise ValueError('crossEntropy: target out of range')
        out = Tensor([1])
        out._op = 'crossEntropy'
        out.requires_grad = a.requires_grad
        x = a.data.reshape(N, C).astype(np.float64)
        probs = np.zeros((N, C), dtype=np.float32)  # saved for backward
        total = 0.0
        for i in range(N):
            m = x[i].max()
            e = np.exp(x[i] - m)
            se = e.sum()
            # JS: probs[b+j] = e (f32 store) then probs[b+j] *= inv —
            # mirror the intermediate f32 rounding of e exactly.
            e32 = e.astype(np.float32)
            inv = 1.0 / se
            probs[i] = (e32.astype(np.float64) * inv).astype(np.float32)
            total += (m + math.log(se)) - x[i, tgt[i]]
        out.data[0] = np.float32(total / N)
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            g = float(out.grad[0])
            for i in range(N):
                row = probs[i].astype(np.float64)
                row[tgt[i]] -= 1.0
                a.grad[i * C:(i + 1) * C] += g * row / N  # one f64->f32 rounding, like JS

        out._backward = bk
        return out

    # -------------------------------------------------- reduce / shape

    def sum(self):
        a = self
        out = Tensor([1])
        out._op = 'sum'
        out.requires_grad = a.requires_grad
        out.data[0] = np.float32(a.data.astype(np.float64).sum())
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            a.grad += np.float64(out.grad[0])

        out._backward = bk
        return out

    def scale(self, c):
        if not math.isfinite(c):
            raise ValueError('non-finite scalar: %r' % (c,))
        a = self
        out = Tensor(list(a.shape))
        out._op = 'scale'
        out.requires_grad = a.requires_grad
        out.data = (a.data.reshape(a.shape).astype(np.float64) * c).astype(np.float32).ravel()
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            # out.grad is already flat (JS: self.grad[i] += g[i] * c on the
            # flat arrays); the old reshape(a.shape) here broke for N-D
            # tensors (a.grad is flat — += needs an (18,) not a (2,3,3)).
            a.grad += out.grad.astype(np.float64) * c

        out._backward = bk
        return out

    def reshape(self, new_shape):
        """Reshape (copies; size preserved). One -1 is allowed and inferred."""
        a = self
        target = [int(s) for s in new_shape]
        neg_at = -1
        known = 1
        for i, s in enumerate(target):
            if s == -1:
                if neg_at >= 0:
                    raise ValueError('reshape: only one -1 allowed')
                neg_at = i
            else:
                if s <= 0:
                    raise ValueError('reshape: bad dim %d' % s)
                known *= s
        if neg_at >= 0:
            if known == 0 or a.size % known != 0:
                raise ValueError('reshape: cannot infer -1')
            target[neg_at] = a.size // known
        if _prod(target) != a.size:
            raise ValueError('reshape: %d elements cannot fill %r' % (a.size, target))
        out = Tensor(target)
        out.data = np.array(a.data, dtype=np.float32)  # copy, flat
        out._op = 'reshape'
        out.requires_grad = a.requires_grad
        out._prev = (a,)

        def bk():
            if not a.requires_grad:
                return
            a.grad += out.grad

        out._backward = bk
        return out

    # -------------------------------------------------------- backward

    def backward(self):
        """Seeds this tensor's grad by 1 and pulls the whole graph backward
        in reverse topological order (iterative DFS — same as JS)."""
        if self.size != 1:
            raise ValueError('backward(): seed must be a scalar loss tensor, got %r' % (self.shape,))
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            t, processed = stack.pop()
            if processed:
                topo.append(t)
                continue
            if id(t) in visited:
                continue
            visited.add(id(t))
            stack.append((t, True))
            for p in t._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        self.grad[0] += np.float32(1.0)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()
        return self


def heInit(fan_in, fan_out, seed, **kw):
    """N(0, sqrt(2/fanIn)), shape [fanIn, fanOut] — feeds straight into
    matmul(x, W). Bit-identical to JS heInit for the same seed."""
    return Tensor.randn([fan_in, fan_out], math.sqrt(2 / fan_in), seed, **kw)


# camelCase aliases so either style reads naturally
he_init = heInit
makeGaussian = make_gaussian
