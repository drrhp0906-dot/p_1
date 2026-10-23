/* ============================================================================
   p_1 · docs/tensor.js — the autograd engine
   ----------------------------------------------------------------------------
   Reverse-mode automatic differentiation in pure JavaScript. Zero
   dependencies. Runs in the browser (window.P1) and in Node (module.exports).

   DESIGN NOTES (the short version)

   - Tensor: shape[], data (Float32Array), grad (Float32Array), requiresGrad.
     Every op allocates a fresh output tensor — no in-place mutation — so
     graphs are immutable and a built graph can be kept alive (e.g. for
     sampling) while training continues elsewhere.

   - Backward: iterative post-order DFS over each tensor's _prev set gives a
     topological order (no recursion, so deep graphs cannot blow the stack).
     Grads ACCUMULATE (+=) into parents, so a tensor consumed twice — the
     weight-tied output head planned for Task 4, or w.mul(w) — receives every
     contribution. The whole transformer will lean on this property.

   - Broadcasting: right-aligned, NumPy-style (bias [1,D] over [M,D], a JS
     number over anything). Backward "unbroadcasts" by summing grads over the
     broadcast dimensions using integer index maps built once at forward time.

   - matmul is 2-D only, with a FIXED i,k,j reduction order and a fixed
     i,k,j order in both backward kernels. The Python twin (Task 4,
     colab/tensor_py.py) must copy these loop orders exactly — float32
     accumulation order is the #1 cause of parity test failures.

   - crossEntropy(logits, targets) is the FUSED softmax+loss: one stable
     forward (max-subtracted logsumexp, so log() can never see 0), one
     backward ((p - onehot)/N). Rowwise softmax also exists standalone for
     inference and sampling.

   - All intermediate math runs in float64 and is stored into Float32Array
     after each op. That is what "float32 engine" means here; the gradient
     check in tensor_test.html is calibrated against exactly this.

   API
     new Tensor(shape, data?, opts?) · Tensor.zeros(shape, opts?)
     Tensor.randn(shape, std, seed, opts?)        seeded, deterministic
     t.add(u) t.sub(u) t.mul(u) t.matmul(u)       u may be a Tensor or number
     t.relu() t.tanh() t.gelu() t.softmax()       elementwise / rowwise
     t.sum() t.scale(c) t.reshape(...shape)
     t.crossEntropy(targets)                      fused; targets = int array
     t.backward() t.zeroGrad() t.item()
     heInit(fanIn, fanOut, seed, opts?)           std = sqrt(2 / fanIn)
     mulberry32(seed) · makeGaussian(rand)        the RNG plumbing

   TASK 4 ADDITIONS (the transformer ops — same names/semantics in
   colab/tensor_py.py, added for architecture parity):
     t.layerNorm(g, b, eps=1e-5)                  last axis, BIASED variance
     t.slice(ranges)                              one [start,end) per axis
     t.transpose(perm)                            out[k] = in[perm[k]]
     t.bmm(u)                                     batched [N,M,K]@[N,K,P]
     t.matmulT(w)                                 x @ w^T (weight-tied head)
     t.embed(indices)                             gather rows of [vocab,D]
                                                  (wte lookup, scatter-add grad)
   ============================================================================ */

(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.P1 = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const VERSION = '1.1.0';
  const GELU_C = Math.sqrt(2 / Math.PI); // tanh-approximation GELU (GPT-2 style)

  /* ---------------------------------------------------------------- utils */

  function prod(shape) {
    let n = 1;
    for (let i = 0; i < shape.length; i++) n *= shape[i];
    return n;
  }

  function stridesOf(shape) {
    const s = new Array(shape.length);
    let acc = 1;
    for (let i = shape.length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
    return s;
  }

  function sameShape(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  function broadcastShapes(a, b) {
    const L = Math.max(a.length, b.length);
    const out = new Array(L);
    for (let i = 0; i < L; i++) {
      const da = i >= L - a.length ? a[i - (L - a.length)] : 1;
      const db = i >= L - b.length ? b[i - (L - b.length)] : 1;
      if (da !== db && da !== 1 && db !== 1) {
        throw new Error('shapes [' + a + '] and [' + b + '] cannot broadcast');
      }
      out[i] = Math.max(da, db);
    }
    return out;
  }

  /* Integer index maps for a broadcast binary op: for every linear output
     index o, maps.a[o] / maps.b[o] give the source element. Built once at
     forward time, reused by backward as the "unbroadcast" reduction. */
  function buildBroadcastMaps(outShape, aShape, bShape) {
    const L = outShape.length;
    const aP = new Array(L).fill(1), bP = new Array(L).fill(1);
    for (let i = 0; i < aShape.length; i++) aP[L - aShape.length + i] = aShape[i];
    for (let i = 0; i < bShape.length; i++) bP[L - bShape.length + i] = bShape[i];
    const aS = stridesOf(aP), bS = stridesOf(bP);
    const aB = new Array(L), bB = new Array(L);
    for (let d = 0; d < L; d++) {
      aB[d] = aP[d] === 1 ? 0 : aS[d];
      bB[d] = bP[d] === 1 ? 0 : bS[d];
    }
    const size = prod(outShape);
    const aMap = new Int32Array(size), bMap = new Int32Array(size);
    const cnt = new Int32Array(L);
    let aLin = 0, bLin = 0;
    for (let o = 0; o < size; o++) {
      aMap[o] = aLin; bMap[o] = bLin;
      for (let d = L - 1; d >= 0; d--) {
        if (++cnt[d] < outShape[d]) { aLin += aB[d]; bLin += bB[d]; break; }
        cnt[d] = 0;
        aLin -= aB[d] * (outShape[d] - 1);
        bLin -= bB[d] * (outShape[d] - 1);
      }
    }
    return { a: aMap, b: bMap };
  }

  /* Row-major odometer + gather map for the Task 4 structural ops
     (transpose / slice): map[o] = srcOf(coords) for every linear output
     index. Forward gathers via map, backward scatter-adds via map (the map
     is injective, so accumulation is exact). */
  function buildGatherMap(outShape, srcOf) {
    const L = outShape.length, size = prod(outShape);
    const map = new Int32Array(size);
    const coords = new Int32Array(L);
    for (let o = 0; o < size; o++) {
      map[o] = srcOf(coords);
      for (let d = L - 1; d >= 0; d--) {
        if (++coords[d] < outShape[d]) break;
        coords[d] = 0;
      }
    }
    return map;
  }

  /* ------------------------------------------------------------ the RNG
     mulberry32: tiny seeded PRNG, trivially portable to Python for the
     parity fixtures of Tasks 3-4. makeGaussian: Marsaglia polar method. */

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeGaussian(rand) {
    let spare = null;
    return function () {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u, v, s;
      do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; }
      while (s === 0 || s >= 1);
      const mul = Math.sqrt((-2 * Math.log(s)) / s);
      spare = v * mul;
      return u * mul;
    };
  }

  /* ------------------------------------------------------------- Tensor */

  class Tensor {
    constructor(shape, data, opts = {}) {
      this.shape = shape.slice();
      this.ndim = this.shape.length;
      this.size = prod(this.shape);
      if (data instanceof Float32Array && data.length === this.size) {
        this.data = data;
      } else {
        this.data = new Float32Array(this.size);
        if (data) this.data.set(data);
      }
      this.grad = new Float32Array(this.size);
      this.requiresGrad = !!opts.requiresGrad;
      this.name = opts.name || '';
      this._prev = null;      // Set of parent Tensors (graph edges)
      this._backward = null;  // closure: out.grad -> parents.grad
      this._op = '';          // debug label
    }

    static zeros(shape, opts) { return new Tensor(shape, null, opts); }

    /* All-ones leaf (layerNorm gains). Python twin: Tensor.ones. */
    static ones(shape, opts) {
      const t = new Tensor(shape, null, opts);
      for (let i = 0; i < t.size; i++) t.data[i] = 1.0;
      return t;
    }

    static randn(shape, std, seed, opts = {}) {
      const t = new Tensor(shape, null, opts);
      const gauss = makeGaussian(mulberry32(seed >>> 0));
      for (let i = 0; i < t.size; i++) t.data[i] = gauss() * std;
      return t;
    }

    zeroGrad() { this.grad.fill(0); return this; }

    item() {
      if (this.size !== 1) throw new Error('item(): only for size-1 tensors');
      return this.data[0];
    }

    /* ------------------------------------------------------- binary ops */

    add(other) {
      const b = ensureTensor(other), a = this;
      const outShape = broadcastShapes(a.shape, b.shape);
      const out = new Tensor(outShape);
      out._op = 'add';
      out.requiresGrad = a.requiresGrad || b.requiresGrad;
      if (sameShape(a.shape, outShape) && sameShape(b.shape, outShape)) {
        for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] + b.data[i];
        out._prev = new Set([a, b]);
        out._backward = () => {
          if (a.requiresGrad) for (let i = 0; i < out.size; i++) a.grad[i] += out.grad[i];
          if (b.requiresGrad) for (let i = 0; i < out.size; i++) b.grad[i] += out.grad[i];
        };
      } else {
        const m = buildBroadcastMaps(outShape, a.shape, b.shape);
        for (let i = 0; i < out.size; i++) out.data[i] = a.data[m.a[i]] + b.data[m.b[i]];
        out._prev = new Set([a, b]);
        out._backward = () => {
          if (a.requiresGrad) for (let i = 0; i < out.size; i++) a.grad[m.a[i]] += out.grad[i];
          if (b.requiresGrad) for (let i = 0; i < out.size; i++) b.grad[m.b[i]] += out.grad[i];
        };
      }
      return out;
    }

    sub(other) {
      const b = ensureTensor(other), a = this;
      const outShape = broadcastShapes(a.shape, b.shape);
      const out = new Tensor(outShape);
      out._op = 'sub';
      out.requiresGrad = a.requiresGrad || b.requiresGrad;
      if (sameShape(a.shape, outShape) && sameShape(b.shape, outShape)) {
        for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] - b.data[i];
        out._prev = new Set([a, b]);
        out._backward = () => {
          if (a.requiresGrad) for (let i = 0; i < out.size; i++) a.grad[i] += out.grad[i];
          if (b.requiresGrad) for (let i = 0; i < out.size; i++) b.grad[i] -= out.grad[i];
        };
      } else {
        const m = buildBroadcastMaps(outShape, a.shape, b.shape);
        for (let i = 0; i < out.size; i++) out.data[i] = a.data[m.a[i]] - b.data[m.b[i]];
        out._prev = new Set([a, b]);
        out._backward = () => {
          if (a.requiresGrad) for (let i = 0; i < out.size; i++) a.grad[m.a[i]] += out.grad[i];
          if (b.requiresGrad) for (let i = 0; i < out.size; i++) b.grad[m.b[i]] -= out.grad[i];
        };
      }
      return out;
    }

    mul(other) {
      const b = ensureTensor(other), a = this;
      const outShape = broadcastShapes(a.shape, b.shape);
      const out = new Tensor(outShape);
      out._op = 'mul';
      out.requiresGrad = a.requiresGrad || b.requiresGrad;
      if (sameShape(a.shape, outShape) && sameShape(b.shape, outShape)) {
        for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] * b.data[i];
        out._prev = new Set([a, b]);
        out._backward = () => {
          // note: when a === b (x*x) both loops hit the same grad array,
          // giving the correct total 2*x*g
          if (a.requiresGrad) for (let i = 0; i < out.size; i++) a.grad[i] += out.grad[i] * b.data[i];
          if (b.requiresGrad) for (let i = 0; i < out.size; i++) b.grad[i] += out.grad[i] * a.data[i];
        };
      } else {
        const m = buildBroadcastMaps(outShape, a.shape, b.shape);
        for (let i = 0; i < out.size; i++) out.data[i] = a.data[m.a[i]] * b.data[m.b[i]];
        out._prev = new Set([a, b]);
        out._backward = () => {
          if (a.requiresGrad) for (let i = 0; i < out.size; i++) a.grad[m.a[i]] += out.grad[i] * b.data[m.b[i]];
          if (b.requiresGrad) for (let i = 0; i < out.size; i++) b.grad[m.b[i]] += out.grad[i] * a.data[m.a[i]];
        };
      }
      return out;
    }

    /* 2-D matmul, fixed i,k,j reduction order — the Python twin MUST copy
       this order (and the backward orders below) for float32 parity.
       PERF (Task 6): the inner j loop is unrolled 4-wide. Every output
       element still accumulates av*B[·,j] in the SAME k-ascending order
       (the four lanes are independent), so results are BIT-IDENTICAL to
       the plain loop — verified by the Task 4 fixture parity test
       (diffs unchanged to all printed digits). ~1.6x faster in V8. */
    matmul(other) {
      const a = this, b = other;
      if (!(b instanceof Tensor)) throw new Error('matmul: needs a Tensor');
      if (a.ndim !== 2 || b.ndim !== 2) {
        throw new Error('matmul: 2-D only, got [' + a.shape + '] x [' + b.shape + ']');
      }
      const M = a.shape[0], K = a.shape[1];
      if (b.shape[0] !== K) {
        throw new Error('matmul: inner dims mismatch [' + a.shape + '] x [' + b.shape + ']');
      }
      const N = b.shape[1];
      const out = new Tensor([M, N]);
      const A = a.data, B = b.data, O = out.data;
      for (let i = 0; i < M; i++) {
        const ob = i * N, ab = i * K;
        for (let j = 0; j < N; j++) O[ob + j] = 0.0;
        for (let k = 0; k < K; k++) {
          const av = A[ab + k], bb = k * N;
          let j = 0;
          for (; j + 4 <= N; j += 4) {
            O[ob + j]     += av * B[bb + j];
            O[ob + j + 1] += av * B[bb + j + 1];
            O[ob + j + 2] += av * B[bb + j + 2];
            O[ob + j + 3] += av * B[bb + j + 3];
          }
          for (; j < N; j++) O[ob + j] += av * B[bb + j];
        }
      }
      out._op = 'matmul';
      out.requiresGrad = a.requiresGrad || b.requiresGrad;
      out._prev = new Set([a, b]);
      out._backward = () => {
        const G = out.grad;
        if (a.requiresGrad) {
          // dA[i,k] = sum_j G[i,j] * B[k,j]
          for (let i = 0; i < M; i++) {
            const ob = i * N, ab = i * K;
            for (let k = 0; k < K; k++) {
              let acc = 0; const kb = k * N;
              for (let j = 0; j < N; j++) acc += G[ob + j] * B[kb + j];
              a.grad[ab + k] += acc;
            }
          }
        }
        if (b.requiresGrad) {
          // dB[k,j] = sum_i A[i,k] * G[i,j]   (4-wide unroll, bit-identical)
          for (let i = 0; i < M; i++) {
            const ob = i * N, ab = i * K;
            for (let k = 0; k < K; k++) {
              const av = A[ab + k], kb = k * N;
              let j = 0;
              for (; j + 4 <= N; j += 4) {
                b.grad[kb + j]     += av * G[ob + j];
                b.grad[kb + j + 1] += av * G[ob + j + 1];
                b.grad[kb + j + 2] += av * G[ob + j + 2];
                b.grad[kb + j + 3] += av * G[ob + j + 3];
              }
              for (; j < N; j++) b.grad[kb + j] += av * G[ob + j];
            }
          }
        }
      };
      return out;
    }

    /* ----------------------------------------------------- unary ops */

    relu() {
      const self = this;
      const out = new Tensor(self.shape);
      const x = self.data, o = out.data;
      for (let i = 0; i < self.size; i++) o[i] = x[i] > 0 ? x[i] : 0;
      out._op = 'relu';
      out.requiresGrad = self.requiresGrad;
      out._prev = new Set([self]);
      out._backward = () => {
        if (!self.requiresGrad) return;
        const g = out.grad;
        for (let i = 0; i < self.size; i++) if (x[i] > 0) self.grad[i] += g[i];
      };
      return out;
    }

    tanh() {
      const self = this;
      const out = new Tensor(self.shape);
      const x = self.data, o = out.data;
      for (let i = 0; i < self.size; i++) o[i] = Math.tanh(x[i]);
      out._op = 'tanh';
      out.requiresGrad = self.requiresGrad;
      out._prev = new Set([self]);
      out._backward = () => {
        if (!self.requiresGrad) return;
        const g = out.grad, t = out.data;
        for (let i = 0; i < self.size; i++) self.grad[i] += g[i] * (1 - t[i] * t[i]);
      };
      return out;
    }

    /* GELU with tanh approximation: 0.5x(1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
       The Python twin must use the identical formula (Task 4 parity). */
    gelu() {
      const self = this;
      const out = new Tensor(self.shape);
      const x = self.data, o = out.data;
      for (let i = 0; i < self.size; i++) {
        const v = x[i];
        o[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
      }
      out._op = 'gelu';
      out.requiresGrad = self.requiresGrad;
      out._prev = new Set([self]);
      out._backward = () => {
        if (!self.requiresGrad) return;
        const g = out.grad;
        for (let i = 0; i < self.size; i++) {
          const v = x[i];
          const t = Math.tanh(GELU_C * (v + 0.044715 * v * v * v));
          const dInner = GELU_C * (1 + 3 * 0.044715 * v * v);
          self.grad[i] += g[i] * (0.5 * (1 + t) + 0.5 * v * (1 - t * t) * dInner);
        }
      };
      return out;
    }

    /* Rowwise softmax over a [N, D] tensor. Max-subtracted, so exp() never
       overflows. Backward: gx = s * (g - sum(g*s)) per row. */
    softmax() {
      const self = this;
      if (self.ndim !== 2) throw new Error('softmax: expects [N, D], got [' + self.shape + ']');
      const N = self.shape[0], D = self.shape[1];
      const out = new Tensor([N, D]);
      const x = self.data, o = out.data;
      for (let i = 0; i < N; i++) {
        const b = i * D;
        let m = -Infinity;
        for (let j = 0; j < D; j++) if (x[b + j] > m) m = x[b + j];
        let se = 0;
        for (let j = 0; j < D; j++) { const e = Math.exp(x[b + j] - m); o[b + j] = e; se += e; }
        const inv = 1 / se;
        for (let j = 0; j < D; j++) o[b + j] *= inv;
      }
      out._op = 'softmax';
      out.requiresGrad = self.requiresGrad;
      out._prev = new Set([self]);
      out._backward = () => {
        if (!self.requiresGrad) return;
        const g = out.grad;
        for (let i = 0; i < N; i++) {
          const b = i * D;
          let dot = 0;
          for (let j = 0; j < D; j++) dot += g[b + j] * o[b + j];
          for (let j = 0; j < D; j++) self.grad[b + j] += o[b + j] * (g[b + j] - dot);
        }
      };
      return out;
    }

    /* FUSED softmax + cross-entropy over logits [N, C] and integer targets.
       Forward: mean over rows of (logsumexp - logit[target]); the row max is
       subtracted first, so the summed exponentials always include 1.0 and
       log() can never receive 0. Backward: (softmax - onehot) / N, one pass,
       no separate softmax node — that is the point of fusing it. */
    crossEntropy(targets) {
      const self = this;
      if (self.ndim !== 2) throw new Error('crossEntropy: expects [N, C], got [' + self.shape + ']');
      const N = self.shape[0], C = self.shape[1];
      if (!targets || targets.length !== N) {
        throw new Error('crossEntropy: targets must have length ' + N);
      }
      const tgt = new Int32Array(N);
      for (let i = 0; i < N; i++) {
        const t = targets[i] | 0;
        if (t < 0 || t >= C || t !== targets[i]) {
          throw new Error('crossEntropy: target ' + i + ' out of range: ' + targets[i]);
        }
        tgt[i] = t;
      }
      const out = new Tensor([1]);
      const probs = new Float32Array(N * C); // saved for backward
      const x = self.data;
      let total = 0;
      for (let i = 0; i < N; i++) {
        const b = i * C;
        let m = -Infinity;
        for (let j = 0; j < C; j++) if (x[b + j] > m) m = x[b + j];
        let se = 0;
        for (let j = 0; j < C; j++) { const e = Math.exp(x[b + j] - m); probs[b + j] = e; se += e; }
        const inv = 1 / se;
        for (let j = 0; j < C; j++) probs[b + j] *= inv; // normalize AFTER accumulating raw se
        total += (m + Math.log(se)) - x[b + tgt[i]];
      }
      out.data[0] = total / N;
      out._op = 'crossEntropy';
      out.requiresGrad = self.requiresGrad;
      out._prev = new Set([self]);
      out._backward = () => {
        if (!self.requiresGrad) return;
        const g = out.grad[0];
        for (let i = 0; i < N; i++) {
          const b = i * C;
          for (let j = 0; j < C; j++) {
            self.grad[b + j] += g * (probs[b + j] - (j === tgt[i] ? 1 : 0)) / N;
          }
        }
      };
      return out;
    }

    /* -------------------------------------------------- reduce / shape */

    sum() {
      const self = this;
      const out = new Tensor([1]);
      let s = 0;
      for (let i = 0; i < self.size; i++) s += self.data[i];
      out.data[0] = s;
      out._op = 'sum';
      out.requiresGrad = self.requiresGrad;
      out._prev = new Set([self]);
      out._backward = () => {
        if (!self.requiresGrad) return;
        const g = out.grad[0];
        for (let i = 0; i < self.size; i++) self.grad[i] += g;
      };
      return out;
    }

    scale(c) {
      const self = this;
      const out = new Tensor(self.shape);
      for (let i = 0; i < self.size; i++) out.data[i] = self.data[i] * c;
      out._op = 'scale';
      out.requiresGrad = self.requiresGrad;
      out._prev = new Set([self]);
      out._backward = () => {
        if (!self.requiresGrad) return;
        const g = out.grad;
        for (let i = 0; i < self.size; i++) self.grad[i] += g[i] * c;
      };
      return out;
    }

    /* Reshape (copies; size preserved). One -1 is allowed and inferred. */
    reshape(newShape) {
      const self = this;
      const target = newShape.slice();
      let negAt = -1, known = 1;
      for (let i = 0; i < target.length; i++) {
        if (target[i] === -1) {
          if (negAt >= 0) throw new Error('reshape: only one -1 allowed');
          negAt = i;
        } else {
          if (!Number.isInteger(target[i]) || target[i] <= 0) {
            throw new Error('reshape: bad dim ' + target[i]);
          }
          known *= target[i];
        }
      }
      if (negAt >= 0) {
        if (known === 0 || self.size % known !== 0) throw new Error('reshape: cannot infer -1');
        target[negAt] = self.size / known;
      }
      if (prod(target) !== self.size) {
        throw new Error('reshape: ' + self.size + ' elements cannot fill [' + target + ']');
      }
      const out = new Tensor(target);
      out.data.set(self.data);
      out._op = 'reshape';
      out.requiresGrad = self.requiresGrad;
      out._prev = new Set([self]);
      out._backward = () => {
        if (!self.requiresGrad) return;
        const g = out.grad;
        for (let i = 0; i < self.size; i++) self.grad[i] += g[i];
      };
      return out;
    }

    /* ============================== TASK 4 OPS ============================== */

    /* LayerNorm over the LAST axis: y = g * (x - mu) / sqrt(var + eps) + b.
       BIASED variance (mean of squared deviations, GPT-2 style); eps default
       1e-5. Internal math in float64, stored to float32 like every op.
       Backward (d = grad * g):
         dx  = inv * (d - mean(d) - xhat * mean(d * xhat))
         dg  = sum_rows(grad * xhat)     db = sum_rows(grad)
       The Python twin (colab/tensor_py.py) implements the identical formula. */
    layerNorm(g, b, eps) {
      const a = this;
      if (!(g instanceof Tensor) || !(b instanceof Tensor)) {
        throw new Error('layerNorm: g and b must be Tensors');
      }
      if (eps === undefined) eps = 1e-5;
      if (a.ndim < 1) throw new Error('layerNorm: needs at least 1 axis');
      const D = a.shape[a.ndim - 1];
      if (g.size !== D || b.size !== D) {
        throw new Error('layerNorm: g/b must have ' + D + ' elements');
      }
      const R = a.size / D;
      const x = a.data, gv = g.data, bv = b.data;
      const out = new Tensor(a.shape);
      const o = out.data;
      for (let r = 0; r < R; r++) {
        const off = r * D;
        let mu = 0;
        for (let j = 0; j < D; j++) mu += x[off + j];
        mu /= D;
        let va = 0;
        for (let j = 0; j < D; j++) { const d = x[off + j] - mu; va += d * d; }
        va /= D;
        const inv = 1 / Math.sqrt(va + eps);
        for (let j = 0; j < D; j++) o[off + j] = gv[j] * ((x[off + j] - mu) * inv) + bv[j];
      }
      out._op = 'layerNorm';
      out.requiresGrad = a.requiresGrad || g.requiresGrad || b.requiresGrad;
      out._prev = new Set([a, g, b]);
      out._backward = () => {
        const G = out.grad;
        if (a.requiresGrad) {
          const delta = new Float64Array(D);
          for (let r = 0; r < R; r++) {
            const off = r * D;
            let mu = 0;
            for (let j = 0; j < D; j++) mu += x[off + j];
            mu /= D;
            let va = 0;
            for (let j = 0; j < D; j++) { const d = x[off + j] - mu; va += d * d; }
            va /= D;
            const inv = 1 / Math.sqrt(va + eps);
            let m1 = 0, m2 = 0;
            for (let j = 0; j < D; j++) {
              const dl = G[off + j] * gv[j];
              delta[j] = dl; m1 += dl;
              m2 += dl * ((x[off + j] - mu) * inv);
            }
            m1 /= D; m2 /= D;
            for (let j = 0; j < D; j++) {
              const xh = (x[off + j] - mu) * inv;
              a.grad[off + j] += (delta[j] - m1 - xh * m2) * inv;
            }
          }
        }
        if (g.requiresGrad) {
          const acc = new Float64Array(D);
          for (let r = 0; r < R; r++) {
            const off = r * D;
            let mu = 0;
            for (let j = 0; j < D; j++) mu += x[off + j];
            mu /= D;
            let va = 0;
            for (let j = 0; j < D; j++) { const d = x[off + j] - mu; va += d * d; }
            va /= D;
            const inv = 1 / Math.sqrt(va + eps);
            for (let j = 0; j < D; j++) acc[j] += G[off + j] * ((x[off + j] - mu) * inv);
          }
          for (let j = 0; j < D; j++) g.grad[j] += acc[j];
        }
        if (b.requiresGrad) {
          const acc = new Float64Array(D);
          for (let r = 0; r < R; r++) {
            const off = r * D;
            for (let j = 0; j < D; j++) acc[j] += G[off + j];
          }
          for (let j = 0; j < D; j++) b.grad[j] += acc[j];
        }
      };
      return out;
    }

    /* General slice: one range per axis, null = keep the whole axis. Each
       range is [start, end) with 0 <= start < end <= dim. Forward gathers
       through a prebuilt index map; backward scatter-adds through the same
       map (exact — the map is injective). */
    slice(ranges) {
      const a = this;
      if (!Array.isArray(ranges) || ranges.length !== a.ndim) {
        throw new Error('slice: need ' + a.ndim + ' ranges, got ' + JSON.stringify(ranges));
      }
      const starts = new Array(a.ndim), outShape = new Array(a.ndim);
      for (let d = 0; d < a.ndim; d++) {
        const r = ranges[d];
        if (r === null || r === undefined) { starts[d] = 0; outShape[d] = a.shape[d]; continue; }
        const s = r[0], e = r[1];
        if (!Number.isInteger(s) || !Number.isInteger(e) || s < 0 || e <= s || e > a.shape[d]) {
          throw new Error('slice: bad range [' + s + ',' + e + ') for axis ' + d + ' (dim ' + a.shape[d] + ')');
        }
        starts[d] = s; outShape[d] = e - s;
      }
      const inStrides = stridesOf(a.shape);
      const out = new Tensor(outShape);
      const map = buildGatherMap(outShape, (coords) => {
        let lin = 0;
        for (let d = 0; d < a.ndim; d++) lin += (coords[d] + starts[d]) * inStrides[d];
        return lin;
      });
      const src = a.data, dst = out.data;
      for (let i = 0; i < out.size; i++) dst[i] = src[map[i]];
      out._op = 'slice';
      out.requiresGrad = a.requiresGrad;
      out._prev = new Set([a]);
      out._backward = () => {
        if (!a.requiresGrad) return;
        const g = out.grad;
        for (let i = 0; i < out.size; i++) a.grad[map[i]] += g[i];
      };
      return out;
    }

    /* Generic axis permutation: out[k] = in[perm[k]] (NumPy convention).
       Same gather-map forward / scatter-add backward as slice. */
    transpose(perm) {
      const a = this;
      if (!Array.isArray(perm) || perm.length !== a.ndim) {
        throw new Error('transpose: perm must have ' + a.ndim + ' entries');
      }
      const seen = new Array(a.ndim).fill(false);
      for (const p of perm) {
        if (!Number.isInteger(p) || p < 0 || p >= a.ndim || seen[p]) {
          throw new Error('transpose: bad perm [' + perm + ']');
        }
        seen[p] = true;
      }
      const outShape = perm.map(p => a.shape[p]);
      const permStrides = perm.map(p => stridesOf(a.shape)[p]);
      const out = new Tensor(outShape);
      const map = buildGatherMap(outShape, (coords) => {
        let lin = 0;
        for (let k = 0; k < perm.length; k++) lin += coords[k] * permStrides[k];
        return lin;
      });
      const src = a.data, dst = out.data;
      for (let i = 0; i < out.size; i++) dst[i] = src[map[i]];
      out._op = 'transpose';
      out.requiresGrad = a.requiresGrad;
      out._prev = new Set([a]);
      out._backward = () => {
        if (!a.requiresGrad) return;
        const g = out.grad;
        for (let i = 0; i < out.size; i++) a.grad[map[i]] += g[i];
      };
      return out;
    }

    /* Batched 2-D matmul: [N, M, K] @ [N, K, P] -> [N, M, P]. Same fixed
       n,i,k,j forward order and n,i,k,j backward orders as matmul — the
       batched sibling with the same float32 accumulation discipline.
       PERF (Task 6): inner j loops unrolled 4-wide, bit-identical results
       (same per-element accumulation order) — see matmul note. */
    bmm(other) {
      const a = this, b = other;
      if (!(b instanceof Tensor)) throw new Error('bmm: needs a Tensor');
      if (a.ndim !== 3 || b.ndim !== 3) {
        throw new Error('bmm: 3-D only, got [' + a.shape + '] x [' + b.shape + ']');
      }
      const N = a.shape[0], M = a.shape[1], K = a.shape[2];
      if (b.shape[0] !== N || b.shape[1] !== K) {
        throw new Error('bmm: shape mismatch [' + a.shape + '] x [' + b.shape + ']');
      }
      const P = b.shape[2];
      const out = new Tensor([N, M, P]);
      const A = a.data, B = b.data, O = out.data;
      for (let n = 0; n < N; n++) {
        const an = n * M * K, bn = n * K * P, on = n * M * P;
        for (let i = 0; i < M; i++) {
          const ob = on + i * P, ab = an + i * K;
          let j = 0;
          for (; j + 4 <= P; j += 4) {
            O[ob + j] = 0.0; O[ob + j + 1] = 0.0;
            O[ob + j + 2] = 0.0; O[ob + j + 3] = 0.0;
          }
          for (; j < P; j++) O[ob + j] = 0.0;
          for (let k = 0; k < K; k++) {
            const av = A[ab + k], bb = bn + k * P;
            j = 0;
            for (; j + 4 <= P; j += 4) {
              O[ob + j]     += av * B[bb + j];
              O[ob + j + 1] += av * B[bb + j + 1];
              O[ob + j + 2] += av * B[bb + j + 2];
              O[ob + j + 3] += av * B[bb + j + 3];
            }
            for (; j < P; j++) O[ob + j] += av * B[bb + j];
          }
        }
      }
      out._op = 'bmm';
      out.requiresGrad = a.requiresGrad || b.requiresGrad;
      out._prev = new Set([a, b]);
      out._backward = () => {
        const G = out.grad;
        if (a.requiresGrad) {
          // dA[n,i,k] = sum_j G[n,i,j] * B[n,k,j]
          for (let n = 0; n < N; n++) {
            const an = n * M * K, bn = n * K * P, gn = n * M * P;
            for (let i = 0; i < M; i++) {
              const gb = gn + i * P, ab = an + i * K;
              for (let k = 0; k < K; k++) {
                let acc = 0; const kb = bn + k * P;
                for (let j = 0; j < P; j++) acc += G[gb + j] * B[kb + j];
                a.grad[ab + k] += acc;
              }
            }
          }
        }
        if (b.requiresGrad) {
          // dB[n,k,j] = sum_i A[n,i,k] * G[n,i,j]   (4-wide, bit-identical)
          for (let n = 0; n < N; n++) {
            const an = n * M * K, bn = n * K * P, gn = n * M * P;
            for (let i = 0; i < M; i++) {
              const ab = an + i * K, gb = gn + i * P;
              for (let k = 0; k < K; k++) {
                const av = A[ab + k], kb = bn + k * P;
                let j = 0;
                for (; j + 4 <= P; j += 4) {
                  b.grad[kb + j]     += av * G[gb + j];
                  b.grad[kb + j + 1] += av * G[gb + j + 1];
                  b.grad[kb + j + 2] += av * G[gb + j + 2];
                  b.grad[kb + j + 3] += av * G[gb + j + 3];
                }
                for (; j < P; j++) b.grad[kb + j] += av * G[gb + j];
              }
            }
          }
        }
      };
      return out;
    }

    /* x @ w^T for 2-D x [M, K] and w [O, K]: out[i, o] = sum_k x[i,k] * w[o,k].
       This is the WEIGHT-TIED HEAD: logits = lnf_out.matmulT(wte) with wte
       [vocab, d_model] — no separate head parameter exists. Backward:
       dx = G @ w, dw += G^T @ x. Fixed i,k,j orders like matmul. Documented
       decision from the Task 4 contract (cleaner than a transpose op node
       or a non-trainable alias). */
    matmulT(other) {
      const a = this, w = other;
      if (!(w instanceof Tensor)) throw new Error('matmulT: needs a Tensor');
      if (a.ndim !== 2 || w.ndim !== 2) {
        throw new Error('matmulT: 2-D only, got [' + a.shape + '] x [' + w.shape + ']');
      }
      const M = a.shape[0], K = a.shape[1];
      if (w.shape[1] !== K) {
        throw new Error('matmulT: inner dims mismatch [' + a.shape + '] x [' + w.shape + ']');
      }
      const O = w.shape[0];
      const out = new Tensor([M, O]);
      const A = a.data, W = w.data, Om = out.data;
      for (let i = 0; i < M; i++) {
        const ob = i * O, ab = i * K;
        for (let j = 0; j < O; j++) Om[ob + j] = 0.0;
        for (let k = 0; k < K; k++) {
          const av = A[ab + k];
          for (let j = 0; j < O; j++) Om[ob + j] += av * W[j * K + k];
        }
      }
      out._op = 'matmulT';
      out.requiresGrad = a.requiresGrad || w.requiresGrad;
      out._prev = new Set([a, w]);
      out._backward = () => {
        const G = out.grad;
        if (a.requiresGrad) {
          // dA[i,k] = sum_j G[i,j] * W[j,k]
          for (let i = 0; i < M; i++) {
            const gb = i * O, ab = i * K;
            for (let k = 0; k < K; k++) {
              let acc = 0;
              for (let j = 0; j < O; j++) acc += G[gb + j] * W[j * K + k];
              a.grad[ab + k] += acc;
            }
          }
        }
        if (w.requiresGrad) {
          // dW[j,k] = sum_i A[i,k] * G[i,j] — loop nest swapped (i,j,k):
          // for each (j,k) the contributions still arrive in ascending-i
          // order, so results are BIT-IDENTICAL, and w.grad rows are now
          // streamed contiguously (the old [j*K+k] stride was slow).
          for (let i = 0; i < M; i++) {
            const gb = i * O, ab = i * K;
            for (let j = 0; j < O; j++) {
              const av = G[gb + j], wj = j * K;
              let k = 0;
              for (; k + 4 <= K; k += 4) {
                w.grad[wj + k]     += av * A[ab + k];
                w.grad[wj + k + 1] += av * A[ab + k + 1];
                w.grad[wj + k + 2] += av * A[ab + k + 2];
                w.grad[wj + k + 3] += av * A[ab + k + 3];
              }
              for (; k < K; k++) w.grad[wj + k] += av * A[ab + k];
            }
          }
        }
      };
      return out;
    }

    /* Embedding lookup (gather rows of this [vocab, D] matrix):
       out[r] = self[indices[r]] for a flat int array of length N -> out
       shape [N, D]. Backward SCATTER-ADDS into the (possibly repeated)
       source rows — the embedding-table sibling of slice(). This is
       wte[token_id] for the transformer; the Python twin
       (colab/tensor_py.py) implements the identical op. */
    embed(indices) {
      const a = this;
      if (a.ndim !== 2) {
        throw new Error('embed: needs a 2-D [vocab, D] tensor, got [' + a.shape + ']');
      }
      const V = a.shape[0], D = a.shape[1];
      if (!indices || indices.length === 0) throw new Error('embed: empty indices');
      const N = indices.length;
      const idx = new Int32Array(N);
      const flat = new Int32Array(N * D); // source position of every output elem
      for (let r = 0; r < N; r++) {
        const t = indices[r] | 0;
        if (t !== indices[r] || t < 0 || t >= V) {
          throw new Error('embed: index out of range [0,' + V + '): ' + indices[r]);
        }
        idx[r] = t;
        const base = t * D;
        for (let j = 0; j < D; j++) flat[r * D + j] = base + j;
      }
      const out = new Tensor([N, D]);
      const src = a.data, dst = out.data;
      for (let i = 0; i < N * D; i++) dst[i] = src[flat[i]];
      out._op = 'embed';
      out.requiresGrad = a.requiresGrad;
      out._prev = new Set([a]);
      out._backward = () => {
        if (!a.requiresGrad) return;
        const g = out.grad;
        for (let i = 0; i < N * D; i++) a.grad[flat[i]] += g[i];
      };
      return out;
    }

    /* ------------------------------------------------------- backward */

    /* Seeds this tensor's grad to 1 and pulls the whole graph backward in
       reverse topological order. Call ONCE per graph, on a scalar (the
       loss). For training loops: rebuild the forward graph each step,
       zeroGrad() the leaves, backward(), update, repeat. */
    backward() {
      if (this.size !== 1) {
        throw new Error('backward(): seed must be a scalar loss tensor, got shape [' + this.shape + ']');
      }
      // Iterative post-order DFS -> topological order (parents before self).
      const topo = [];
      const visited = new Set();
      const stack = [[this, false]];
      while (stack.length) {
        const frame = stack.pop();
        const t = frame[0];
        if (frame[1]) { topo.push(t); continue; }
        if (visited.has(t)) continue;
        visited.add(t);
        stack.push([t, true]);
        if (t._prev) {
          for (const p of t._prev) if (!visited.has(p)) stack.push([p, false]);
        }
      }
      this.grad[0] += 1;
      for (let i = topo.length - 1; i >= 0; i--) {
        const t = topo[i];
        if (t._backward) t._backward();
      }
      return this;
    }
  }

  function ensureTensor(v) {
    if (v instanceof Tensor) return v;
    if (typeof v === 'number') {
      if (!isFinite(v)) throw new Error('non-finite scalar: ' + v);
      return new Tensor([], Float32Array.of(v));
    }
    throw new TypeError('expected a Tensor or number, got ' + typeof v);
  }

  /* He initialization: N(0, sqrt(2/fanIn)), shape [fanIn, fanOut] so it
     feeds straight into matmul(x, W). */
  function heInit(fanIn, fanOut, seed, opts) {
    return Tensor.randn([fanIn, fanOut], Math.sqrt(2 / fanIn), seed, opts);
  }

  return {
    VERSION,
    Tensor,
    heInit,
    mulberry32,
    makeGaussian,
    broadcastShapes
  };
});
