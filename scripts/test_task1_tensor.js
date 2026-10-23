#!/usr/bin/env node
/* ============================================================================
   p_1 · scripts/test_task1_tensor.js — Task 1 acceptance gate (Node side)
   ----------------------------------------------------------------------------
   Node mirror of colab/test_tensor_py.py (which is itself the Python mirror
   of this test): the float32 engine (docs/tensor.js) backward is compared
   against an INDEPENDENT float64 reference evaluated with central
   differences (h = 1e-5).

       rel = |a - n| / max(|a|, |n|, 1e-3)          PASS: max rel < 1e-3

   The composite exercises ALL 12 Task 1 ops (matmul, add, sub, mul, relu,
   tanh, gelu, softmax, crossEntropy, sum, scale, reshape) AND the Task 4
   ops (layerNorm, matmulT, transpose, slice, bmm, embed) exactly like the
   Python mirror — plus the Task 1 XOR training gate (2-layer MLP to loss
   < 0.01 within 2000 steps), which the browser page docs/tensor_test.html
   also runs.

   Run:  node scripts/test_task1_tensor.js        →  prints ALL TESTS PASSED
   RNG prints (mulberry32 / gaussian / heInit) are cross-compared against
   python3 colab/test_tensor_py.py output — same seeds, same numbers.
   ============================================================================ */
'use strict';

const path = require('path');
const P1 = require(path.join(__dirname, '..', 'docs', 'tensor.js'));

const { Tensor, heInit, mulberry32, makeGaussian, VERSION } = P1;

const FH = 1e-5;          // central-difference step (float64 reference)
const REL_FLOOR = 1e-3;   // denominator floor of the rel metric
const THRESHOLD = 1e-3;   // the spec's bar
let failures = 0;
const ok = (cond, msg) => { console.log((cond ? '  PASS ' : '  FAIL ') + msg); if (!cond) failures++; return cond; };

/* ------------------------------------------------------------------ f64 ref */
// Plain-JS float64 mirror of the loss graph — independent of the engine on
// purpose (same math, same shapes, same constants as the Python mirror).
// Shapes: X[4,8] g[8] b[8] W1[8,6] b2[1,6] WT[5,6] Y[2,3,4] Z[2,3,3]
//         C[2,3,3] V[6,8] W2[5,3]
const GELU_C = 0.7978845608028654;   // sqrt(2/pi) — keep in sync with engine

function shapeOf(a) { const s = []; let x = a; while (Array.isArray(x)) { s.push(x.length); x = x[0]; } return s; }
function zeros(shape) { let a = 1; for (const d of shape) a *= d; return new Array(a).fill(0); }
function at(a, shape, idx) { let o = 0; for (let i = 0; i < shape.length; i++) o = o * shape[i] + idx[i]; return a[o]; }
function setAt(a, shape, idx, v) { let o = 0; for (let i = 0; i < shape.length; i++) o = o * shape[i] + idx[i]; a[o] = v; }

function mirrorLoss(X, g, b, W1, b2, WT, Y, Z, C, V, W2) {
  // t = gelu(layerNorm(X) @ W1 + b2) @ WT^T ; loss per the Python mirror.
  const mu = new Array(4), varr = new Array(4), xhat = [], h = [];
  for (let i = 0; i < 4; i++) {
    let m = 0; for (let j = 0; j < 8; j++) m += X[i][j]; m /= 8; mu[i] = m;
    let v = 0; for (let j = 0; j < 8; j++) v += (X[i][j] - m) * (X[i][j] - m); v /= 8; varr[i] = v;
    xhat.push([]); h.push([]);
    for (let j = 0; j < 8; j++) {
      xhat[i].push((X[i][j] - m) / Math.sqrt(v + 1e-5));
      h[i].push(g[j] * xhat[i][j] + b[j]);
    }
  }
  const q = [], a = [], t = [];
  for (let i = 0; i < 4; i++) {
    q.push([]); a.push([]); t.push([]);
    for (let j = 0; j < 6; j++) {
      let s = b2[0][j]; for (let k = 0; k < 8; k++) s += h[i][k] * W1[k][j];
      q[i].push(s);
      const c = GELU_C * (s + 0.044715 * s * s * s);
      a[i].push(0.5 * s * (1 + Math.tanh(c)));
    }
    for (let j = 0; j < 5; j++) {
      let s = 0; for (let k = 0; k < 6; k++) s += a[i][k] * WT[j][k];   // @ WT^T
      t[i].push(s);
    }
  }
  let e1 = 0; for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) e1 += t[i][j] * t[i][j];
  e1 *= 0.05;
  // Yt = Y.transpose(0,2,1)[:,1:4,:] @ Z  → [2,3,3];  e2 = sum(softmax(0.2*sc)*C)
  let e2 = 0;
  for (let n = 0; n < 2; n++) {
    const sc = [];
    for (let i = 0; i < 3; i++) {
      sc.push([]);
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += Y[n][k][i + 1] * Z[n][k][j];   // Yt[n,i,k]=Y[n,k,i+1]
        sc[i].push(s * 0.2);
      }
    }
    for (let i = 0; i < 3; i++) {
      let mx = -Infinity; for (let j = 0; j < 3; j++) mx = Math.max(mx, sc[i][j]);
      let se = 0; const p = [];
      for (let j = 0; j < 3; j++) { p.push(Math.exp(sc[i][j] - mx)); se += p[j]; }
      for (let j = 0; j < 3; j++) e2 += (p[j] / se) * C[n][i][j];
    }
  }
  let e3 = 0, e4 = 0;
  for (let i = 1; i < 5; i++) for (let j = 2; j < 6; j++) {
    const r = V[i][j] + 0.5;
    e3 += Math.max(r, 0) * Math.max(r, 0);
    e4 += Math.tanh(r);
  }
  e3 *= 0.02; e4 *= 0.01;
  const tg = [0, 2, 1, 2];
  let ce = 0;
  for (let i = 0; i < 4; i++) {
    let mx = -Infinity; for (let j = 0; j < 3; j++) mx = Math.max(mx, t[i] === undefined ? 0 : 0); // noop
    mx = -Infinity; const lg = [];
    for (let j = 0; j < 3; j++) {
      let s = 0; for (let k = 0; k < 5; k++) s += t[i][k] * W2[k][j];
      lg.push(s); mx = Math.max(mx, s);
    }
    let se = 0; for (let j = 0; j < 3; j++) se += Math.exp(lg[j] - mx);
    ce += (mx + Math.log(se)) - lg[tg[i]];
  }
  ce /= 4.0;
  return e1 + e2 + e3 + e4 + ce;
}

function mirrorLossBcast(A, B1, B2, M2, S2, G, Bb) {
  let e1 = 0, e2 = 0, e3 = 0, e5 = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 4; k++) {
    e1 += (A[i][j][k] + B1[k]) * (A[i][j][k] + B1[k]);
    const pr = A[i][j][k] * M2[i][j][0] + B2[i][0][k] + 0.5;
    e2 += pr * pr;
    e3 += Math.tanh(A[i][j][k] - S2[j][k]);
    const ab = A[i][j][k] * B1[k];
    e5 += 0.5 * ab * (1 + Math.tanh(GELU_C * (ab + 0.044715 * ab * ab * ab)));
  }
  e1 *= 0.03; e2 *= 0.04; e3 *= 0.05;
  let e4 = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
    let m = 0; for (let k = 0; k < 4; k++) m += A[i][j][k]; m /= 4;
    let v = 0; for (let k = 0; k < 4; k++) v += (A[i][j][k] - m) * (A[i][j][k] - m); v /= 4;
    for (let k = 0; k < 4; k++) e4 += G[k] * (A[i][j][k] - m) / Math.sqrt(v + 1e-5) + Bb[k];
  }
  e4 *= 0.06; e5 *= 0.01;
  return e1 + e2 + e3 + e4 + e5;
}

const EMBED_IDX = [0, 2, 2, 0, 4, 2];
function mirrorLossEmbed(wte, R) {
  let s = 0;
  for (let i = 0; i < 6; i++) for (let j = 0; j < 3; j++) s += wte[EMBED_IDX[i]][j] * R[i][j];
  return s * 0.1;
}

/* ------------------------------------------------------ engine-side graphs */
function buildLeaves() {
  return {
    X:  Tensor.randn([4, 8], 1.0, 11, { requiresGrad: true, name: 'X' }),
    g:  Tensor.randn([8], 0.5, 12, { requiresGrad: true, name: 'g' }),
    b:  Tensor.randn([8], 0.5, 13, { requiresGrad: true, name: 'b' }),
    W1: heInit(8, 6, 14, { requiresGrad: true, name: 'W1' }),
    b2: Tensor.randn([1, 6], 0.3, 20, { requiresGrad: true, name: 'b2' }),
    WT: Tensor.randn([5, 6], 0.5, 15, { requiresGrad: true, name: 'WT' }),
    Y:  Tensor.randn([2, 3, 4], 0.8, 16, { requiresGrad: true, name: 'Y' }),
    Z:  Tensor.randn([2, 3, 3], 0.8, 17, { requiresGrad: true, name: 'Z' }),
    C:  Tensor.randn([2, 3, 3], 1.0, 18, { requiresGrad: true, name: 'C' }),
    V:  Tensor.randn([6, 8], 0.7, 19, { requiresGrad: true, name: 'V' }),
    W2: heInit(5, 3, 21, { requiresGrad: true, name: 'W2' }),
  };
}

function buildGraph(L) {
  const h = L.X.layerNorm(L.g, L.b);
  const q = h.matmul(L.W1).add(L.b2);
  const a = q.gelu();
  const t = a.matmulT(L.WT);
  const e1 = t.mul(t).sum().scale(0.05);
  const Yt = L.Y.transpose([0, 2, 1]);
  const Ys = Yt.slice([null, [1, 4], null]);
  const sc = Ys.bmm(L.Z);
  const sm = sc.scale(0.2).reshape([6, 3]).softmax().reshape([2, 3, 3]);
  const e2 = sm.reshape([18]).mul(L.C.reshape([18])).sum();
  const Vs = L.V.slice([[1, 5], [2, 6]]);
  const r = Vs.add(0.5);
  const rl = r.relu();
  const th = r.tanh();
  const e3 = rl.mul(rl).sum().scale(0.02);
  const e4 = th.sum().scale(0.01);
  const lg = t.matmul(L.W2);
  const ce = lg.crossEntropy([0, 2, 1, 2]);
  return e1.add(e2).add(e3).add(e4).add(ce);
}

function buildLeavesBcast() {
  return {
    A:  Tensor.randn([2, 3, 4], 1.0, 31, { requiresGrad: true, name: 'A' }),
    B1: Tensor.randn([4], 1.0, 32, { requiresGrad: true, name: 'B1' }),
    B2: Tensor.randn([2, 1, 4], 0.7, 33, { requiresGrad: true, name: 'B2' }),
    M2: Tensor.randn([2, 3, 1], 0.8, 34, { requiresGrad: true, name: 'M2' }),
    S2: Tensor.randn([3, 4], 0.9, 35, { requiresGrad: true, name: 'S2' }),
    G:  Tensor.randn([4], 0.5, 36, { requiresGrad: true, name: 'G' }),
    Bb: Tensor.randn([4], 0.5, 37, { requiresGrad: true, name: 'Bb' }),
  };
}

function buildGraphBcast(L) {
  const e1 = L.A.add(L.B1).mul(L.A.add(L.B1)).sum().scale(0.03);
  const prod = L.A.mul(L.M2).add(L.B2).add(0.5);
  const e2 = prod.mul(prod).sum().scale(0.04);
  const e3 = L.A.sub(L.S2).tanh().sum().scale(0.05);
  const e4 = L.A.layerNorm(L.G, L.Bb).sum().scale(0.06);
  const e5 = L.A.mul(L.B1).gelu().sum().scale(0.01);
  return e1.add(e2).add(e3).add(e4).add(e5);
}

function buildLeavesEmbed() {
  return {
    wte: Tensor.randn([5, 3], 1.0, 41, { requiresGrad: true, name: 'wte' }),
    R:   Tensor.randn([6, 3], 1.0, 42, { requiresGrad: true, name: 'R' }),
  };
}

function buildGraphEmbed(L) {
  return L.wte.embed(EMBED_IDX).mul(L.R).sum().scale(0.1);
}

/* ------------------------------------------------------------- grad driver */
// nested-array view over a Tensor's flat data (float64 reference domain)
function toNested(data, shape) {
  function build(off, dims) {
    if (dims.length === 0) return data[off];
    const out = [];
    const stride = dims.slice(1).reduce((a, b) => a * b, 1);
    for (let i = 0; i < dims[0]; i++) out.push(build(off + i * stride, dims.slice(1)));
    return out;
  }
  return build(0, shape);
}

function gradCheck(L, mirrorFn, label, mirrorArgsFn) {
  const loss = mirrorArgsFn.graph(L);
  loss.backward();
  const engineLoss = loss.item();
  const flat = {}, shapes = {}, grads = {};
  for (const k of Object.keys(L)) {
    flat[k] = Array.from(L[k].data);            // f64 copy, mutable
    shapes[k] = L[k].shape.slice();
    grads[k] = Array.from(L[k].grad);
  }
  const call = (f) => mirrorFn(...mirrorArgsFn.args(f));
  const mirror = call(flat);
  console.log('  engine loss (f32)     : ' + engineLoss.toFixed(7));
  console.log('  mirror loss (f64 ref) : ' + mirror.toFixed(7) +
              '   |diff| ' + Math.abs(engineLoss - mirror).toExponential(1));
  let maxRel = 0, maxAt = '';
  const perLeaf = {};
  for (const k of Object.keys(flat)) {
    const size = flat[k].length;
    let leafMax = 0;
    for (let i = 0; i < size; i++) {
      const orig = flat[k][i];
      flat[k][i] = orig + FH;
      const fp = call(flat);
      flat[k][i] = orig - FH;
      const fm = call(flat);
      flat[k][i] = orig;
      const num = (fp - fm) / (2 * FH);
      const a_ = grads[k][i];
      const rel = Math.abs(a_ - num) / Math.max(Math.abs(a_), Math.abs(num), REL_FLOOR);
      if (rel > leafMax) leafMax = rel;
      if (rel > maxRel) { maxRel = rel; maxAt = k + '[' + i + ']'; }
    }
    perLeaf[k] = leafMax;
  }
  for (const k of Object.keys(perLeaf)) {
    console.log('  ' + k.padEnd(3) + ' ' + shapes[k].join('x').padEnd(9) +
                ' max rel err ' + perLeaf[k].toExponential(2));
  }
  console.log('  max over all elements : ' + maxRel.toExponential(3) +
              '  (at ' + maxAt + ', threshold ' + THRESHOLD.toExponential(1) + ')');
  return maxRel < THRESHOLD;
}

const argsA = (f) => {
  const g = (k, sh) => toNested(f[k], sh);
  return [g('X', [4, 8]), g('g', [8]), g('b', [8]), g('W1', [8, 6]), g('b2', [1, 6]),
          g('WT', [5, 6]), g('Y', [2, 3, 4]), g('Z', [2, 3, 3]), g('C', [2, 3, 3]),
          g('V', [6, 8]), g('W2', [5, 3])];
};

/* ------------------------------------------------------------------- XOR */
function xorTest() {
  // 2-layer MLP on the engine, trained with plain SGD — Task 1 gate (b):
  // loss < 0.01 within 2000 steps. 4 samples, 2 classes.
  const x = new Tensor([4, 2], Float32Array.from([0, 0, 0, 1, 1, 0, 1, 1]));
  const W1 = heInit(2, 8, 101, { requiresGrad: true });
  const b1 = new Tensor([8], null, { requiresGrad: true });
  b1.data.fill(0.1);
  const W2 = heInit(8, 2, 102, { requiresGrad: true });
  const b2 = new Tensor([2], null, { requiresGrad: true });
  b2.data.fill(0.1);
  const params = [W1, b1, W2, b2];
  const targets = [0, 1, 1, 0];
  const lr = 0.5;
  let loss = NaN;
  for (let step = 0; step < 2000; step++) {
    for (const p of params) p.zeroGrad();
    const h = x.matmul(W1).add(b1).relu();
    const logits = h.matmul(W2).add(b2);
    const ce = logits.crossEntropy(targets);   // scalar loss tensor
    loss = ce.item();
    ce.backward();
    for (const p of params) for (let i = 0; i < p.data.length; i++) p.data[i] -= lr * p.grad[i];
    if (loss < 0.01) break;
  }
  console.log('\n  XOR — final loss ' + loss.toFixed(6) + ' (bar 0.01, 2000 steps max)');
  return loss < 0.01;
}

/* -------------------------------------------------------------------- main */
function main() {
  console.log('p_1 · test_task1_tensor.js — engine gradient check (engine v' + VERSION + ')');

  // RNG parity proof (cross-compare with python3 colab/test_tensor_py.py)
  const r = mulberry32(42);
  const u10 = []; for (let i = 0; i < 10; i++) u10.push(r());
  console.log('  mulberry32(42) first 10 uniforms:');
  console.log('   ', u10.map((v) => v.toFixed(9)).join(' '));
  const ga = makeGaussian(mulberry32(7));
  const g5 = []; for (let i = 0; i < 5; i++) g5.push(ga());
  console.log('  make_gaussian(mulberry32(7)) first 5 gaussians:');
  console.log('   ', g5.map((v) => v.toFixed(9)).join(' '));
  const h = heInit(4, 3, 123);
  console.log('  heInit(4,3,123) flat first 6 (std=sqrt(2/4)):');
  console.log('   ', Array.from(h.data.slice(0, 6)).map((v) => v.toFixed(9)).join(' '));

  console.log('\n  TEST A — all-ops gradient check (engine vs f64 central diff)');
  const okA = gradCheck(buildLeaves(), mirrorLoss, 'A', { graph: buildGraph, args: argsA });

  console.log('\n  TEST A2 — broadcast torture (3-D add/sub/mul, scalar, 3-D layerNorm)');
  const okB = gradCheck(buildLeavesBcast(), mirrorLossBcast, 'A2', {
    graph: buildGraphBcast,
    args: (f) => [toNested(f.A, [2, 3, 4]), toNested(f.B1, [4]), toNested(f.B2, [2, 1, 4]),
                  toNested(f.M2, [2, 3, 1]), toNested(f.S2, [3, 4]), toNested(f.G, [4]),
                  toNested(f.Bb, [4])],
  });

  console.log('\n  TEST A3 — embed lookup, repeated indices (grad accumulation)');
  const okC = gradCheck(buildLeavesEmbed(), mirrorLossEmbed, 'A3', {
    graph: buildGraphEmbed,
    args: (f) => [toNested(f.wte, [5, 3]), toNested(f.R, [6, 3])],
  });

  const okX = xorTest();

  const all = okA && okB && okC && okX;
  console.log('\n  ' + (all ? 'PASS' : 'FAIL'));
  if (all) console.log('\nALL TESTS PASSED — float32 engine backward matches the f64 mirror; XOR learned.');
  return all ? 0 : 1;
}

process.exit(main());
