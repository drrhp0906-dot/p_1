#!/usr/bin/env python3
"""
p_1 · colab/test_tensor_py.py — gradient check for the NumPy engine
--------------------------------------------------------------------------------
Python mirror of the Task 1 test (scripts/test_task1_tensor.js): the float32
engine backward is compared against an INDEPENDENT float64 reference evaluated
with central differences (h = 1e-5).

    rel = |a - n| / max(|a|, |n|, 1e-3)          PASS: max rel < 1e-3

The composite exercises ALL ops of colab/tensor_py.py — the 12 Task 1 ops
(matmul, add, sub, mul, relu, tanh, gelu, softmax, crossEntropy, sum, scale,
reshape) AND the 5 Task 4 ops (layerNorm, matmulT, transpose, slice, bmm) —
so the new ops are gradient-checked on the Python side exactly as required.

TEST A2 is a broadcast torture composite: every binary op (add/sub/mul) is
gradient-checked in the shapes that actually occur inside the transformer
([2,3,4] op [4], op [2,1,4], op [3,4], op [2,3,1], op scalar) plus 3-D
layerNorm — i.e. the _unbroadcast paths the main composite does not hit.

Also prints the RNG parity proof (mulberry32 uniforms + Marsaglia-polar
gaussians for fixed seeds) to compare against the Node side.

Run:  python3 colab/test_tensor_py.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor_py import (Tensor, heInit, mulberry32, make_gaussian,
                       GELU_C, VERSION)

FH = 1e-5          # central-difference step (float64 reference)
REL_FLOOR = 1e-3   # denominator floor of the rel metric
THRESHOLD = 1e-3   # the spec's bar


# --------------------------------------------------------------------------
# float64 reference mirror — independent of tensor_py on purpose.
# Same math, same shapes, same constants as the engine graph below.
# Shapes: X[4,8] g[8] b[8] W1[8,6] b2[1,6] WT[5,6] Y[2,3,4] Z[2,3,3]
#         C[2,3,3] V[6,8] W2[5,3]
# loss = 0.05*sum(t^2) + sum(softmax(0.2*(Yt[:,1:4,:] @ Z)) * C)
#        + 0.02*sum(relu(V[1:5,2:6]+0.5)^2) + 0.01*sum(tanh(V[1:5,2:6]+0.5))
#        + CE(t @ W2, [0,2,1,2])
# where t = gelu(layerNorm(X) @ W1 + b2) @ WT^T
# --------------------------------------------------------------------------
def mirror_loss(X, g, b, W1, b2, WT, Y, Z, C, V, W2):
    mu = X.mean(axis=-1, keepdims=True)
    var = ((X - mu) ** 2).mean(axis=-1, keepdims=True)
    xhat = (X - mu) / np.sqrt(var + 1e-5)
    h = g * xhat + b                                   # [4,8]
    q = h @ W1 + b2                                    # [4,6] (b2 broadcasts)
    a = 0.5 * q * (1.0 + np.tanh(GELU_C * (q + 0.044715 * q * q * q)))
    t = a @ WT.T                                       # [4,5]  (matmulT)
    e1 = (t * t).sum() * 0.05
    Yt = Y.transpose(0, 2, 1)[:, 1:4, :]               # transpose + slice
    sc = Yt @ Z                                        # bmm [2,3,3]
    z2 = sc * 0.2
    m = z2.max(axis=-1, keepdims=True)
    e_ = np.exp(z2 - m)
    p_ = e_ / e_.sum(axis=-1, keepdims=True)
    e2 = (p_.reshape(18) * C.reshape(18)).sum()        # reshape + mul + sum
    Vs = V[1:5, 2:6]                                   # 2-D slice
    r = Vs + 0.5
    rl = np.maximum(r, 0.0)
    th = np.tanh(r)
    e3 = (rl * rl).sum() * 0.02
    e4 = th.sum() * 0.01
    lg = t @ W2                                        # [4,3]
    tg = [0, 2, 1, 2]
    ce = 0.0
    for i in range(4):
        mm = lg[i].max()
        se = np.exp(lg[i] - mm).sum()
        ce += (mm + math.log(se)) - lg[i, tg[i]]
    ce /= 4.0
    return e1 + e2 + e3 + e4 + ce


# engine-side graph, all 17 ops represented
def build_leaves():
    return {
        'X':  Tensor.randn([4, 8], 1.0, 11, requires_grad=True, name='X'),
        'g':  Tensor.randn([8], 0.5, 12, requires_grad=True, name='g'),
        'b':  Tensor.randn([8], 0.5, 13, requires_grad=True, name='b'),
        'W1': heInit(8, 6, 14, requires_grad=True, name='W1'),
        'b2': Tensor.randn([1, 6], 0.3, 20, requires_grad=True, name='b2'),
        'WT': Tensor.randn([5, 6], 0.5, 15, requires_grad=True, name='WT'),
        'Y':  Tensor.randn([2, 3, 4], 0.8, 16, requires_grad=True, name='Y'),
        'Z':  Tensor.randn([2, 3, 3], 0.8, 17, requires_grad=True, name='Z'),
        'C':  Tensor.randn([2, 3, 3], 1.0, 18, requires_grad=True, name='C'),
        'V':  Tensor.randn([6, 8], 0.7, 19, requires_grad=True, name='V'),
        'W2': heInit(5, 3, 21, requires_grad=True, name='W2'),
    }


def build_graph(L):
    h = L['X'].layerNorm(L['g'], L['b'])                    # layerNorm
    q = h.matmul(L['W1']).add(L['b2'])                      # matmul + bcast add
    a = q.gelu()                                            # gelu
    t = a.matmulT(L['WT'])                                  # matmulT
    e1 = t.mul(t).sum().scale(0.05)                         # mul + sum + scale
    Yt = L['Y'].transpose([0, 2, 1])                        # transpose
    Ys = Yt.slice([None, [1, 4], None])                     # slice (mixed)
    sc = Ys.bmm(L['Z'])                                     # bmm
    sm = sc.scale(0.2).reshape([6, 3]).softmax().reshape([2, 3, 3])  # scale + softmax + reshape
    e2 = sm.reshape([18]).mul(L['C'].reshape([18])).sum()   # reshape + mul + sum
    Vs = L['V'].slice([[1, 5], [2, 6]])                     # 2-D slice
    r = Vs.add(0.5)                                         # scalar add
    rl = r.relu()
    th = r.tanh()
    e3 = rl.mul(rl).sum().scale(0.02)
    e4 = th.sum().scale(0.01)
    lg = t.matmul(L['W2'])                                  # second matmul
    ce = lg.crossEntropy([0, 2, 1, 2])                      # fused CE
    return e1.add(e2).add(e3).add(e4).add(ce)


# --------------------------------------------------------------------------
# TEST A2 — broadcast torture: gradient-check the _unbroadcast paths
# (3-D broadcasts on every axis + 1-D + scalar sides) that the main
# composite does not exercise. All smooth (tanh/gelu/squares — no relu kinks
# near 0) so central differences stay clean.
#
# Shapes: A[2,3,4] B1[4] B2[2,1,4] M2[2,3,1] S2[3,4] G[4] Bb[4]
# loss = 0.03*sum((A+B1)^2) + 0.04*sum((A*M2+B2+0.5)^2)
#        + 0.05*sum(tanh(A-S2)) + 0.06*sum(layerNorm(A,G,Bb))
#        + 0.01*sum(gelu(A*B1))
# --------------------------------------------------------------------------
def mirror_loss_bcast(A, B1, B2, M2, S2, G, Bb):
    e1 = ((A + B1) ** 2).sum() * 0.03
    e2 = ((A * M2 + B2 + 0.5) ** 2).sum() * 0.04
    e3 = np.tanh(A - S2).sum() * 0.05
    mu = A.mean(axis=-1, keepdims=True)
    var = ((A - mu) ** 2).mean(axis=-1, keepdims=True)
    xhat = (A - mu) / np.sqrt(var + 1e-5)
    e4 = (G * xhat + Bb).sum() * 0.06
    e5 = (0.5 * (A * B1) * (1.0 + np.tanh(
        GELU_C * ((A * B1) + 0.044715 * (A * B1) ** 3)))).sum() * 0.01
    return e1 + e2 + e3 + e4 + e5


def build_leaves_bcast():
    return {
        'A':  Tensor.randn([2, 3, 4], 1.0, 31, requires_grad=True, name='A'),
        'B1': Tensor.randn([4], 1.0, 32, requires_grad=True, name='B1'),
        'B2': Tensor.randn([2, 1, 4], 0.7, 33, requires_grad=True, name='B2'),
        'M2': Tensor.randn([2, 3, 1], 0.8, 34, requires_grad=True, name='M2'),
        'S2': Tensor.randn([3, 4], 0.9, 35, requires_grad=True, name='S2'),
        'G':  Tensor.randn([4], 0.5, 36, requires_grad=True, name='G'),
        'Bb': Tensor.randn([4], 0.5, 37, requires_grad=True, name='Bb'),
    }


def build_graph_bcast(L):
    e1 = L['A'].add(L['B1']).mul(L['A'].add(L['B1'])).sum().scale(0.03)
    prod = L['A'].mul(L['M2']).add(L['B2']).add(0.5)          # 3-D bcast mul/add + scalar
    e2 = prod.mul(prod).sum().scale(0.04)
    e3 = L['A'].sub(L['S2']).tanh().sum().scale(0.05)         # 3-D bcast sub
    e4 = L['A'].layerNorm(L['G'], L['Bb']).sum().scale(0.06)  # 3-D layerNorm
    e5 = L['A'].mul(L['B1']).gelu().sum().scale(0.01)         # 1-D bcast mul
    return e1.add(e2).add(e3).add(e4).add(e5)


# --------------------------------------------------------------------------
# TEST A3 — embed (wte lookup) with REPEATED indices: grads must accumulate
# across occurrences (the weight-tied embedding gets contributions from both
# the lookup and the head inside the transformer).
# Shapes: wte[5,3] R[6,3], idx=[0,2,2,0,4,2]
# loss = 0.1 * sum(wte[idx] * R)
# --------------------------------------------------------------------------
EMBED_IDX = [0, 2, 2, 0, 4, 2]


def mirror_loss_embed(wte, R):
    gathered = wte[EMBED_IDX]                        # f64 fancy-index gather
    return (gathered * R).sum() * 0.1


def build_leaves_embed():
    return {
        'wte': Tensor.randn([5, 3], 1.0, 41, requires_grad=True, name='wte'),
        'R':   Tensor.randn([6, 3], 1.0, 42, requires_grad=True, name='R'),
    }


def build_graph_embed(L):
    return (L['wte'].embed(EMBED_IDX).mul(L['R']).sum().scale(0.1))


def grad_check(L, mirror_fn, label):
    """Shared driver: engine backward vs central differences of mirror_fn."""
    builders = {'A': build_graph, 'A2': build_graph_bcast, 'A3': build_graph_embed}
    loss = builders[label](L)
    loss.backward()
    engine_loss = loss.item()
    ref = {k: L[k].data.reshape(L[k].shape).astype(np.float64).copy() for k in L}
    mirror = mirror_fn(*[ref[k] for k in ref])
    print('  engine loss (f32)     : %.7f' % engine_loss)
    print('  mirror loss (f64 ref) : %.7f   |diff| %.1e'
          % (mirror, abs(engine_loss - mirror)))
    ana = {k: L[k].grad.astype(np.float64).copy() for k in L}
    max_rel, max_at = 0.0, ''
    per_leaf = {}
    for k in ref:
        arr = ref[k]
        leaf_max = 0.0
        for i in range(arr.size):
            orig = arr.flat[i]
            arr.flat[i] = orig + FH
            fp = mirror_fn(*[ref[j] for j in ref])
            arr.flat[i] = orig - FH
            fm = mirror_fn(*[ref[j] for j in ref])
            arr.flat[i] = orig
            num = (fp - fm) / (2 * FH)
            a_ = ana[k].flat[i]
            rel = abs(a_ - num) / max(abs(a_), abs(num), REL_FLOOR)
            leaf_max = max(leaf_max, rel)
            if rel > max_rel:
                max_rel, max_at = rel, '%s[%d]' % (k, i)
        per_leaf[k] = leaf_max
    for k in per_leaf:
        print('  %-3s %-9s max rel err %.2e' % (k, 'x'.join(map(str, L[k].shape)), per_leaf[k]))
    print('  max over all elements : %.3e  (at %s, threshold %.1e)'
          % (max_rel, max_at, THRESHOLD))
    return max_rel < THRESHOLD


def main():
    print('p_1 · test_tensor_py.py — NumPy engine gradient check (engine v%s)' % VERSION)

    # ---- RNG parity proof (compare with scripts/test_task4_model.js output)
    r = mulberry32(42)
    u10 = [r() for _ in range(10)]
    print('  mulberry32(42) first 10 uniforms:')
    print('   ', ' '.join('%.9f' % v for v in u10))
    ga = make_gaussian(mulberry32(7))
    g5 = [ga() for _ in range(5)]
    print('  make_gaussian(mulberry32(7)) first 5 gaussians:')
    print('   ', ' '.join('%.9f' % v for v in g5))
    h = heInit(4, 3, 123)
    print('  heInit(4,3,123) flat first 6 (std=sqrt(2/4)):')
    print('   ', ' '.join('%.9f' % v for v in h.data[:6]))

    # ---- TEST A: all-ops composite
    print('\n  TEST A — all-ops gradient check (engine vs f64 central diff)')
    L = build_leaves()
    ok_a = grad_check(L, mirror_loss, 'A')

    # ---- TEST A2: broadcast torture
    print('\n  TEST A2 — broadcast torture (3-D add/sub/mul, scalar, 3-D layerNorm)')
    L2 = build_leaves_bcast()
    ok_b = grad_check(L2, mirror_loss_bcast, 'A2')

    # ---- TEST A3: embed with repeated indices (scatter-add accumulation)
    print('\n  TEST A3 — embed lookup, repeated indices (grad accumulation)')
    L3 = build_leaves_embed()
    ok_c = grad_check(L3, mirror_loss_embed, 'A3')

    print('\n  ' + ('PASS' if (ok_a and ok_b and ok_c) else 'FAIL'))
    return 0 if (ok_a and ok_b and ok_c) else 1


if __name__ == '__main__':
    sys.exit(main())
