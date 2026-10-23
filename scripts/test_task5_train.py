#!/usr/bin/env python3
"""
p_1 · scripts/test_task5_train.py — Task 5 LOCAL VALIDATION
--------------------------------------------------------------------------------
Proves the overnight-training machinery on REAL corpus data through the REAL
BPE, at tiny scale (n_layer 2, d_model 64, ctx 128, B 8, 150 steps):

  1. RUN A (one process)  — 150 uninterrupted steps: loss printed every 25,
     all finite, last-25 avg < first-25 avg, checkpoint at step 75 (forced
     via cfg['ckpt_at_steps'] which calls save_now()).
  2. KILL / RESUME TEST   — RUN B1: new process trains to step 75, saves,
     exits (the "crash"). RUN B2: ANOTHER new process auto-resumes from the
     checkpoint and continues to 150. Acceptance: resume starts at 75,
     config/tokenizer_version validated, and
       |avg loss 76-150 (resumed) - avg loss 76-150 (uninterrupted)| < 0.15
       no single resumed step > uninterrupted running-max at that point + 0.5
     Side-by-side trajectories printed + verdict "RESUME CLEAN".
     (Steps 1-75 must additionally be IDENTICAL across A and B — seeded
     determinism: per-step batches = default_rng([seed, step]).)
  3. CHECKPOINT STRUCTURE — contract keys (step int64 / config JSON /
     tokenizer_version / param: m: v: float32 in canonical shapes),
     latest.txt content, no .tmp leftovers (atomic replace), junk .tmp
     ignored, stale latest.txt falls back to glob-max.
  4. CONFIG B SPEED       — 5 real steps at the largest batch this box's RAM
     allows (B=16 ctx=512 on >=8 GB; B=4/B=2 here), steps/min + tokens/min +
     projected wall-clock for the 20,000-step schedule -> honest expectations
     for the notebook's md.
  5. CSV PLOT CHECK       — plot_history from run A's CSV -> scripts/
     task5_loss_smoke.png.

Run:  python3 scripts/test_task5_train.py               # the whole suite
      python3 scripts/test_task5_train.py --mode run-a  --ckpt-dir DIR   (child)
      python3 scripts/test_task5_train.py --mode run-b1 --ckpt-dir DIR   (child)
      python3 scripts/test_task5_train.py --mode run-b2 --ckpt-dir DIR   (child)
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'colab'))     # tensor_py / 03_model / tokenizer_py / train_lib

import importlib
import train_lib as tl

M03 = importlib.import_module('03_model')
from tokenizer_py import P1BPE

TINY = {'n_layer': 2, 'n_head': 2, 'd_model': 64, 'd_ff': 128,
        'ctx': 128, 'vocab_size': 4096}
STEPS = 150
WARMUP = 20
LR = 1e-3
BATCH = 8
LOG_EVERY = 5
EVAL_EVERY = 50
CKPT_AT = [75]
SEED = 1234
MERGES = REPO / 'corpus' / 'merges.json'
HEALTH_QA = REPO / 'corpus' / 'health_qa.jsonl'
CACHE = Path('/tmp/p1_task5_streams.npz')


def base_cfg(total_steps):
    return {'lr': LR, 'warmup': WARMUP, 'total_steps': total_steps,
            'batch_size': BATCH, 'clip': 1.0, 'eval_every': EVAL_EVERY,
            'ckpt_every_minutes': 1e9,      # no wall-clock saves in the test
            'log_every': LOG_EVERY, 'seed': SEED, 'keep_last': 0,
            'val_batch_windows': 8, 'final_save': True}


def make_data(ctx, max_val_windows=16):
    """Real corpus -> real BPE -> packed streams (cached to /tmp so the
    NEW-process children of the kill test don't each re-encode 2 MB)."""
    tokver = tl.tokenizer_version_from_file(MERGES)
    if CACHE.exists():
        z = np.load(str(CACHE), allow_pickle=False)
        if str(z['tokver']) == tokver:
            tr, va = z['train_stream'], z['val_stream']
            z.close()
            return tl.PackedData(tr, va, ctx, seed=SEED,
                                 max_val_windows=max_val_windows)
    tok = P1BPE(str(MERGES))
    data = tl.build_data(HEALTH_QA, tok, ctx, seed=SEED,
                         max_val_windows=max_val_windows)
    np.savez(str(CACHE), train_stream=data.train_stream,
             val_stream=data.val_stream, tokver=tokver,
             n_train=data.n_train_entries, n_val=data.n_val_entries)
    return data


def build_trainer(ckpt_dir, total_steps, ckpt_at_steps=None):
    tokver = tl.tokenizer_version_from_file(MERGES)
    data = make_data(TINY['ctx'])
    model = M03.Transformer(dict(TINY), seed=42)
    cfg = base_cfg(total_steps)
    if ckpt_at_steps:
        cfg['ckpt_at_steps'] = ckpt_at_steps
    return tl.Trainer(model, tokver, ckpt_dir, data, cfg=cfg)


# ------------------------------------------------------------- child modes

def mode_run(mode, ckpt_dir):
    max_steps = {'run-a': STEPS, 'run-b1': 75, 'run-b2': STEPS}[mode]
    trainer = build_trainer(ckpt_dir, STEPS,
                            ckpt_at_steps=CKPT_AT if mode == 'run-a' else None)
    trainer.train(max_steps=max_steps)

def run_child(mode, ckpt_dir):
    cmd = [sys.executable, str(HERE / 'test_task5_train.py'),
           '--mode', mode, '--ckpt-dir', str(ckpt_dir)]
    print('\n===== child: %s (fresh process: %s) =====' % (mode, ckpt_dir))
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    print('===== child %s finished in %.1fs (exit %d) ====='
          % (mode, dt, r.returncode))
    assert r.returncode == 0, 'child %s FAILED' % mode


def read_csv(ckpt_dir):
    """[(step, kind, loss)] + raw line count; header must appear exactly once."""
    path = Path(ckpt_dir) / 'train_log.csv'
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines[0] == tl.CSV_HEADER, 'CSV header mismatch: %r' % lines[0]
    assert sum(1 for ln in lines if ln.startswith('step,kind')) == 1, \
        'header written more than once (append corruption?)'
    rows = []
    for ln in lines[1:]:
        step, kind, loss = ln.split(',')[:3]
        rows.append((int(step), kind, float(loss)))
    steps = [s for s, _, _ in rows]
    assert steps == sorted(steps), 'CSV steps not non-decreasing (corruption?)'
    for kind in ('train', 'val'):          # unique within each kind (a step
        ks = [s for s, k, _ in rows        # may legitimately have BOTH a
              if k == kind]                # train row and a val row)
        assert ks == sorted(ks) and len(set(ks)) == len(ks), \
            'CSV %s steps not strictly increasing (append corruption?)' % kind
    return rows


def check_structure(ckpt_path, expect_step, expect_cfg, expect_tokver,
                    expect_latest):
    print('\n--- checkpoint structure: %s' % ckpt_path)
    z = np.load(str(ckpt_path), allow_pickle=False)
    files = z.files
    assert 'step' in files and 'config' in files and 'tokenizer_version' in files
    step_arr = z['step']
    assert step_arr.dtype == np.int64 and step_arr.shape == (), \
        'step must be a scalar int64, got %s %s' % (step_arr.dtype, step_arr.shape)
    assert int(step_arr) == expect_step
    cfg = json.loads(str(z['config']))
    assert cfg == expect_cfg, 'config mismatch: %r' % cfg
    assert str(z['tokenizer_version']) == expect_tokver
    names = M03.param_names(expect_cfg)
    for prefix in ('param:', 'm:', 'v:'):
        got = sorted(k[len(prefix):] for k in files if k.startswith(prefix))
        assert got == sorted(names), '%s* keys != canonical names' % prefix
    # spot-check a couple of names + every dtype/shape class
    shape_of = {'wte': (expect_cfg['vocab_size'], expect_cfg['d_model']),
                'wpe': (expect_cfg['ctx'], expect_cfg['d_model']),
                'lnf.g': (expect_cfg['d_model'],),
                'blk1.mlp.fc.w': (expect_cfg['d_model'], expect_cfg['d_ff'])}
    for nm, shp in shape_of.items():
        for prefix in ('param:', 'm:', 'v:'):
            a = z[prefix + nm]
            assert a.dtype == np.float32, (prefix + nm, a.dtype)
            assert a.shape == shp, (prefix + nm, a.shape, shp)
    n_param_keys = sum(1 for k in files if k.startswith('param:'))
    print('    keys OK: step(int64 scalar)=%d · config JSON == model config · '
          'tokenizer_version=%r' % (int(step_arr), str(z['tokenizer_version'])))
    print('    %d param:* + %d m:* + %d v:* float32 keys (canonical order, '
          'logical shapes; spot: param:wte%s m:wte%s v:lnf.g%s)'
          % (n_param_keys, n_param_keys, n_param_keys,
             z['param:wte'].shape, z['m:wte'].shape, z['v:lnf.g'].shape))
    z.close()
    d = Path(ckpt_path).parent
    latest = (d / 'latest.txt').read_text(encoding='utf-8')
    assert latest == expect_latest, 'latest.txt %r != %r' % (latest, expect_latest)
    print('    latest.txt = %r  ✓ (newest checkpoint)' % latest)
    tmps = sorted(d.glob('*.tmp'))
    assert not tmps, 'leftover .tmp files: %r' % tmps
    print('    no .tmp leftovers (atomic os.replace worked)  ✓')


def robustness_probes(ckpt_dir, real_name):
    """(i) junk .tmp file ignored by find_resume; (ii) stale latest.txt ->
    glob-max fallback."""
    d = Path(ckpt_dir)
    (d / 'ckpt_step9999.npz.tmp').write_bytes(b'junk from a crashed write')
    p = tl.find_resume(d)
    assert Path(p).name == real_name, 'junk .tmp changed resume pick: %r' % p
    print('    probe (i): ckpt_step9999.npz.tmp present -> find_resume still '
          'picks %s  ✓' % Path(p).name)
    (d / 'ckpt_step9999.npz.tmp').unlink()
    lt = d / 'latest.txt'
    keep = lt.read_text(encoding='utf-8')
    lt.write_text('ckpt_step424242.npz\n', encoding='utf-8')
    p = tl.find_resume(d)
    assert Path(p).name == real_name, 'stale latest.txt broke fallback: %r' % p
    print('    probe (ii): latest.txt -> missing file -> glob fallback picks '
          '%s  ✓' % Path(p).name)
    lt.write_text(keep, encoding='utf-8')


def side_by_side(rows_a, rows_b):
    print('\n    step |   uninterrupted |      resumed |  |diff|')
    print('    -----+-----------------+--------------+---------')
    worst = 0.0
    for (sa, ka, la), (sb, kb, lb) in zip(rows_a, rows_b):
        assert sa == sb and ka == kb
        d = abs(la - lb)
        worst = max(worst, d)
        print('    %4d | %14.4f  | %12.4f | %+.4f %s'
              % (sa, la, lb, lb - la, '(train)' if ka == 'train' else '(val)  '))
    return worst


def kill_resume_test(dir_a, dir_b):
    rows_a = read_csv(dir_a)
    rows_b = read_csv(dir_b)
    ra = [r for r in rows_a if r[1] == 'train']
    rb = [r for r in rows_b if r[1] == 'train']
    va = [r for r in rows_a if r[1] == 'val']
    vb = [r for r in rows_b if r[1] == 'val']

    print('\n================ KILL / RESUME TEST ================')
    print('run A: %d train rows (steps %d..%d), %d val rows'
          % (len(ra), ra[0][0], ra[-1][0], len(va)))
    print('run B: %d train rows (steps %d..%d), %d val rows '
          '(1-75 from the pre-kill process, 76-150 resumed in a NEW process)'
          % (len(rb), rb[0][0], rb[-1][0], len(vb)))

    first = [r for r in rb if r[0] <= 75]
    resumed = [r for r in rb if r[0] > 75]
    assert resumed and resumed[0][0] == 80, \
        'resumed run must start at step 80, got %r' % (resumed[:1],)

    # -- determinism: pre-kill phase identical in both runs
    la = {s: l for s, _, l in ra}
    lb = {s: l for s, _, l in rb}
    pre = sorted(s for s, _, _ in first)
    pre_diff = max(abs(la[s] - lb[s]) for s in pre)
    print('\n    determinism steps 5..75: max |A-B| = %.6g (expect 0.0 — '
          'same seeds, per-step rng)' % pre_diff)
    assert pre_diff < 1e-6, 'pre-kill trajectories diverge — determinism broken'

    # -- acceptance metric 1: avg steps 76-150
    rng = [s for s, _, _ in resumed]
    avg_a = sum(la[s] for s in rng) / len(rng)
    avg_b = sum(lb[s] for s in rng) / len(rng)
    d_avg = abs(avg_a - avg_b)
    print('    avg loss steps 76-150: uninterrupted %.4f · resumed %.4f · '
          '|diff| = %.4f (bar < 0.15)' % (avg_a, avg_b, d_avg))
    assert d_avg < 0.15, 'resumed average deviates too much'

    # -- acceptance metric 2: no resumed step above running max + 0.5
    running_max = -math.inf
    worst = ('', -math.inf)
    ok = True
    for s in rng:
        a_here = la[s]
        running_max = max(running_max, a_here)
        excess = lb[s] - running_max
        if excess > worst[1]:
            worst = ('step %d: resumed %.4f vs uninterrupted-running-max %.4f'
                     % (s, lb[s], running_max), excess)
        if excess > 0.5:
            ok = False
    print('    spike check: worst resumed-vs-running-max excess = %+.4f '
          '(bar <= +0.5) at %s' % (worst[1], worst[0]))
    assert ok, 'loss spike after resume'

    worst_pair = side_by_side([(s, 'train', la[s]) for s in rng],
                              [(s, 'train', lb[s]) for s in rng])
    print('    worst per-step |diff| over 76-150: %.4f' % worst_pair)
    print('\n    >>> VERDICT: RESUME CLEAN  ✓')
    return True


def small_scale_proof(rows_a):
    tr = [r for r in rows_a if r[1] == 'train']
    losses = [l for _, _, l in tr]
    assert all(math.isfinite(l) for l in losses), 'non-finite loss in run A'
    first25 = [l for s, _, l in tr if s <= 25]
    last25 = [l for s, _, l in tr if s > STEPS - 25]
    f, l = sum(first25) / len(first25), sum(last25) / len(last25)
    print('\n================ SMALL-SCALE PROOF (run A) ================')
    print('    %d logged train steps, all finite  ✓' % len(losses))
    print('    first-25 avg loss = %.4f   last-25 avg loss = %.4f   '
          '(must fall) %s' % (f, l, '✓' if l < f else '✗'))
    assert l < f
    print('    loss every 25 steps:')
    for s, k, v in tr:
        if s % 25 == 0:
            print('      step %3d · loss %.4f' % (s, v))
    vals = [r for r in rows_a if r[1] == 'val']
    for s, _, v in vals:
        print('      val   %3d · loss %.4f' % (s, v))
    return f, l


def _available_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def speed_test():
    """5 real Config B steps at the largest batch THIS box's RAM allows.
    MEASURED memory (this suite): B=8 ctx=512 peaks ~3.6 GB RSS, B=16 ~7 GB
    (OOM-killed on this 4 GB box) — B=16 fits Colab's ~12.7 GB VM. We measure
    B=4/B=2 locally and extrapolate; the notebook prints its own numbers."""
    print('\n================ CONFIG B SPEED ESTIMATE ================')
    avail = _available_gb()
    batches = (16, 8) if avail >= 8.0 else ((4, 2) if avail >= 3.0 else (2,))
    print('    MemAvailable: %.1f GB -> measuring batches %r' % (avail, batches))
    tokver = tl.tokenizer_version_from_file(MERGES)
    data = make_data(M03.CONFIG_B['ctx'], max_val_windows=2)
    model = M03.Transformer(M03.CONFIG_B, seed=42)
    print('    Config B params: %d' % model.numParams())
    sd = Path('/tmp/p1_task5_speed')
    results = {}
    import gc
    import resource
    for B in batches:
        if sd.exists():
            shutil.rmtree(sd)
        try:
            tr = tl.Trainer(model, tokver, sd, data,
                            cfg={'lr': 1e-3, 'warmup': 1, 'total_steps': 5,
                                 'batch_size': B, 'clip': 1.0, 'eval_every': 0,
                                 'log_every': 1, 'ckpt_every_minutes': 1e9,
                                 'seed': SEED, 'final_save': False,
                                 'val_batch_windows': 1})
            hist = tr.train(max_steps=5)
        except MemoryError:
            print('    B=%d: MemoryError — skipped' % B)
            continue
        steps = [h for h in hist if h['kind'] == 'train']
        dts = [steps[0]['elapsed_s']] + \
              [b['elapsed_s'] - a['elapsed_s'] for a, b in zip(steps, steps[1:])]
        for h, dt in zip(steps, dts):
            print('      B=%d step %d · loss %.4f · %.2f s'
                  % (B, h['step'], h['loss'], dt))
        avg = sum(dts) / len(dts)
        tps = B * M03.CONFIG_B['ctx'] / avg
        spm = 60.0 / avg
        results[B] = (avg, tps, spm)
        print('    B=%d: avg %.2f s/step · %.0f tok/s (%.0f tok/min) · '
              '%.1f steps/min · 20,000-step projection ≈ %.1f h · '
              'peak RSS so far %.1f GB'
              % (B, avg, tps, tps * 60.0, spm, 20000 * avg / 3600.0,
                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
        gc.collect()
    shutil.rmtree(sd, ignore_errors=True)
    print('\n    (losses differ across B only because each batch is a different'
          ' draw — the model/seed are identical; this cell measures SPEED.)')
    print('    measured memory (this suite): B=8 ~3.6 GB RSS, B=16 ~7 GB '
          '(OOM-killed here); B=16 fits Colab\'s ~12.7 GB, B=8 is the safe '
          'notebook default.')
    if 4 in results:
        avg4, tps4, _ = results[4]
        print('    -> rough extrapolation to the notebook\'s batch (tok/s '
              'grows sub-linearly with B; treat as a RANGE):')
        for b, mult in ((8, (1.2, 1.8)), (16, (1.5, 2.5))):
            tok_lo, tok_hi = tps4 * mult[0], tps4 * mult[1]
            total = 20000 * b * M03.CONFIG_B['ctx']
            print('       B=%-2d ~%.0f-%.0f tok/s (%.0f-%.0f tok/min) -> '
                  '20,000 steps ≈ %.0f-%.0f h pure compute on THIS box '
                  '(%d cores, 4 GB)' % (b, tok_lo, tok_hi, tok_lo * 60.0,
                                        tok_hi * 60.0,
                                        total / (tok_hi * 3600.0),
                                        total / (tok_lo * 3600.0),
                                        os.cpu_count()))
    print('    (Colab CPU VMs differ — the notebook prints ITS own numbers '
          'in the first log rows; treat these as the honest local baseline.)')
    return results


def plot_check(dir_a, out_png):
    import matplotlib
    matplotlib.use('Agg')
    rows = read_csv(dir_a)
    hist = [{'kind': k, 'step': s, 'loss': l} for s, k, l in rows]
    fig = tl.plot_history(hist, logy=True,
                          title='p_1 · Task 5 smoke — tiny config, real BPE corpus')
    fig.savefig(out_png, dpi=130)
    print('\n================ CSV PLOT CHECK ================')
    print('    saved %s (%.1f KB) from %d CSV rows  ✓'
          % (out_png, out_png.stat().st_size / 1024.0, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='all',
                    choices=['all', 'run-a', 'run-b1', 'run-b2'])
    ap.add_argument('--ckpt-dir', default=None)
    args = ap.parse_args()

    if args.mode in ('run-a', 'run-b1', 'run-b2'):
        assert args.ckpt_dir, 'child modes need --ckpt-dir'
        mode_run(args.mode, args.ckpt_dir)
        return

    t0 = time.time()
    tokver = tl.tokenizer_version_from_file(MERGES)
    print('p_1 Task 5 local validation · numpy %s · %d CPUs'
          % (np.__version__, os.cpu_count()))
    print('tiny config: %r' % TINY)
    print('tokenizer_version = %r · corpus = %s' % (tokver, HEALTH_QA.name))

    data = make_data(TINY['ctx'])
    st = data.stats()
    print('data: %d train entries -> %d tokens | %d val entries -> %d tokens | '
          'val windows %d | ~%d windows/epoch at ctx %d'
          % (st['n_train_entries'], st['train_tokens'], st['n_val_entries'],
             st['val_tokens'], st['val_windows'],
             st['train_windows_per_epoch'], st['ctx']))

    dir_a = Path('/tmp/p1_task5_run_a')
    dir_b = Path('/tmp/p1_task5_run_b')
    for d in (dir_a, dir_b):
        shutil.rmtree(d, ignore_errors=True)

    # 1) RUN A — uninterrupted 150 steps, ckpt forced at 75
    run_child('run-a', dir_a)
    rows_a = read_csv(dir_a)
    small_scale_proof(rows_a)
    # NOTE: dir A's latest.txt points at the FINAL ckpt (step 150) — the
    # step-75 file proves the mid-run forced save itself.
    check_structure(dir_a / 'ckpt_step75.npz', 75, TINY, tokver,
                    expect_latest='ckpt_step150.npz')
    check_structure(dir_a / 'ckpt_step150.npz', 150, TINY, tokver,
                    expect_latest='ckpt_step150.npz')
    robustness_probes(dir_a, 'ckpt_step150.npz')

    # determinism cross-check: dir A's step-75 ckpt vs dir B1's step-75 ckpt
    # (byte-equal params) is asserted AFTER run B1 below.

    # 2) KILL / RESUME — B1 "crashes" at 75, B2 resumes in a new process
    run_child('run-b1', dir_b)
    check_structure(dir_b / 'ckpt_step75.npz', 75, TINY, tokver,
                    expect_latest='ckpt_step75.npz')
    za = np.load(str(dir_a / 'ckpt_step75.npz'))
    zb = np.load(str(dir_b / 'ckpt_step75.npz'))
    same = all(np.array_equal(za['param:' + n], zb['param:' + n])
               and np.array_equal(za['m:' + n], zb['m:' + n])
               and np.array_equal(za['v:' + n], zb['v:' + n])
               for n in M03.param_names(TINY))
    za.close()
    zb.close()
    print('\n    run A @75 vs run B1 @75 checkpoints byte-equal (params+m+v):',
          'YES  ✓ (determinism)' if same else 'NO  ✗')
    assert same

    run_child('run-b2', dir_b)
    check_structure(dir_b / 'ckpt_step150.npz', 150, TINY, tokver,
                    expect_latest='ckpt_step150.npz')
    kill_resume_test(dir_a, dir_b)

    # 3) CSV corruption check across the resume boundary
    rows_b = read_csv(dir_b)
    assert len(rows_b) == len(rows_a), \
        'row count differs across resume (%d vs %d)' % (len(rows_b), len(rows_a))

    # 4) Config B speed + 5) plot
    speed_test()
    plot_check(dir_a, REPO / 'scripts' / 'task5_loss_smoke.png')

    print('\n================ SUMMARY ================')
    print('    small-scale training proof ............. PASS (loss falls, all finite)')
    print('    checkpoint contract structure .......... PASS (keys/latest.txt/atomic)')
    print('    robustness probes (.tmp + stale latest)  PASS')
    print('    kill/resume in separate processes ...... PASS — RESUME CLEAN')
    print('    Config B speed ......................... see numbers above')
    print('    plot_history -> scripts/task5_loss_smoke.png  PASS')
    print('    total %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
