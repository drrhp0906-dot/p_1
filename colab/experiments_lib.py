"""
p_1 · colab/experiments_lib.py — Task 11: the experiment battery (the science)
--------------------------------------------------------------------------------
Reusable logic behind colab/09_experiments.ipynb. Pure Python + NumPy ONLY;
reuses the existing libs (never reimplements):

  train_lib    Trainer / build_data / checkpoints  (Task 5, resumable contract)
  03_model     Transformer CONFIG_A / CONFIG_B     (Task 4 contract)
  tokenizer_py P1BPE                               (Task 3)
  eval_lib     4 answer modes / mythVerdict / tau  (Task 9, p1-metrics-v1)
  retrieval_py BM25 index                          (Task 8 twin)

THE FOUR EXPERIMENTS
  E1  myth contamination A/B : same seed + schedule, clean vs contaminated
      corpus -> myth-parroting rate (eval_lib.mythVerdict) on the 20 traps.
  E2  grounding delta        : clean model, 3 eval sets x 4 modes
      (RAW/GROUNDED/VERIFIED/ABSTAINING) -> hallucination-rate table.
  E3  abstention tau sweep   : sweep ABSTAIN_TAU x {0.2 .. 1.5} (12 values);
      true-refusal on unanswerables vs false-refusal on answerables ->
      precision/recall/F1, pick max-F1 operating point, compare vs the
      Task 9 hardcoded tau. NOTE: the tau decision is model-INDEPENDENT
      (best top-1 BM25 + threshold), so E3 reuses E2's per-question records
      (falls back to pure retrieval if they are missing) — no regeneration.
  E4  seed variance          : BABY config x seeds (42/7/1234) on the clean
      corpus -> RAW vs GROUNDED hallucination rate per seed -> mean ± spread.

BABY-CONFIG CHOICE (documented, per the Task 11 brief "Config A dims,
char-level or BPE-tiny — pick what trains in minutes"):
  We pick BPE-tiny: Config A's transformer dims (n_layer 2, n_head 4,
  d_model 128, d_ff 512, ctx 256) but vocab_size 4096, so the ONE tokenizer
  (corpus/merges.json) and the whole eval machinery apply unchanged; a
  char-level baby (vocab 96) would need a second tokenizer + data pipeline
  for zero scientific gain. Measured on the dev box: trains in minutes on CPU.
  In LOCAL_SMOKE the baby shrinks further (d_model 64 / d_ff 128 / ctx 128)
  — smoke proves CODE PATHS, not numbers.

RESUMABILITY / HONESTY CONTRACT
  * run_training() auto-resumes: a checkpoint at/after the target step is
    REUSED (no retrain); a partial checkpoint CONTINUES (Trainer's always-on
    resume); a config/tokenizer mismatch goes through Trainer's loud
    fresh-start path.
  * Every payload is JSON-safe and saved to results/ so a restarted kernel
    (or a re-run cell) picks up where it left off.
  * No number is invented: if an experiment fails, its section says so and
    the rest of the battery continues.
"""

import datetime
import importlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np                       # engine contract: f32 tensors

# sibling imports (this file lives in colab/ next to the other libs)
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_lib as tl                     # noqa: E402
import eval_lib as el                      # noqa: E402
import retrieval_py as rp                  # noqa: E402
from tokenizer_py import P1BPE             # noqa: E402

M03 = importlib.import_module('03_model')  # module name starts with a digit

VERSION = '1.0.0'
FORMAT = 'p1-experiments-v1'

# ============================================================================
# MODE KNOBS — the notebook sets exactly one of these per run
# ============================================================================

MODES = ('smoke', 'full')

# E1/E2 model: tiny (path proof) vs the real Config B sibling (04_train contract)
MODEL_CONFIGS = {
    'smoke': {'n_layer': 2, 'n_head': 2, 'd_model': 64, 'd_ff': 128,
              'ctx': 128, 'vocab_size': 4096},          # == train_lib.TINY_CONFIG
    'full': None,                                       # -> M03.CONFIG_B at call time
}

# E4 baby: Config A dims + BPE vocab (see docstring); smoke shrinks the dims
BABY_CONFIGS = {
    'smoke': {'n_layer': 2, 'n_head': 2, 'd_model': 64, 'd_ff': 128,
              'ctx': 128, 'vocab_size': 4096},
    'full': {'n_layer': 2, 'n_head': 4, 'd_model': 128, 'd_ff': 512,
             'ctx': 256, 'vocab_size': 4096},
}

# Schedules. smoke = the train_lib selftest tiny recipe (measured 0.17 s/step);
# full  = the 04_train.ipynb CONFIG_B contract (seed 42, B=4, lr 3e-4,
# warmup 200, total_steps 6000) — the A/B arms share it IDENTICALLY.
SCHEDULES = {
    'smoke': {'lr': 1e-3, 'warmup': 20, 'batch_size': 8},
    'full': {'lr': 3e-4, 'warmup': 200, 'batch_size': 4},
}

# Steps per experiment kind. E4 full: 3 seeds x 800 steps (minutes per seed).
STEPS = {
    'smoke': {'ab': 150, 'seed': 60},
    'full': {'ab': 6000, 'seed': 800},
}

SEEDS = (42, 7, 1234)                  # E4 seeds (42 = the canonical project seed)
BASE_SEED = 42                         # E1 A/B seed — IDENTICAL arms
DATA_SEED = 1234                       # pack/stream seed (train_lib contract)

TAU_MULTIPLIERS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
MAX_NEW_TOKENS = 80                    # brief: max 80 new tokens (08_eval contract)
TEMPERATURE = 0.7                      # greedy argmax; scaling honors the spec


def model_config(mode):
    """E1/E2 model config for a mode (fresh dict each call)."""
    if mode == 'full':
        return dict(M03.CONFIG_B)
    return dict(MODEL_CONFIGS[mode])


def baby_config(mode):
    """E4 baby config for a mode (fresh dict each call)."""
    return dict(BABY_CONFIGS[mode])


def schedule(mode):
    """Train schedule for a mode (fresh dict each call)."""
    return dict(SCHEDULES[mode])


def steps_for(mode, kind):
    """Training steps: kind 'ab' (E1/E2) or 'seed' (E4)."""
    return int(STEPS[mode][kind])


# ============================================================================
# small utilities
# ============================================================================

def default_merges():
    """corpus/merges.json via the retrieval lib's data-dir convention."""
    return rp.data_dir() / 'merges.json'


def jsonable(o):
    """Recursively coerce a payload into strict JSON types (numpy scalars,
    tuples, Paths). Payloads are written to results/ and must survive a
    kernel restart."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, str)):
        return o
    if isinstance(o, int):
        return int(o)
    if isinstance(o, float):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    if o is None:
        return None
    if hasattr(o, 'item'):                    # numpy scalar
        return jsonable(o.item())
    return str(o)                             # last resort: keep the info


def save_payload(payload, path):
    """JSON-dump a payload (format-tagged, atomic rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault('format', FORMAT)
    body['experiments_lib_version'] = VERSION
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(jsonable(body), f, ensure_ascii=False, indent=1)
        f.write('\n')
    os.replace(str(tmp), str(path))
    return path


def load_payload(path):
    """Read a payload back (None if missing/unreadable) — resume path."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        traceback.print_exc()
        return None


def md_table(headers, rows):
    """Markdown table (the four summary tables all go through this)."""
    def cell(v):
        s = '' if v is None else str(v)
        return s.replace('|', '\\|').replace('\n', ' ')
    out = ['| ' + ' | '.join(cell(h) for h in headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(cell(v) for v in r) + ' |')
    return '\n'.join(out)


def fmt_pct(x, nd=1):
    return 'n/a' if x is None else ('%.' + str(nd) + 'f%%') % (100.0 * float(x))


def fmt3(x):
    return 'n/a' if x is None else '%.3f' % float(x)


# ============================================================================
# TRAINING (E1 arms + E4 seeds) — resumable, honest
# ============================================================================

def run_training(corpus_path, config, steps, seed, ckpt_dir, *,
                 schedule=None, data_seed=DATA_SEED, merges_path=None,
                 label='', log=print):
    """Train (or REUSE/CONTINUE) one model.

    Returns {'model', 'tok', 'tokver', 'config', 'step', 'ckpt', 'resumed',
             'final_train_loss', 'val_loss', 'wall_s', 'data_stats',
             'summary'} — 'model' is the live Transformer; 'summary' is the
    JSON-safe slice that goes into payloads.
    """
    schedule = dict(schedule or SCHEDULES['smoke'])
    corpus_path = Path(corpus_path)
    merges_path = Path(merges_path) if merges_path else default_merges()
    if not corpus_path.exists():
        raise FileNotFoundError('training corpus not found: %s' % corpus_path)

    tok = P1BPE(str(merges_path))
    tokver = tl.tokenizer_version_from_file(str(merges_path))
    ckpt_dir = Path(ckpt_dir)

    # ---- already done? (kernel restart / notebook re-run) ------------------
    prov = tl.resume_provenance(str(ckpt_dir))
    if prov and 'error' not in prov and prov.get('config') == dict(config) \
            and int(prov.get('step', 0)) >= int(steps):
        model = M03.Transformer(dict(config), seed=seed)   # params replaced below
        opt = tl.AdamW(lr=0.0)                             # moment slots only
        ckpt = tl.find_resume(str(ckpt_dir))
        loaded = tl.load_checkpoint(ckpt, model, opt)
        log('[exp] %s: %s already at step %d >= %d — REUSED (no retrain)'
            % (label or 'run', Path(ckpt).name, loaded['step'], steps))
        vloss = _val_loss_of(model, tok, corpus_path, config, data_seed)
        return _finish_run(model=model, tok=tok, tokver=tokver, config=config,
                           step=loaded['step'], ckpt=str(ckpt), resumed=True,
                           final_train_loss=None, val_loss=vloss, wall_s=0.0,
                           data_stats=None, label=label)

    # ---- fresh or continue (Trainer auto-resume is always on) --------------
    t0 = time.time()
    data = tl.build_data(corpus_path, tok, int(config['ctx']), seed=data_seed,
                         max_val_windows=8)
    st = data.stats()
    log('[exp] %s: corpus %s -> %d train / %d val entries, %d train tokens'
        % (label or 'run', corpus_path.name, st['n_train_entries'],
           st['n_val_entries'], st['train_tokens']))
    model = M03.Transformer(dict(config), seed=seed)
    tcfg = {'lr': schedule['lr'], 'warmup': schedule['warmup'],
            'total_steps': int(steps), 'batch_size': schedule['batch_size'],
            'clip': 1.0, 'eval_every': 0, 'log_every': max(1, int(steps) // 10),
            'seed': seed, 'ckpt_every_minutes': 600.0, 'final_save': True,
            'keep_last': 1}
    trainer = tl.Trainer(model, tokver, str(ckpt_dir), data, cfg=tcfg)
    trainer.train(max_steps=int(steps))
    vloss, _tps = trainer.eval_val()
    rows = [r for r in trainer.history if r['kind'] == 'train']
    final_loss = rows[-1]['loss'] if rows else None
    wall = time.time() - t0
    ckpt = tl.find_resume(str(ckpt_dir))
    log('[exp] %s: step %d · final train loss %s · val loss %.4f · %.1fs wall'
        % (label or 'run', trainer.step,
           'n/a' if final_loss is None else '%.4f' % final_loss, vloss, wall))
    return _finish_run(model=model, tok=tok, tokver=tokver, config=config,
                       step=trainer.step, ckpt=str(ckpt),
                       resumed=trainer.resume_info is not None,
                       final_train_loss=final_loss, val_loss=vloss, wall_s=wall,
                       data_stats=st, label=label)


def _val_loss_of(model, tok, corpus_path, config, data_seed):
    """Val loss for a REUSED checkpoint (rebuild the same fixed val windows)."""
    try:
        data = tl.build_data(corpus_path, tok, int(config['ctx']),
                             seed=data_seed, max_val_windows=8)
        total, toks = 0.0, 0
        ctx = int(config['ctx'])
        for x, y, _n in data.val_batches(8):
            logits = model.forward(x.reshape(-1), len(x) // ctx)
            loss = logits.crossEntropy(y.reshape(-1))
            total += loss.item() * x.size
            toks += x.size
        return (total / toks) if toks else None
    except Exception:
        traceback.print_exc()
        return None


def _finish_run(**kw):
    label = kw.pop('label', '')
    summary = {
        'label': label,
        'config': dict(kw['config']),
        'step': int(kw['step']),
        'ckpt': kw['ckpt'],
        'resumed': bool(kw['resumed']),
        'final_train_loss': kw['final_train_loss'],
        'val_loss': kw['val_loss'],
        'wall_s': kw['wall_s'],
        'data_stats': kw['data_stats'],
    }
    return dict(kw, summary=summary)


def load_trained(ckpt_dir, *, merges_path=None, seed=0, log=print):
    """Load the newest checkpoint into a fresh Transformer (kernel-restart
    path for E2 when E1's in-memory model is gone). Same run-dict shape as
    run_training; None if there is no checkpoint."""
    merges_path = Path(merges_path) if merges_path else default_merges()
    ckpt = tl.find_resume(str(ckpt_dir))
    if ckpt is None:
        return None
    info = tl.resume_provenance(str(ckpt_dir))
    if 'error' in info:
        raise RuntimeError('unreadable checkpoint in %s: %r' % (ckpt_dir, info))
    config = info['config']
    tok = P1BPE(str(merges_path))
    model = M03.Transformer(dict(config), seed=seed)
    opt = tl.AdamW(lr=0.0)
    loaded = tl.load_checkpoint(ckpt, model, opt)
    live_tokver = tl.tokenizer_version_from_file(str(merges_path))
    if loaded['tokenizer_version'] != live_tokver:
        log('!!! LOUD WARNING: ckpt tokenizer_version %r != live merges %r '
            '(ids may be misaligned)' % (loaded['tokenizer_version'], live_tokver))
    log('[exp] loaded %s (step %d) from %s'
        % (Path(ckpt).name, loaded['step'], ckpt_dir))
    return _finish_run(model=model, tok=tok, tokver=live_tokver, config=config,
                       step=loaded['step'], ckpt=str(ckpt), resumed=True,
                       final_train_loss=None, val_loss=None, wall_s=0.0,
                       data_stats=None, label='loaded')


# ============================================================================
# ANSWERING + EVAL (eval_lib does the measuring; we only wire the brain in)
# ============================================================================

def make_model_answerer(model, tok, *, max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE):
    """Greedy answerer over a trained brain — the EXACT 08_eval.ipynb recipe:
    prompt = bos + 'Q: <q>\nA:', left-truncated ctx window, argmax (temp 0.7
    scaling is argmax-invariant, kept to honor the spec), newline-stopped,
    '(no answer tokens)' fallback."""
    ctx = int(model.config['ctx'])

    def answerer(question: str) -> str:
        prompt = 'Q: %s\nA:' % question
        ids = ([tok.bos] + tok.encode(prompt))[-ctx:]
        new_ids = []
        for _ in range(int(max_new_tokens)):
            window = np.asarray(ids[-ctx:], dtype=np.int64)
            logits = model.forward(window, 1)                 # [T, vocab]
            last = logits.data.reshape(logits.shape[-2:]).astype('float64')[-1]
            last = last / float(temperature)
            nxt = int(last.argmax())                          # greedy
            if nxt == tok.eos:
                break
            ids.append(nxt)
            new_ids.append(nxt)
        text = tok.decode(new_ids).split('\n')[0].strip()
        return text if text else '(no answer tokens)'

    answerer.planted = {}
    answerer.kind = 'model'
    return answerer


def eval_model(model, sets, *, index, tok=None, tau=el.ABSTAIN_TAU,
               max_new_tokens=MAX_NEW_TOKENS, log=print):
    """Evaluate one model over `sets` ({'answerable': rows, ...}, any subset)
    through eval_lib's 4-mode harness. Returns (metrics, records) exactly as
    el.evaluate_sets — the single yardstick for every experiment table."""
    answerer = make_model_answerer(model, tok, max_new_tokens=max_new_tokens)
    return el.evaluate_sets(index, answerer, sets, tau=tau, log=log)


# ============================================================================
# E1 · MYTH CONTAMINATION A/B
# ============================================================================

def myth_ab_eval(index, answerer, traps, *, tau=el.ABSTAIN_TAU):
    """Parroting measurement for ONE answerer over the myth-trap rows, via
    eval_lib's mythVerdict machinery. The RAW mode is the contamination
    signal (the model's own words); the other 3 modes are context."""
    per_trap = []
    per_mode_hits = {m: 0 for m in el.MODES}
    examples = []
    for row in traps:
        q = row['question']
        modes = el.answer_all_modes(q, answerer, index, tau)
        myth = {}
        for m in el.MODES:
            mv = el.mythVerdict(modes[m]['answer'], row['folk_belief'],
                                row['correction'])
            myth[m] = mv
            if mv['parroting']:
                per_mode_hits[m] += 1
                if m == 'raw' and len(examples) < 3:
                    examples.append({
                        'id': row.get('id'), 'question': q,
                        'answer': str(modes[m]['answer'])[:220],
                        'folk_f1': mv['folk_f1'],
                        'correction_f1': mv['correction_f1']})
        per_trap.append({'id': row.get('id'), 'question': q,
                         'raw_answer': str(modes['raw']['answer'])[:220],
                         'myth': myth})
    n = len(traps) or 1
    return {
        'n': len(traps),
        'parroting_rate': per_mode_hits['raw'] / n,
        'n_parroted_raw': per_mode_hits['raw'],
        'per_mode_rate': {m: per_mode_hits[m] / n for m in el.MODES},
        'examples': examples,
        'per_trap': per_trap,
    }


def run_exp1(index, traps, *, corpora, mode, ckpt_root, tau=el.ABSTAIN_TAU,
             merges_path=None, log=print):
    """E1 end-to-end: train BOTH arms (identical seed/schedule, corpora
    differ) and measure myth-parroting on the trap set.
    Returns (payload, runs) — runs holds the live models (clean is reused by
    E2); payload is JSON-safe and saved by the notebook."""
    config = model_config(mode)
    steps = steps_for(mode, 'ab')
    sched = schedule(mode)
    ckpt_root = Path(ckpt_root)
    log('[exp1] myth contamination A/B · mode=%s · config=%s · steps=%d · '
        'seed=%d (IDENTICAL arms; only the corpus differs)'
        % (mode, json.dumps(jsonable(config)), steps, BASE_SEED))
    runs = {}
    results = {}
    for name in ('clean', 'contaminated'):
        run = run_training(corpora[name], config, steps, BASE_SEED,
                           ckpt_root / ('exp1_%s' % name), schedule=sched,
                           merges_path=merges_path,
                           label='exp1-%s' % name, log=log)
        runs[name] = run
        answerer = make_model_answerer(run['model'], run['tok'])
        log('[exp1] measuring myth-parroting of the %s model on %d traps...'
            % (name, len(traps)))
        results[name] = myth_ab_eval(index, answerer, traps, tau=tau)
    payload = {
        'experiment': 'exp1_myth_ab',
        'mode': mode,
        'seed': BASE_SEED,
        'steps': steps,
        'schedule': sched,
        'corpora': {'clean': Path(corpora['clean']).name,
                    'contaminated': Path(corpora['contaminated']).name},
        'traps': len(traps),
        'runs': {k: v['summary'] for k, v in runs.items()},
        'results': results,
        'expected_direction_holds':
            results['contaminated']['parroting_rate']
            >= results['clean']['parroting_rate'],
    }
    if mode == 'smoke':
        payload['n_traps_note'] = (
            'Smoke run: %d steps on a tiny model — both brains are near-babble, '
            'so rates are noisy; we only assert the MEASUREMENT exists (see '
            'notes for non-experts).' % steps)
    return payload, runs


def myth_ab_table(payload):
    """(headers, rows) for the A/B summary table: model x parroting rate."""
    headers = ['model', 'corpus', 'parroted (of %d)' % payload['traps'],
               'parroting rate', 'example quote (RAW)']
    rows = []
    for name, disp in (('clean', 'clean'), ('contaminated', 'contaminated')):
        r = payload['results'][name]
        ex = (r['examples'][0]['answer'][:110] + '…') if r['examples'] else '—'
        rows.append([disp, payload['corpora'][name], r['n_parroted_raw'],
                     fmt_pct(r['parroting_rate']), ex])
    return headers, rows


def myth_ab_modewise_table(payload):
    """Bonus table: per-mode parroting rates for both models."""
    headers = ['mode', 'clean parroting', 'contaminated parroting']
    rows = [[m,
             fmt_pct(payload['results']['clean']['per_mode_rate'][m]),
             fmt_pct(payload['results']['contaminated']['per_mode_rate'][m])]
            for m in el.MODES]
    return headers, rows


# ============================================================================
# E2 · GROUNDING DELTA
# ============================================================================

def run_exp2(index, sets, clean_run, *, mode, tau=el.ABSTAIN_TAU,
             max_new_tokens=MAX_NEW_TOKENS, log=print):
    """E2 end-to-end: the CLEAN model, 3 eval sets x 4 modes via eval_lib.
    Returns (payload, metrics, records). records feed E3's sweep."""
    log('[exp2] grounding delta · clean model from E1 (step %d) · %s'
        % (clean_run['step'],
           {k: len(v) for k, v in sets.items()}))
    metrics, records = eval_model(clean_run['model'], sets, index=index,
                                  tok=clean_run['tok'], tau=tau,
                                  max_new_tokens=max_new_tokens, log=log)
    payload = {
        'experiment': 'exp2_grounding_delta',
        'mode': mode,
        'answerer': 'clean model (exp1 checkpoint, step %d)' % clean_run['step'],
        'metrics': jsonable(metrics),
        'records': jsonable(records),
        'best_bm25': jsonable(best_bm25_from_records(records)),
    }
    return payload, metrics, records


def grounding_delta_table(metrics):
    """(headers, rows): 3 sets x 4 modes hallucination rates (+ refusals)."""
    headers = ['eval set', 'mode', 'hallucination rate', 'refusal rate',
               'parroting rate']
    rows = []
    sets = metrics['sets']
    for set_name in ('answerable', 'mythtraps', 'unanswerable'):
        pm = sets.get(set_name, {}).get('per_mode', {})
        for m in el.MODES:
            e = pm.get(m)
            if e is None:
                rows.append([set_name, m, '—', '—', '—'])
                continue
            par = e.get('parroting_rate')
            rows.append([set_name, m, fmt3(e.get('hallucination_rate')),
                         fmt_pct(e.get('refusal_rate')),
                         fmt_pct(par) if par is not None else '—'])
    return headers, rows


def best_bm25_from_records(records):
    """Per-question top-1 BM25 scores from evaluate_sets records — the
    model-independent substrate of the tau sweep (E3)."""
    out = {}
    for set_name, per_q in records.items():
        out[set_name] = [float(q['modes']['abstaining']['best_bm25'])
                         for q in per_q]
    return out


# ============================================================================
# E3 · ABSTENTION THRESHOLD SWEEP
# ============================================================================

def best_bm25_from_index(index, rows):
    """Fallback substrate: top-1 BM25 straight from the Task 8 index
    (the sweep is model-independent — same numbers either way)."""
    out = []
    for r in rows:
        top = index['search'](r['question'], k=1, method='bm25')
        out.append(float(top[0]['score']) if top else 0.0)
    return out


def tau_sweep(unanswerable_bests, answerable_bests, *, calibrated_tau,
              multipliers=TAU_MULTIPLIERS, hardcoded_tau=el.ABSTAIN_TAU):
    """Sweep tau = multiplier * calibrated median over ~12 values.

    Semantics (documented): a question is REFUSED iff best top-1 BM25 < tau —
    the tau-driven component of the ABSTAINING mode (the final mode also
    inherits GROUNDED's low-retrieval refusals; that part is tau-independent
    and excluded here so the sweep isolates tau's own tradeoff).
      TP = refused unanswerables (true refusal),  FN = answered unanswerables
      FP = refused answerables   (false refusal), TN = answered answerables
    Best operating point = max F1; ties break toward the calibrated tau
    (deterministic, documented).
    """
    rows = []
    for mult in multipliers:
        tau = float(calibrated_tau) * float(mult)
        tp = sum(1 for s in unanswerable_bests if s < tau)
        fp = sum(1 for s in answerable_bests if s < tau)
        fn = len(unanswerable_bests) - tp
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) \
            if (precision + recall) else 0.0
        rows.append({'multiplier': float(mult), 'tau': tau,
                     'true_refusals': int(tp),
                     'n_unanswerable': len(unanswerable_bests),
                     'false_refusals': int(fp),
                     'n_answerable': len(answerable_bests),
                     'precision': precision, 'recall': recall, 'f1': f1})
    best = max(rows, key=lambda r: (r['f1'],
                                    -abs(r['tau'] - float(calibrated_tau))))
    return {'calibrated_tau': float(calibrated_tau),
            'hardcoded_tau': float(hardcoded_tau),
            'tau_matches_task9':
                abs(float(calibrated_tau) - float(hardcoded_tau)) < 5e-4,
            'multipliers': [float(m) for m in multipliers],
            'rows': rows, 'best': best,
            'note': ('refusal iff best top-1 BM25 < tau (the tau-driven part '
                     'of ABSTAINING); best = max F1, ties broken toward the '
                     'calibrated tau; model-independent (retrieval-only)')}


def run_exp3(index, answerable_rows, unanswerable_rows, e2_payload=None, *,
             mode, log=print):
    """E3 end-to-end: sweep tau, pick max-F1 operating point, compare vs the
    hardcoded Task 9 tau. Reuses E2's per-question best-BM25 scores when the
    payload is available; otherwise (or if a set is missing) falls back to
    pure retrieval — the sweep is model-independent."""
    cal = float(el.calibrate_tau(index, answerable_rows))
    log('[exp3] abstention threshold sweep · calibrated median tau re-derived '
        '= %.6f (hardcoded %.4f, match %s)'
        % (cal, el.ABSTAIN_TAU, abs(cal - el.ABSTAIN_TAU) < 5e-4))
    substrate = {}
    bb = (e2_payload or {}).get('best_bm25') or {}
    if bb.get('unanswerable') and bb.get('answerable'):
        substrate['unanswerable'] = [float(v) for v in bb['unanswerable']]
        substrate['answerable'] = [float(v) for v in bb['answerable']]
        source = 'E2 records'
    else:
        substrate['unanswerable'] = best_bm25_from_index(index, unanswerable_rows)
        substrate['answerable'] = best_bm25_from_index(index, answerable_rows)
        source = 'retrieval-only fallback'
    log('[exp3] substrate: %d unanswerable + %d answerable best-BM25 scores (%s)'
        % (len(substrate['unanswerable']), len(substrate['answerable']), source))
    sweep = tau_sweep(substrate['unanswerable'], substrate['answerable'],
                      calibrated_tau=cal)
    b = sweep['best']
    log('[exp3] operating point: tau=%.3f (%.2fx calibrated) · precision %.3f · '
        'recall %.3f · F1 %.3f | hardcoded tau F1 = %.3f'
        % (b['tau'], b['multiplier'], b['precision'], b['recall'], b['f1'],
           next(r['f1'] for r in sweep['rows'] if r['multiplier'] == 1.0)))
    payload = {
        'experiment': 'exp3_tau_sweep',
        'mode': mode,
        'substrate': source,
        'n_unanswerable': len(substrate['unanswerable']),
        'n_answerable': len(substrate['answerable']),
        **sweep,
    }
    return payload


def tau_sweep_table(sweep):
    """(headers, rows) for the sweep table; the best row is starred."""
    headers = ['×calibrated', 'tau', 'true refusals (unans.)',
               'false refusals (ans.)', 'precision', 'recall', 'F1']
    rows = []
    for r in sweep['rows']:
        star = ' ← best F1' if r is sweep['best'] else ''
        rows.append(['%.2f%s' % (r['multiplier'], star), '%.3f' % r['tau'],
                     '%d/%d' % (r['true_refusals'], r['n_unanswerable']),
                     '%d/%d' % (r['false_refusals'], r['n_answerable']),
                     '%.3f' % r['precision'], '%.3f' % r['recall'],
                     '%.3f' % r['f1']])
    return headers, rows


# ============================================================================
# E4 · SEED VARIANCE
# ============================================================================

def seed_variance_summarize(rows):
    """mean ± spread over per-seed metric rows (spread = max − min, the
    honest range for n=3; population std also reported)."""
    def stats(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return {'mean': None, 'spread': None, 'std': None}
        return {'mean': sum(vals) / len(vals),
                'spread': max(vals) - min(vals),
                'std': statistics.pstdev(vals) if len(vals) > 1 else 0.0}
    return {'raw_hallucination_rate': stats('raw_hallucination_rate'),
            'grounded_hallucination_rate': stats('grounded_hallucination_rate'),
            'spread_definition':
                'spread = max − min over seeds (n=%d); std = population'
                % len(rows)}


def run_exp4(index, answerable_rows, *, corpus_path, mode, ckpt_root,
             tau=el.ABSTAIN_TAU, seeds=SEEDS, merges_path=None, log=print):
    """E4 end-to-end: train the BABY config with 3 seeds on the CLEAN corpus;
    per seed measure RAW vs GROUNDED hallucination on the answerable set.
    Returns (payload, rows)."""
    cfg = baby_config(mode)
    steps = steps_for(mode, 'seed')
    sched = schedule(mode)
    ckpt_root = Path(ckpt_root)
    log('[exp4] seed variance · baby config=%s · steps=%d/seed · seeds=%s · '
        'corpus=%s' % (json.dumps(jsonable(cfg)), steps, list(seeds),
                       Path(corpus_path).name))
    rows = []
    for seed in seeds:
        run = run_training(corpus_path, cfg, steps, int(seed),
                           ckpt_root / ('exp4_seed%d' % seed), schedule=sched,
                           merges_path=merges_path,
                           label='exp4-seed%d' % seed, log=log)
        metrics, _records = eval_model(
            run['model'], {'answerable': answerable_rows}, index=index,
            tok=run['tok'], tau=tau, log=log)
        pm = metrics['sets']['answerable']['per_mode']
        row = {
            'seed': int(seed),
            'wall_s': run['summary']['wall_s'],
            'step': run['summary']['step'],
            'val_loss': run['summary']['val_loss'],
            'n_questions': len(answerable_rows),
            'raw_hallucination_rate': pm['raw']['hallucination_rate'],
            'grounded_hallucination_rate': pm['grounded']['hallucination_rate'],
        }
        log('[exp4] seed %d: RAW %.3f · GROUNDED %.3f'
            % (seed, row['raw_hallucination_rate'],
               row['grounded_hallucination_rate']))
        rows.append(row)
    payload = {
        'experiment': 'exp4_seed_variance',
        'mode': mode,
        'config': cfg,
        'steps_per_seed': steps,
        'corpus': Path(corpus_path).name,
        'seeds': [int(s) for s in seeds],
        'rows': rows,
        'summary': seed_variance_summarize(rows),
    }
    return payload, rows


def seed_variance_table(payload):
    """(headers, rows): per-seed RAW vs GROUNDED hallucination + summary."""
    headers = ['seed', 'train wall (s)', 'RAW halluc rate',
               'GROUNDED halluc rate', 'Δ (raw − grounded)']
    rows = []
    for r in payload['rows']:
        delta = (r['raw_hallucination_rate'] - r['grounded_hallucination_rate']) \
            if (r.get('raw_hallucination_rate') is not None
                and r.get('grounded_hallucination_rate') is not None) else None
        rows.append([r['seed'],
                     ('%.0f' % r['wall_s']) if r.get('wall_s') is not None else '—',
                     fmt3(r['raw_hallucination_rate']),
                     fmt3(r['grounded_hallucination_rate']),
                     fmt3(delta)])
    s = payload['summary']
    for key, disp in (('raw_hallucination_rate', 'RAW'),
                      ('grounded_hallucination_rate', 'GROUNDED')):
        st = s[key]
        if st['mean'] is not None:
            rows.append(['%s mean ± spread' % disp, '—',
                         '%.3f ± %.3f' % (st['mean'], st['spread']),
                         '%.3f ± %.3f' % (st['mean'], st['spread']),
                         'spread = max−min'])
    return headers, rows


# ============================================================================
# SUMMARY WRITER — results/summary.md
# ============================================================================

def guess_headline(results):
    """One-line answer to 'does grounding reduce hallucination in a model we
    built from scratch?' from the E2 metrics (honest when missing)."""
    exp2 = results.get('exp2') or {}
    metrics = exp2.get('metrics') or {}
    pm = metrics.get('sets', {}).get('answerable', {}).get('per_mode', {})
    raw = pm.get('raw', {}).get('hallucination_rate')
    grn = pm.get('grounded', {}).get('hallucination_rate')
    ver = pm.get('verified', {}).get('hallucination_rate')
    if raw is None or grn is None:
        return ('INCOMPLETE — the grounding-delta experiment did not finish, '
                'so the research question is unanswered by this file.')
    if grn < raw:
        cut = (1 - grn / raw) * 100 if raw > 0 else 0.0
        return ('YES — grounding cut the hallucination rate from %.3f (RAW) '
                'to %.3f (GROUNDED), a %.0f%% relative reduction; verified %.3f. '
                'Measured on a transformer we trained from raw arithmetic.'
                % (raw, grn, cut, ver if ver is not None else float('nan')))
    if grn == raw:
        return ('NO REDUCTION MEASURED — RAW and GROUNDED hallucination rates '
                'are identical (%.3f) in this run.' % raw)
    return ('NOT IN THIS RUN — GROUNDED hallucination rate %.3f came out ABOVE '
            'RAW %.3f (small-n noise or an undertrained model; see the notes).'
            % (grn, raw))


def write_summary_md(path, *, mode, results, provenance):
    """Write results/summary.md: title + one-line answer + FOUR tables +
    notes for non-experts + provenance. `results` = {'exp1': payload|None,
    'exp2': ..., 'exp3': ..., 'exp4': ...}; a missing experiment renders as
    an explicit '(did not complete)' — never silently skipped."""
    stub = (mode == 'smoke')
    L = []
    L.append('# p_1 · experiment summary (Task 11)')
    L.append('')
    L.append('**Does grounding reduce hallucination in a model we built from '
             'scratch?**')
    L.append('')
    L.append('**%s**' % guess_headline(results))
    L.append('')
    if stub:
        L.append('> ⚠ **STUB RUN (LOCAL_SMOKE = True)** — tiny configs, few '
                 'steps. This file proves every code path and the table '
                 'SHAPES; the numbers come from undertrained toys and carry '
                 'no scientific meaning. Re-run the notebook with '
                 'LOCAL_SMOKE = False in Colab for the real battery.')
        L.append('')
    else:
        L.append('> Real battery (full configs). Corpus: curated health Q&A; '
                 'model trained from scratch — a toy science demo, not '
                 'medical advice.')
        L.append('')

    # ---- experiment 1 ------------------------------------------------------
    L.append('## Experiment 1 · myth contamination A/B')
    L.append('')
    e1 = results.get('exp1')
    if not e1:
        L.append('_Experiment did not complete — see the notebook output for '
                 'the traceback._')
    else:
        L.append('Two siblings, identical seed (%s) and schedule, differing '
                 'ONLY in the training corpus (clean vs +300 planted myth '
                 'pairs). Parroting = the model echoes the folk belief on the '
                 '%d myth-trap questions (eval_lib.mythVerdict; RAW mode).'
                 % (e1.get('seed', BASE_SEED), e1.get('traps', 20)))
        L.append('')
        headers, rows = myth_ab_table(e1)
        L.append(md_table(headers, rows))
        L.append('')
        hold = e1.get('expected_direction_holds')
        note = (' %s' % e1['n_traps_note']) if e1.get('n_traps_note') else ''
        L.append('Expected direction (contaminated ≥ clean): **%s**.%s'
                 % ('holds' if hold else 'does NOT hold in this run', note))
        L.append('')
        h2, r2 = myth_ab_modewise_table(e1)
        L.append('Per-mode parroting rates (context):')
        L.append('')
        L.append(md_table(h2, r2))
    L.append('')

    # ---- experiment 2 ------------------------------------------------------
    L.append('## Experiment 2 · grounding delta (the headline)')
    L.append('')
    e2 = results.get('exp2')
    if not e2:
        L.append('_Experiment did not complete — see the notebook output._')
    else:
        headers, rows = grounding_delta_table(e2['metrics'])
        L.append(md_table(headers, rows))
        L.append('')
        summ = e2['metrics'].get('summary', {})
        if summ.get('headline'):
            L.append('eval_lib headline: _%s_' % summ['headline'])
    L.append('')

    # ---- experiment 3 ------------------------------------------------------
    L.append('## Experiment 3 · abstention threshold sweep')
    L.append('')
    e3 = results.get('exp3')
    if not e3:
        L.append('_Experiment did not complete — see the notebook output._')
    else:
        headers, rows = tau_sweep_table(e3)
        L.append(md_table(headers, rows))
        L.append('')
        b = e3['best']
        f1_at_default = None
        for r in e3['rows']:
            if abs(r['multiplier'] - 1.0) < 1e-9:
                f1_at_default = r['f1']
        L.append('**Operating point: max F1 = %.3f at tau ≈ %.3f (%.2f× the '
                 'calibrated median %.3f).** Task 9\'s hardcoded tau %.4f '
                 '(calibration match: %s) sits at F1 %s — it buys recall but '
                 'refuses %d/%d answerable questions.'
                 % (b['f1'], b['tau'], b['multiplier'], e3['calibrated_tau'],
                    e3['hardcoded_tau'],
                    'yes' if e3['tau_matches_task9'] else 'NO',
                    ('%.3f' % f1_at_default) if f1_at_default is not None
                    else 'n/a',
                    b['false_refusals'], b['n_answerable']))
    L.append('')

    # ---- experiment 4 ------------------------------------------------------
    L.append('## Experiment 4 · seed variance')
    L.append('')
    e4 = results.get('exp4')
    if not e4:
        L.append('_Experiment did not complete — see the notebook output._')
    else:
        headers, rows = seed_variance_table(e4)
        L.append(md_table(headers, rows))
        L.append('')
        s = e4['summary']
        if s['raw_hallucination_rate']['mean'] is not None:
            L.append('Reading: the RAW column moves by ±%.3f across seeds '
                     '(mean %.3f) and the GROUNDED column by ±%.3f (mean '
                     '%.3f) — that is the measurement noise floor any '
                     'SINGLE-seed number should be read against. (%s)'
                     % (s['raw_hallucination_rate']['spread'],
                        s['raw_hallucination_rate']['mean'],
                        s['grounded_hallucination_rate']['spread'],
                        s['grounded_hallucination_rate']['mean'],
                        s['spread_definition']))
    L.append('')

    # ---- notes for non-experts ---------------------------------------------
    L.append('## Notes for non-experts')
    L.append('')
    L.append('- **Hallucination rate** = the share of the model\'s answer '
             'statements that our rule-based verifier could NOT find support '
             'for in the retrieved corpus. It is a lexical yardstick, not '
             'ground truth — but it is the SAME yardstick for every mode, so '
             'comparisons are fair.')
    L.append('- **RAW** answers come straight from the model; **GROUNDED** '
             'answers are assembled only from retrieved corpus sentences '
             '(with citations); **VERIFIED** checks each RAW claim and '
             'rewrites or flags failures; **ABSTAINING** additionally '
             'refuses when retrieval looks too weak (score below tau).')
    L.append('- **Myth-parroting** = answering a folk-belief trap the way the '
             'myth wants it answered. The A/B siblings differ ONLY in '
             'training data — if contamination moves the parroting rate, '
             'corpus hygiene matters.')
    L.append('- **The tau sweep** trades "refusing things we could have '
             'answered" (false refusals) against "answering things we should '
             'have refused" (missed refusals). F1 balances both; the sweep '
             'shows the Task 9 median-tau default is recall-heavy.')
    L.append('- **Seed variance** says how much of any number here is luck of '
             'the draw: with only 3 seeds the spread is a rough noise floor, '
             'not a confidence interval.')
    L.append('- An undertrained model (like the stub run) babbles: many claims '
             'score near zero and land in the verifier\'s "not-in-corpus" '
             'bucket, which is EXCLUDED from the hallucination rate — so '
             'read stub numbers as plumbing, not psychology.')
    L.append('')

    # ---- provenance --------------------------------------------------------
    L.append('## Provenance')
    L.append('')
    L.append(md_table(['field', 'value'], sorted(provenance.items())))
    L.append('')
    body = '\n'.join(L) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    return body


def provenance_block(*, mode, model_cfg, baby_cfg, steps_ab, steps_seed,
                     seeds, seed_base, corpora, tau, results_dir,
                     max_new_tokens, extra=None):
    """The provenance dict rendered at the bottom of summary.md."""
    block = {
        'run mode': ('LOCAL_SMOKE (stub)' if mode == 'smoke'
                     else 'FULL (real battery)'),
        'stub_or_real': 'stub' if mode == 'smoke' else 'real',
        'date (UTC)': datetime.datetime.now(
            datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'experiments_lib': VERSION,
        'eval_lib': el.VERSION,
        'model config (E1/E2)': json.dumps(jsonable(model_cfg)),
        'baby config (E4)': json.dumps(jsonable(baby_cfg)),
        'E1/E2 steps': str(steps_ab),
        'E4 steps per seed': str(steps_seed),
        'seeds (E4)': str(list(seeds)),
        'A/B seed (E1)': str(seed_base),
        'train schedule': json.dumps(jsonable(schedule(mode))),
        'corpora': corpora,
        'abstain tau (Task 9)': '%.4f (calibrated median top-1 BM25)' % tau,
        'max new tokens': str(max_new_tokens),
        'results dir': str(results_dir),
    }
    if extra:
        block.update(extra)
    return block


# ============================================================================
# DEMO / SELF-CHECK (python3 colab/experiments_lib.py)
# ============================================================================

def _demo():
    print('experiments_lib v%s (format %s)' % (VERSION, FORMAT))
    print('smoke model cfg :', model_config('smoke'))
    print('full model cfg  :', model_config('full'))
    print('smoke baby cfg  :', baby_config('smoke'))
    print('full baby cfg   :', baby_config('full'))
    print('schedules       :', SCHEDULES)
    print('steps           :', STEPS)
    print('seeds           :', SEEDS, '| tau multipliers:', TAU_MULTIPLIERS)
    # tau math on synthetic scores: 10 unanswerable (low), 10 answerable (high)
    sw = tau_sweep([5.0 * i for i in range(10)],
                   [40.0 + 5.0 * i for i in range(10)], calibrated_tau=20.0)
    print('synthetic sweep best:', {k: (round(v, 3) if isinstance(v, float)
                                        else v)
                                   for k, v in sw['best'].items()})
    # summary writer on EMPTY results (path-proof: missing experiments render)
    body = write_summary_md(
        '/tmp/t11_demo_summary.md', mode='smoke',
        results={'exp1': None, 'exp2': None, 'exp3': None, 'exp4': None},
        provenance={'stub_or_real': 'stub', 'date (UTC)': 'demo'})
    assert 'did not complete' in body and 'STUB RUN' in body
    print('empty-results summary written to /tmp/t11_demo_summary.md '
          '(path proof)')
    print('OK — demo ran (pure logic; corpus runs live in 09_experiments.ipynb)')
    return 0


if __name__ == '__main__':
    raise SystemExit(_demo())
