"""
p_1 · colab/train_lib.py — Task 5: the overnight trainer (AdamW + schedule +
clip + BPE data pipeline + resumable Trainer)
--------------------------------------------------------------------------------
Pure Python + NumPy ONLY. No Colab-specific imports (the Drive mount lives in
the notebook), no ML libraries. Consumes the Task 4 engine/model contract:

  model : object with .forward(flat_ids, B) -> logits Tensor [B*T, vocab],
          .params() / .paramNames() (canonical order), .zeroGrad(), .config
  param : engine Tensor with .data (flat f32), .grad (flat f32), .shape,
          .ndim, .name
  tok   : P1BPE (needs .encode(text), .bos, .eos)

CHECKPOINT CONTRACT (Task 7's export script reads these — DO NOT DEVIATE)
  <ckpt_dir>/ckpt_step{N}.npz — np.savez_compressed, written ATOMICALLY
  (file opened as ckpt_step{N}.npz.tmp, then os.replace()):
    step               np.int64 scalar — last COMPLETED optimizer step
    config             UTF-8 JSON string of the full model config dict
    tokenizer_version  UTF-8 string "p1-bpe-v1:<merges_count>"
    param:<name>       float32 array (param's logical shape), one per
                       model.paramNames() entry, canonical order
    m:<name>           float32 AdamW first moment (same shape)
    v:<name>           float32 AdamW second moment (same shape)
  <ckpt_dir>/latest.txt — just the newest checkpoint FILENAME, also written
  atomically (.tmp + os.replace()).
  Auto-resume: newest checkpoint (latest.txt first, glob-max-step fallback)
  restores step / config / params / moments. tokenizer_version is asserted
  against the live merges file — on mismatch: LOUD warning + fresh start,
  and the offending file is renamed *.mismatch-tokver so it cannot poison
  every future resume. A *.npz.tmp leftover (crash mid-write) is ignored by
  design — the previous checkpoint is always intact (kill -9 safe).

AdamW (from scratch)
  bias correction uses a timestep that is RESTORED on resume (t == ckpt
  step — one optimizer step per train step); decoupled weight decay 0.01
  applied ONLY to 2-D params (param.ndim == 2 — every bias and LayerNorm
  g/b is 1-D and exempt; param groups are implicit). Math in float64, one
  float32 rounding per param per step (engine house style).

Schedule  linear warmup (default 200 steps) to peak_lr (3e-4), then cosine
          decay to peak_lr * 0.1 over total_steps (20000). Callable lr(step).
Clipping  global norm: L2 over ALL grad arrays (f64 accumulation), scale =
          min(1, clip / norm), applied in float32, in place.

Determinism  the batch for step k is drawn from
np.random.default_rng([seed, k]) — data order is a pure function of
(seed, step), so a resumed run replays the same data an uninterrupted run
would have seen at the same step. Validation windows are fixed
(non-overlapping, evenly spaced if capped) and never consume RNG.

NaN policy (p1_TASKS.md): loss NaN → lower LR; verify gradient clipping
actually applied; crossEntropy must clamp log(0) (it subtracts the row max).
Any non-finite loss or grad RAISES with that remedy in the message — no
silent NaN ever reaches a checkpoint.

Memory hygiene (fix added in Task 4-b-cont): the engine's per-node _backward
closures are reference CYCLES by design (mirrors docs/tensor.js). Nearly all
step allocations are refcount-freed, so CPython's cyclic-GC counter rarely
reaches its threshold — dead graphs pile up, each still holding its f64
intermediates (measured on this box: tiny-config RSS 316 MB → 2.4 GB over 70
steps, then cgroup OOM-kill mid-run; with an explicit collect, RSS is flat).
The Trainer therefore runs gc.collect() once per train step and after every
val eval — negligible next to one forward+backward, and the difference
between an overnight run surviving and dying.

__main__ CLI (added by Task 4-b-cont; minimal, documented here)
  python3 colab/train_lib.py selftest --steps N --tiny --ckpt-dir DIR [--resume]
  Quick in-process sanity: REAL corpus + REAL BPE (paths via p1_root() /
  P1_DATA_DIR) through the tiny trainer config (n_layer 2, d_model 64,
  ctx 128, B 8, lr 1e-3, warmup 20, clip 1.0; model seed 42, data seed 1234 —
  deterministic). Auto-resume is ALWAYS on (contract). --resume only makes
  the EXPECTATION explicit: it fails loudly if no checkpoint was resumed.
  Drop --tiny to run the same probe on full CONFIG_B (slow — CPU). Exits 0
  and prints "SELFTEST OK" on success. The full acceptance suite (kill/
  resume across processes) lives in scripts/test_task5_train.py; this CLI is
  the 30-second smoke, not the gate.
"""

import atexit
import gc
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# paths: P1_DATA_DIR env  >  /content/drive/MyDrive/p_1  >  repo-relative
# --------------------------------------------------------------------------

DEFAULT_BOS = 4094      # specials from the Task 3 contract (merges.json also
DEFAULT_EOS = 4095      # carries them; build_data prefers tok.bos/tok.eos)

CKPT_GLOB = 'ckpt_step*.npz'
CSV_HEADER = 'step,kind,loss,lr,tokens_per_sec,elapsed_s,grad_norm'


def p1_root():
    """Root that holds corpus/ and checkpoints/ — first match wins."""
    env = os.environ.get('P1_DATA_DIR')
    if env:
        return Path(env)
    drive = Path('/content/drive/MyDrive/p_1')
    if drive.is_dir():
        return drive
    return Path(__file__).resolve().parent.parent      # repo root (colab/..)


def corpus_dir():
    return p1_root() / 'corpus'


def default_ckpt_dir():
    return p1_root() / 'checkpoints'


def tokenizer_version_from_file(merges_path):
    """'p1-bpe-v1:3838' — format id + merge count from merges.json."""
    with open(merges_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return '%s:%d' % (obj.get('format', 'p1-bpe-v1'), len(obj['merges']))


# --------------------------------------------------------------------------
# AdamW — from scratch
# --------------------------------------------------------------------------

class AdamW:
    """Decoupled-weight-decay Adam over named engine params.

    - per-param float32 m/v (checkpoint contract), math in float64 with one
      f32 rounding per param per step.
    - timestep-aware across resumes: self.t increments once per step() and
      is restored by load_state_dict(), so bias correction continues
      exactly where a crashed run stopped.
    - decoupled weight decay, 2-D params only (param.ndim == 2): biases and
      LayerNorm g/b (1-D) are exempt. Groups are implicit — no config.
    """

    def __init__(self, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0                      # optimizer timestep (restored on resume)
        self.m = {}                     # name -> flat f32 array
        self.v = {}                     # name -> flat f32 array

    def _slots(self, name, size):
        if name not in self.m:
            self.m[name] = np.zeros(size, dtype=np.float32)
            self.v[name] = np.zeros(size, dtype=np.float32)
        return self.m[name], self.v[name]

    def step(self, params, grads=None, lr=None):
        """params: list of engine Tensors (uses .name, .ndim, .data, .grad).
        grads: optional list aligned with params (default: param.grad).
        lr: optional per-call override (the schedule's value)."""
        self.t += 1
        lr = self.lr if lr is None else float(lr)
        bc1 = 1.0 - self.beta1 ** self.t          # bias corrections for THIS t
        bc2 = 1.0 - self.beta2 ** self.t
        for i, p in enumerate(params):
            g = p.grad if grads is None else grads[i]
            g = np.asarray(g, dtype=np.float64).reshape(-1)
            m, v = self._slots(p.name, p.size)
            m64 = m.astype(np.float64)
            v64 = v.astype(np.float64)
            m64 = self.beta1 * m64 + (1.0 - self.beta1) * g
            v64 = self.beta2 * v64 + (1.0 - self.beta2) * (g * g)
            mhat = m64 / bc1
            vhat = v64 / bc2
            upd = mhat / (np.sqrt(vhat) + self.eps)
            p64 = p.data.astype(np.float64)
            if p.ndim == 2 and self.weight_decay > 0.0:      # 2-D only
                p64 = p64 - lr * self.weight_decay * p64      # decoupled: from OLD p
            p64 = p64 - lr * upd
            p.data = p64.astype(np.float32)                   # one f32 rounding
            m[:] = m64.astype(np.float32)
            v[:] = v64.astype(np.float32)
        return self

    def zero_state(self):
        self.t = 0
        self.m.clear()
        self.v.clear()

    # ---------------------------------------------------------- state io

    def state_dict(self):
        """{'t': int, 'm': {name: f32 array}, 'v': {name: f32 array}} —
        the m/v dicts are the {name: arrays} the checkpoint stores."""
        return {'t': int(self.t),
                'm': {k: np.asarray(a, dtype=np.float32) for k, a in self.m.items()},
                'v': {k: np.asarray(a, dtype=np.float32) for k, a in self.v.items()}}

    def load_state_dict(self, sd):
        self.t = int(sd.get('t', 0))
        self.m = {k: np.asarray(a, dtype=np.float32).reshape(-1)
                  for k, a in sd.get('m', {}).items()}
        self.v = {k: np.asarray(a, dtype=np.float32).reshape(-1)
                  for k, a in sd.get('v', {}).items()}
        return self


# --------------------------------------------------------------------------
# LR schedule + gradient clipping
# --------------------------------------------------------------------------

class CosineWarmup:
    """lr(step): linear warmup 0->peak over `warmup_steps`, then cosine decay
    to peak*min_ratio (0.1) at `total_steps`, held there after. Callable."""

    def __init__(self, warmup_steps=200, peak_lr=3e-4, total_steps=20000,
                 min_ratio=0.1):
        self.warmup_steps = int(warmup_steps)
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.min_ratio = float(min_ratio)

    def __call__(self, step):
        s = int(step)
        w = max(1, self.warmup_steps)
        if s <= w:
            return self.peak_lr * s / w
        span = max(1, self.total_steps - w)
        prog = min(1.0, max(0.0, (s - w) / span))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.peak_lr * (self.min_ratio + (1.0 - self.min_ratio) * cos)


def global_grad_norm(params):
    """L2 norm over ALL grad arrays, accumulated in float64."""
    total = 0.0
    for p in params:
        g = p.grad
        total += float(np.square(g.astype(np.float64)).sum())
    return math.sqrt(total)


def clip_grads_global(params, clip):
    """Scale = min(1, clip/norm) applied IN PLACE in float32. Returns
    (raw_norm, applied_scale) — the raw norm goes to the CSV for debugging."""
    norm = global_grad_norm(params)
    scale = 1.0
    if norm > clip:
        scale = clip / max(norm, 1e-12)
        s32 = np.float32(scale)
        for p in params:
            p.grad *= s32                      # float32 in place
    return norm, scale


# --------------------------------------------------------------------------
# Data pipeline: health_qa.jsonl -> BPE id streams -> windows
# --------------------------------------------------------------------------

def load_health_qa(jsonl_path):
    """Returns (train_texts, val_texts); text = 'Q: <question>\\nA: <answer>'."""
    train, val = [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            text = 'Q: %s\nA: %s' % (o['question'], o['answer'])
            (val if o.get('split', 'train') == 'val' else train).append(text)
    return train, val


def pack_texts(texts, tok, bos=None, eos=None):
    """Each entry becomes [bos] + encode('Q: ..\\nA: ..') + [eos]; all entries
    are concatenated into ONE long id stream (np.int32). bos/eos come from
    the tokenizer when it exposes them."""
    if bos is None:
        bos = int(getattr(tok, 'bos', DEFAULT_BOS))
    if eos is None:
        eos = int(getattr(tok, 'eos', DEFAULT_EOS))
    ids = []
    enc = tok.encode
    for t in texts:
        ids.append(bos)
        ids.extend(enc(t))
        ids.append(eos)
    return np.asarray(ids, dtype=np.int32)


class PackedData:
    """Packed train/val id streams + window logic.

    Train windows: random starts — the batch for step k comes from
    default_rng([seed, k]) so resumed runs replay identical data.
    Val windows:   FIXED non-overlapping windows (starts 0, ctx, 2*ctx, ...),
    capped at max_val_windows by even spacing across the stream.
    Targets are the window shifted by one inside the stream.
    """

    def __init__(self, train_stream, val_stream, ctx, seed=1234,
                 max_val_windows=32, n_train_entries=0, n_val_entries=0):
        self.train_stream = train_stream
        self.val_stream = val_stream
        self.ctx = int(ctx)
        self.seed = int(seed)
        self.n_train_entries = int(n_train_entries)
        self.n_val_entries = int(n_val_entries)
        self.train_tokens = int(train_stream.size)
        self.val_tokens = int(val_stream.size)
        max_start = self.train_tokens - self.ctx - 1
        if max_start < 0:
            raise ValueError('train stream (%d tokens) shorter than ctx+1 (%d)'
                             % (self.train_tokens, self.ctx + 1))
        self._max_start = max_start
        starts = list(range(0, max(1, self.val_tokens - self.ctx), self.ctx))
        if len(starts) > max_val_windows:            # even spacing, deterministic
            idx = np.linspace(0, len(starts) - 1, max_val_windows).round().astype(int)
            starts = [starts[i] for i in dict.fromkeys(idx.tolist())]
        self.val_starts = starts

    def train_batch(self, step, B):
        """->[x, y] int arrays shaped [B, ctx]; y = x shifted by 1 in-stream.
        Deterministic per (seed, step)."""
        rng = np.random.default_rng([self.seed, int(step)])
        starts = rng.integers(0, self._max_start + 1, size=int(B))
        ctx, st = self.ctx, self.train_stream
        x = np.stack([st[s:s + ctx] for s in starts])
        y = np.stack([st[s + 1:s + 1 + ctx] for s in starts])
        return x, y

    def val_batches(self, windows_per_forward=8):
        """Yields (x_flat, y_flat, n_windows) over the fixed val windows."""
        ctx, st = self.ctx, self.val_stream
        for i in range(0, len(self.val_starts), int(windows_per_forward)):
            chunk = self.val_starts[i:i + int(windows_per_forward)]
            x = np.concatenate([st[s:s + ctx] for s in chunk])
            y = np.concatenate([st[s + 1:s + 1 + ctx] for s in chunk])
            yield x, y, len(chunk)

    def stats(self):
        return {'n_train_entries': self.n_train_entries,
                'n_val_entries': self.n_val_entries,
                'train_tokens': self.train_tokens,
                'val_tokens': self.val_tokens,
                'ctx': self.ctx,
                'train_windows_per_epoch': self.train_tokens // self.ctx,
                'val_windows': len(self.val_starts),
                'pack_seed': self.seed}


def build_data(jsonl_path, tok, ctx, seed=1234, max_val_windows=32):
    """Load + tokenize + pack. Returns a PackedData."""
    train_texts, val_texts = load_health_qa(jsonl_path)
    tr = pack_texts(train_texts, tok)
    va = pack_texts(val_texts, tok)
    return PackedData(tr, va, ctx, seed=seed, max_val_windows=max_val_windows,
                      n_train_entries=len(train_texts),
                      n_val_entries=len(val_texts))


# --------------------------------------------------------------------------
# Checkpoints — atomic write, atomic latest.txt, validated resume
# --------------------------------------------------------------------------

def _as_str(np_scalar):
    a = np.asarray(np_scalar)
    v = a.item() if a.shape == () else a.reshape(-1)[0]
    if isinstance(v, bytes):
        v = v.decode('utf-8')
    return str(v)


def _ckpt_step(path):
    name = Path(path).name
    return int(name[len('ckpt_step'):-len('.npz')])


def find_resume(ckpt_dir):
    """Newest checkpoint path, or None. Preference: latest.txt (if it names
    an existing file), else max step from glob. .tmp leftovers are ignored
    (the glob only matches *.npz)."""
    d = Path(ckpt_dir)
    latest = d / 'latest.txt'
    if latest.exists():
        name = latest.read_text(encoding='utf-8').strip()
        if name:
            p = d / name
            if p.exists() and p.name.endswith('.npz'):
                return p
    cands = sorted(d.glob(CKPT_GLOB))
    if not cands:
        return None
    return max(cands, key=_ckpt_step)


def save_checkpoint(model, optimizer, step, config, tokenizer_version,
                    ckpt_dir, keep_last=0):
    """Write ckpt_step{N}.npz ATOMICALLY (tmp + os.replace), then latest.txt
    atomically. Contract keys: step / config / tokenizer_version /
    param:<name> / m:<name> / v:<name> (float32, logical param shapes)."""
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    arrays = {'step': np.int64(step),
              'config': json.dumps(dict(config)),
              'tokenizer_version': str(tokenizer_version)}
    names = model.paramNames()
    params = model.params()
    if len(names) != len(params):
        raise AssertionError('paramNames/params length mismatch')
    for name, p in zip(names, params):
        shp = list(p.shape)
        arrays['param:' + name] = np.asarray(p.data, dtype=np.float32).reshape(shp)
        m = optimizer.m.get(name)
        v = optimizer.v.get(name)
        arrays['m:' + name] = (np.zeros(p.size, dtype=np.float32) if m is None
                               else np.asarray(m, dtype=np.float32).reshape(shp))
        arrays['v:' + name] = (np.zeros(p.size, dtype=np.float32) if v is None
                               else np.asarray(v, dtype=np.float32).reshape(shp))
    final = d / ('ckpt_step%d.npz' % int(step))
    tmp = Path(str(final) + '.tmp')
    with open(tmp, 'wb') as fh:                    # explicit handle: np.savez
        np.savez_compressed(fh, **arrays)          # must not append '.npz'
    os.replace(str(tmp), str(final))               # ATOMIC — kill -9 safe
    latest = d / 'latest.txt'
    ltmp = Path(str(latest) + '.tmp')
    with open(ltmp, 'w', encoding='utf-8') as fh:
        fh.write(final.name)
        fh.flush()
    os.replace(str(ltmp), str(latest))
    if keep_last and int(keep_last) > 0:           # newest N checkpoints kept
        cands = sorted(d.glob(CKPT_GLOB), key=_ckpt_step, reverse=True)
        for old in cands[int(keep_last):]:
            try:
                old.unlink()
            except OSError:
                pass
    return final


def load_checkpoint(path, model, optimizer):
    """Restore step/config/params/moments from a contract npz. Returns
    {'step', 'config', 'tokenizer_version'}. Raises on param-name/shape
    mismatch (a hard contract error, not a resume-mismatch)."""
    z = np.load(str(path), allow_pickle=False)
    step = int(z['step'])
    config = json.loads(_as_str(z['config']))
    tokver = _as_str(z['tokenizer_version'])
    names = model.paramNames()
    for name, p in zip(names, model.params()):
        key = 'param:' + name
        if key not in z.files:
            raise KeyError('checkpoint %s missing %r (has %d param keys)'
                           % (Path(path).name, name,
                              sum(1 for k in z.files if k.startswith('param:'))))
        arr = np.asarray(z[key], dtype=np.float32).reshape(-1)
        if arr.size != p.size:
            raise ValueError('checkpoint %s: %s has %d elements, model wants %d'
                             % (Path(path).name, key, arr.size, p.size))
        p.data = arr.copy()
        p.grad.fill(0)
        mkey, vkey = 'm:' + name, 'v:' + name
        optimizer.m[name] = (np.asarray(z[mkey], dtype=np.float32).reshape(-1)
                             if mkey in z.files
                             else np.zeros(p.size, dtype=np.float32))
        optimizer.v[name] = (np.asarray(z[vkey], dtype=np.float32).reshape(-1)
                             if vkey in z.files
                             else np.zeros(p.size, dtype=np.float32))
    optimizer.t = step        # one optimizer step per train step -> t == step
    z.close()
    return {'step': step, 'config': config, 'tokenizer_version': tokver}


def resume_provenance(ckpt_dir=None):
    """What the notebook prints: last checkpoint step + save time (or None
    if there is nothing to resume from)."""
    d = Path(ckpt_dir) if ckpt_dir else default_ckpt_dir()
    p = find_resume(d)
    if p is None:
        return None
    try:
        z = np.load(str(p), allow_pickle=False)
        info = {'path': str(p),
                'name': p.name,
                'step': int(z['step']),
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S',
                                          time.localtime(p.stat().st_mtime)),
                'tokenizer_version': _as_str(z['tokenizer_version']),
                'config': json.loads(_as_str(z['config']))}
        z.close()
        return info
    except Exception as e:                         # unreadable newest ckpt
        return {'path': str(p), 'name': p.name, 'error': repr(e)}


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

class Trainer:
    """Resumable training loop for the Task 4 transformer.

    cfg keys (defaults):
      lr 3e-4 (peak) · warmup 200 · total_steps 20000 · batch_size 16 ·
      clip 1.0 · eval_every 500 · ckpt_every_minutes 15 · log_every 50 ·
      seed 42 (data-order seed) · weight_decay 0.01 ·
      val_batch_windows 8 · keep_last 0 (keep ALL checkpoints) ·
      ckpt_at_steps None (extra forced-save steps; used by tests) ·
      final_save True (save on graceful end)

    Auto-resumes from ckpt_dir on construction (validating tokenizer_version
    and config). train(max_minutes=None, max_steps=None) — max_steps is an
    ABSOLUTE step target (a resumed run at step 75 with max_steps=150 does
    75 more steps). Returns this session's history rows:
      {'kind': 'train'|'val', step, loss, lr, tokens_per_sec, elapsed_s,
       grad_norm}.
    """

    def __init__(self, model, tokenizer_version, ckpt_dir, data, cfg=None):
        self.model = model
        self.tokenizer_version = str(tokenizer_version)
        self.ckpt_dir = Path(ckpt_dir)
        self.data = data
        c = dict(cfg or {})
        self.cfg = {
            'lr': 3e-4, 'warmup': 200, 'total_steps': 20000,
            'batch_size': 16, 'clip': 1.0, 'eval_every': 500,
            'ckpt_every_minutes': 15.0, 'log_every': 50, 'seed': 42,
            'weight_decay': 0.01, 'val_batch_windows': 8,
            'keep_last': 0, 'ckpt_at_steps': None, 'final_save': True,
        }
        for k, v in c.items():
            if k not in self.cfg:
                raise KeyError('Trainer cfg: unknown key %r (known: %r)'
                               % (k, sorted(self.cfg)))
            self.cfg[k] = v
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(lr=self.cfg['lr'],
                               weight_decay=self.cfg['weight_decay'])
        self.schedule = CosineWarmup(warmup_steps=self.cfg['warmup'],
                                     peak_lr=self.cfg['lr'],
                                     total_steps=self.cfg['total_steps'])
        self.step = 0                     # last COMPLETED step
        self.history = []
        self.csv_path = self.ckpt_dir / 'train_log.csv'
        self.resume_info = None
        self._dirty = False
        self._last_ckpt = None

        # ---- auto-resume (validated) -------------------------------------
        init_state = {p.name: p.data.copy() for p in model.params()}
        path = find_resume(self.ckpt_dir)
        if path is not None:
            info = load_checkpoint(path, model, self.optimizer)
            reason = None
            if info['tokenizer_version'] != self.tokenizer_version:
                reason = ('tokenizer_version %r in checkpoint != current %r'
                          % (info['tokenizer_version'], self.tokenizer_version))
                suffix = '.mismatch-tokver'
            elif info['config'] != dict(model.config):
                reason = ('config %r in checkpoint != model config %r'
                          % (info['config'], dict(model.config)))
                suffix = '.mismatch-config'
            if reason is not None:
                self._warn_fresh(reason, path, suffix, init_state)
            else:
                self.resume_info = dict(info, path=str(path), name=Path(path).name,
                                        saved_at=time.strftime(
                                            '%Y-%m-%d %H:%M:%S',
                                            time.localtime(Path(path).stat().st_mtime)))
                self.step = info['step']
        else:
            print('[trainer] no checkpoint in %s — fresh start' % self.ckpt_dir)

        self._params = model.params()
        atexit.register(self._atexit_save)
        # Provenance + the single most important resume signal: the val loss
        # RIGHT NOW, printed so a clean resume is visible at a glance.
        vloss = self.eval_val()[0] if self.data.val_starts else float('nan')
        if self.resume_info:
            print('[trainer] RESUMED %s at step %d (saved %s, tokver %s)'
                  % (self.resume_info['name'], self.step,
                     self.resume_info.get('saved_at', '?'),
                     self.resume_info['tokenizer_version']))
            print('[trainer] val loss right after resume: %.4f  '
                  '(a clean resume continues this value — no spike)' % vloss)
        else:
            print('[trainer] fresh start · pre-training val loss: %.4f '
                  '(~ln(vocab)=%.2f means untrained)' % (vloss, math.log(
                      self.model.config['vocab_size'])))

    # -------------------------------------------------------------- resume

    def _warn_fresh(self, reason, path, suffix, init_state):
        print('!' * 72)
        print('!!! RESUME REFUSED — STARTING FRESH !!!')
        print('!!! %s' % reason)
        print('!!! checkpoint: %s' % path)
        print('!!! restoring freshly-initialized params; the offending file is')
        print('!!! renamed *%s so it cannot poison future resumes.' % suffix)
        print('!' * 72)
        for p in self.model.params():              # back to init weights
            p.data = init_state[p.name].copy()
            p.grad.fill(0)
        self.optimizer.zero_state()
        self.step = 0
        try:
            os.replace(str(path), str(path) + suffix)
        except OSError:
            pass
        # keep latest.txt pointing at a VALID checkpoint (or remove it) so a
        # fresh start never leaves a dangling pointer behind
        try:
            lt = self.ckpt_dir / 'latest.txt'
            nxt = find_resume(self.ckpt_dir)
            if nxt is None:
                lt.unlink(missing_ok=True)
            else:
                ltmp = Path(str(lt) + '.tmp')
                ltmp.write_text(Path(nxt).name, encoding='utf-8')
                os.replace(str(ltmp), str(lt))
        except OSError:
            pass

    def _atexit_save(self):
        if not self._dirty:
            return
        try:
            self.save_now()
            print('[trainer] atexit: final checkpoint saved at step %d'
                  % self.step)
        except Exception as e:                     # never mask the real error
            print('[trainer] atexit final save FAILED: %r' % (e,))

    # ------------------------------------------------------------ eval/csv

    def eval_val(self):
        """Mean cross-entropy over the FIXED val windows (no update)."""
        total, toks = 0.0, 0
        t0 = time.time()
        B = self.cfg['val_batch_windows']
        ctx = self.data.ctx
        for x, y, _n in self.data.val_batches(B):
            logits = self.model.forward(x.reshape(-1), len(x) // ctx)
            loss = logits.crossEntropy(y.reshape(-1))
            total += loss.item() * x.size
            toks += x.size
        dt = max(1e-9, time.time() - t0)
        gc.collect()          # val graph is a cycle pile too — free it now
        return (total / toks if toks else float('nan')), toks / dt

    def _csv_append(self, row):
        new = not self.csv_path.exists()
        with open(self.csv_path, 'a', encoding='utf-8') as f:
            if new:
                f.write(CSV_HEADER + '\n')
            f.write('%d,%s,%.6f,%.8g,%.2f,%.1f,%s\n'
                    % (row['step'], row['kind'], row['loss'], row['lr'],
                       row['tokens_per_sec'], row['elapsed_s'],
                       '' if row.get('grad_norm') is None
                       else '%.4f' % row['grad_norm']))
            f.flush()                              # per row — crash safe

    def save_now(self):
        """Force an atomic checkpoint RIGHT NOW (also used by tests and the
        ckpt_at_steps hook)."""
        path = save_checkpoint(self.model, self.optimizer, self.step,
                               self.model.config, self.tokenizer_version,
                               self.ckpt_dir, keep_last=self.cfg['keep_last'])
        self._dirty = False
        self._last_ckpt = {'step': self.step, 'path': str(path),
                           'time': time.time()}
        return path

    # --------------------------------------------------------------- train

    def train(self, max_minutes=None, max_steps=None):
        cfg = self.cfg
        model, data, opt = self.model, self.data, self.optimizer
        B = int(cfg['batch_size'])
        ctx = data.ctx
        tokens_per_step = B * ctx
        target = int(max_steps) if max_steps is not None else int(cfg['total_steps'])
        interval = float(cfg['ckpt_every_minutes']) * 60.0
        ckpt_at = set(int(s) for s in (cfg['ckpt_at_steps'] or []))
        t_start = time.time()
        next_ckpt = t_start + interval
        last_log_t, last_log_step = time.time(), self.step
        interrupted = False

        print('[trainer] training to step %d (batch %d x ctx %d = %d tok/step)'
              % (target, B, ctx, tokens_per_step))
        try:
            while self.step < target:
                if max_minutes is not None and \
                        (time.time() - t_start) >= max_minutes * 60.0:
                    print('[trainer] max_minutes budget reached at step %d'
                          % self.step)
                    break
                self.step += 1
                s = self.step

                x, y = data.train_batch(s, B)          # deterministic per step
                logits = model.forward(x.reshape(-1), B)   # [B*ctx, vocab]
                loss = logits.crossEntropy(y.reshape(-1))
                lval = loss.item()
                if not math.isfinite(lval):
                    raise RuntimeError(
                        'loss is %s at step %d — per p1_TASKS.md: LOWER THE LR '
                        "(cfg['lr'], now %g), verify gradient clipping is "
                        "applied (cfg['clip'], now %g), and confirm "
                        'crossEntropy subtracts the row max before log(). '
                        'No checkpoint of this step was written; the last '
                        'good checkpoint still resumes.'
                        % (lval, s, cfg['lr'], cfg['clip']))
                model.zeroGrad()
                loss.backward()
                for p in self._params:
                    if not np.all(np.isfinite(p.grad)):
                        raise RuntimeError(
                            'non-finite gradient in %r at step %d — per '
                            'p1_TASKS.md: lower the LR and verify global-norm '
                            'clipping (clip=%g). Loss was %.4f.'
                            % (p.name, s, cfg['clip'], lval))
                gnorm, _scale = clip_grads_global(self._params, cfg['clip'])
                lr = self.schedule(s)
                opt.step(self._params, lr=lr)
                self._dirty = True
                # Previous step's graph is unreachable by now (locals were
                # rebound) but is a reference CYCLE — see the module docstring
                # 'Memory hygiene' note for why this collect is mandatory.
                gc.collect()

                now = time.time()
                if s % int(cfg['log_every']) == 0 or s == target:
                    tps = ((s - last_log_step) * tokens_per_step
                           / max(1e-9, now - last_log_t))
                    row = {'kind': 'train', 'step': s, 'loss': lval, 'lr': lr,
                           'tokens_per_sec': tps, 'elapsed_s': now - t_start,
                           'grad_norm': gnorm}
                    self.history.append(row)
                    self._csv_append(row)
                    last_log_t, last_log_step = now, s
                    print('[train] step %6d · loss %.4f · lr %.3g · '
                          '%.0f tok/s · gn %.2f · %ds'
                          % (s, lval, lr, tps, gnorm, int(now - t_start)))

                ev = int(cfg['eval_every'])
                if ev and s % ev == 0:
                    vloss, vtps = self.eval_val()
                    row = {'kind': 'val', 'step': s, 'loss': vloss, 'lr': lr,
                           'tokens_per_sec': vtps, 'elapsed_s': now - t_start,
                           'grad_norm': None}
                    self.history.append(row)
                    self._csv_append(row)
                    print('[ val ] step %6d · val loss %.4f · %.0f tok/s'
                          % (s, vloss, vtps))

                if s in ckpt_at or now >= next_ckpt:
                    self.save_now()
                    if s in ckpt_at:
                        print('[trainer] forced checkpoint at step %d -> %s'
                              % (s, self._last_ckpt['path']))
                    else:
                        print('[trainer] %g-min checkpoint at step %d -> %s'
                              % (cfg['ckpt_every_minutes'], s,
                                 self._last_ckpt['path']))
                    next_ckpt = time.time() + interval
        except KeyboardInterrupt:
            interrupted = True
            self.save_now()
            print('[trainer] INTERRUPTED at step %d — checkpoint saved (%s). '
                  'Re-run the notebook; it resumes cleanly.'
                  % (self.step, self._last_ckpt['path']))

        if not interrupted:
            if cfg['final_save']:
                if self._dirty or self._last_ckpt is None:
                    self.save_now()
                    print('[trainer] final checkpoint at step %d -> %s'
                          % (self.step, self._last_ckpt['path']))
            else:
                self._dirty = False    # explicit "do not persist" (tests)
        return self.history


# --------------------------------------------------------------------------
# plot_history — matplotlib is a pencil, not intelligence (charts only)
# --------------------------------------------------------------------------

def plot_history(history, logy=False, title='p_1 · training loss',
                 figsize=(9, 4.5)):
    """Loss curves (train line + val points) from Trainer history rows or
    CSV-derived dicts. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    tr = [(r['step'], r['loss']) for r in history
          if r.get('kind', 'train') == 'train' and r.get('loss') is not None]
    va = [(r['step'], r['loss']) for r in history
          if r.get('kind') == 'val' and r.get('loss') is not None]
    fig, ax = plt.subplots(figsize=figsize)
    if tr:
        xs, ys = zip(*tr)
        ax.plot(xs, ys, lw=1.2, color='tab:blue', label='train')
    if va:
        xs, ys = zip(*va)
        ax.plot(xs, ys, 'o-', ms=4, lw=1.2, color='tab:red', label='val')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel('cross-entropy loss')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# __main__ — minimal selftest CLI (see module docstring for the contract)
# --------------------------------------------------------------------------

TINY_CONFIG = {'n_layer': 2, 'n_head': 2, 'd_model': 64, 'd_ff': 128,
               'ctx': 128, 'vocab_size': 4096}


def _selftest(args):
    """REAL corpus -> REAL BPE -> tiny Trainer, N steps, deterministic."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules
    from tokenizer_py import P1BPE
    import importlib
    M03 = importlib.import_module('03_model')

    cfg_model = dict(TINY_CONFIG if args.tiny else M03.CONFIG_B)
    root = p1_root()
    merges = root / 'corpus' / 'merges.json'
    health_qa = root / 'corpus' / 'health_qa.jsonl'
    if not health_qa.exists():
        raise SystemExit('selftest: %s not found (set P1_DATA_DIR?)' % health_qa)
    tokver = tokenizer_version_from_file(merges)
    tok = P1BPE(str(merges))
    data = build_data(health_qa, tok, cfg_model['ctx'], seed=1234,
                      max_val_windows=8)
    st = data.stats()
    print('[selftest] %d train / %d val entries -> %d / %d tokens '
          '(ctx %d, model %s, seed 42/1234)'
          % (st['n_train_entries'], st['n_val_entries'], st['train_tokens'],
             st['val_tokens'], st['ctx'], 'tiny' if args.tiny else 'CONFIG_B'))

    model = M03.Transformer(cfg_model, seed=42)
    tcfg = {'lr': 1e-3, 'warmup': 20, 'batch_size': 8, 'clip': 1.0,
            'eval_every': 0,
            'log_every': int(args.log_every) or max(1, int(args.steps) // 10),
            'seed': 42, 'ckpt_every_minutes': 15.0, 'final_save': True}
    trainer = Trainer(model, tokver, args.ckpt_dir, data, cfg=tcfg)
    if args.resume and trainer.resume_info is None:
        raise SystemExit('selftest --resume: no checkpoint found in %s'
                         % args.ckpt_dir)
    trainer.train(max_steps=int(args.steps))
    vloss, _tps = trainer.eval_val()
    rows = [r for r in trainer.history if r['kind'] == 'train']
    if not rows:
        raise SystemExit(
            'selftest: 0 steps ran — checkpoint in %s is already at/after '
            'step %d (auto-resume is always on; use a fresh --ckpt-dir or a '
            'larger --steps)' % (args.ckpt_dir, trainer.step))
    assert all(math.isfinite(r['loss']) for r in rows), 'non-finite losses'
    print('[selftest] final val loss %.4f · ckpt %s'
          % (vloss, find_resume(args.ckpt_dir)))
    print('SELFTEST OK — %d steps · losses finite · checkpoint + latest.txt '
          'written atomically' % len(rows))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='p_1 train_lib selftest — quick REAL-corpus training smoke')
    sub = ap.add_subparsers(dest='cmd', required=True)
    st = sub.add_parser('selftest', help='run a short real-data training smoke')
    st.add_argument('--steps', type=int, default=10, help='train steps (default 10)')
    st.add_argument('--tiny', action='store_true',
                    help='tiny model config (default without flag: CONFIG_B)')
    st.add_argument('--ckpt-dir', required=True, help='checkpoint directory')
    st.add_argument('--resume', action='store_true',
                    help='REQUIRE that an existing checkpoint is resumed')
    st.add_argument('--log-every', type=int, default=0,
                    help='log every N steps (default: max(1, steps//10)); '
                         '1 = every step, for trajectory-level tests')
    _args = ap.parse_args()
    _selftest(_args)
