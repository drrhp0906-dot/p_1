"""
p_1 · colab/05_export.py — Task 7 weights bridge: checkpoint -> browser
--------------------------------------------------------------------------------
Turns a Task 5 checkpoint (contract npz) into the little-endian binary the
browser loads, plus a human/debug manifest and the 10-prompt parity fixture.

PURE Python + NumPy ONLY. Reuses the Task 5 machinery instead of
reimplementing it: train_lib.find_resume (latest.txt first, newest npz
fallback) and train_lib.load_checkpoint (fills model params from
'param:<name>' keys), and the Task 3 tokenizer for the parity prompts.

BINARY FORMAT "p1wm" (docs/weights/web_model.bin) — EXACT layout, all
integers uint32 LE, all floats float32 LE, tensors in the CANONICAL
paramNames() order (weight-tied head: wte stored ONCE — it IS the head):

    magic        4 ASCII bytes  "P1WM"
    version      uint32 LE      = 1
    configLen    uint32 LE      byte length of the config JSON
    config JSON  UTF-8          full model config dict (n_layer / n_head /
                                d_model / d_ff / ctx / vocab_size) PLUS
                                tokenizer_version + step trained
    per tensor:  nameLen uint32 LE -> name UTF-8
                 ndim uint32 LE -> ndim x uint32 LE dims (row-major)
                 raw float32 LE data (prod(shape) values)

MANIFEST (docs/weights/manifest.json) — for humans/debug, NOT needed to parse:
    {"format":"p1wm","version":1,"config":{...},"tokenizer_version":"...",
     "step":N,"num_params":N,
     "tensors":[{"name","shape":[...],"offset","nbytes"},...],
     "total_bytes":N}            offsets = absolute byte positions of each
                                 tensor's DATA block in web_model.bin

PARITY FIXTURE (results/parity_outputs.json, format "p1-parity-v1"):
the 10 FIXED prompts (eval_answerable.jsonl ea-01..ea-10), BPE-encoded with
bos, LEFT-truncated to the last ctx tokens, LEFT-padded with pad id 0 to
exactly ctx (final position = last real token). Per prompt: the first 64
logits at the final position (rounded 6 dp), the top-5 softmax
[[id,prob]x5] at the final position, and 20 greedy tokens. The browser twin
(docs/parity_test.html + docs/weights.js) replays the SAME ids through
model.js and must match (bar: max |logit diff| < 1e-2; measured ~1e-6).

CLI
  python3 colab/05_export.py --ckpt <path-or-ckpt-dir> --out <docs/weights dir>
                             [--repo-root auto] [--no-snapshot]
  python3 colab/05_export.py --ckpt <path> --parity-fixtures <out.json>
                             [--snapshot <docs/parity_outputs_data.js>]
  (--ckpt accepts a single ckpt_stepN.npz OR a checkpoint directory — a
   directory is resolved with find_resume. Both --out and --parity-fixtures
   may be given in one call. --repo-root auto = the dir above colab/.)

Importable as a module (06_parity.ipynb uses importlib on this file) —
public helpers: resolve_ckpt, read_ckpt_meta, load_model_from_ckpt,
build_prompts, final_position_logits, greedy_continue, build_parity_json.
"""

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:               # sibling imports (train_lib, ...)
    sys.path.insert(0, str(_HERE))

import train_lib                              # noqa: E402  (Task 5: reuse)
from train_lib import find_resume, load_checkpoint   # noqa: E402
from tokenizer_py import P1BPE                       # noqa: E402  (Task 3)
import importlib.util as _ilu                        # noqa: E402

_spec = _ilu.spec_from_file_location('p1_model_03', _HERE / '03_model.py')
_p1_model_03 = _ilu.module_from_spec(_spec)
sys.modules['p1_model_03'] = _p1_model_03
_spec.loader.exec_module(_p1_model_03)

MAGIC = b'P1WM'
FORMAT_VERSION = 1
WEB_BIN_NAME = 'web_model.bin'
MANIFEST_NAME = 'manifest.json'
SNAPSHOT_NAME = 'web_model_data.js'          # file:// fallback snapshot

PARITY_FORMAT = 'p1-parity-v1'
PARITY_IDS = ['ea-%02d' % i for i in range(1, 11)]   # THE 10 fixed prompts
LOGITS_FIRST_N = 64                                  # first 64 logits @ final pos
GREEDY_N = 20                                        # greedy continuation len
PAD_ID = 0                                           # left-pad token (raw byte 0)


# --------------------------------------------------------------------------
# checkpoint loading (thin wrappers over the Task 5 contract helpers)
# --------------------------------------------------------------------------

def resolve_ckpt(path):
    """A checkpoint file -> itself; a checkpoint DIRECTORY -> newest npz via
    train_lib.find_resume (latest.txt first, max-step glob fallback)."""
    p = Path(path)
    if p.is_dir():
        ck = find_resume(p)
        if ck is None:
            raise FileNotFoundError('no ckpt_step*.npz in %r (and no valid '
                                    'latest.txt)' % str(p))
        return ck
    if not p.exists():
        raise FileNotFoundError('checkpoint not found: %s' % str(p))
    return p


def read_ckpt_meta(ckpt):
    """{'step','config','tokenizer_version','path'} from a contract npz
    (read-only peek — does not need a model instance)."""
    ckpt = Path(ckpt)
    with np.load(str(ckpt), allow_pickle=False) as z:
        step = int(z['step'])
        config = json.loads(train_lib._as_str(z['config']))
        tokver = train_lib._as_str(z['tokenizer_version'])
    return {'step': step, 'config': config, 'tokenizer_version': tokver,
            'path': ckpt}


def load_model_from_ckpt(ckpt, optimizer=None):
    """Build a Transformer from the checkpoint's OWN config and fill its
    params via train_lib.load_checkpoint (REUSED, not reimplemented).
    Returns (model, meta) with meta = read_ckpt_meta + load_checkpoint info."""
    ckpt = resolve_ckpt(ckpt)
    peek = read_ckpt_meta(ckpt)
    model = _p1_model_03.Transformer(peek['config'], seed=0)
    if optimizer is None:
        optimizer = train_lib.AdamW()
    info = load_checkpoint(str(ckpt), model, optimizer)   # fills model params
    meta = dict(peek)
    meta.update(info)
    return model, meta


# --------------------------------------------------------------------------
# web_model.bin + manifest.json
# --------------------------------------------------------------------------

def _bin_config(meta):
    """Config JSON object stored in the bin: full model config dict PLUS
    tokenizer_version + step trained."""
    cfg = dict(meta['config'])
    cfg['tokenizer_version'] = meta['tokenizer_version']
    cfg['step'] = int(meta['step'])
    return cfg


def export_web_model(ckpt, out_dir, snapshot=True, quiet=False):
    """checkpoint -> <out_dir>/web_model.bin + manifest.json (+ file://
    snapshot). Returns dict with paths and sizes. Tensors are walked in the
    canonical paramNames() order of the freshly loaded model — which is the
    same order train_lib.save_checkpoint wrote them in."""
    ckpt = resolve_ckpt(ckpt)
    model, meta = load_model_from_ckpt(ckpt)
    cfg_obj = _bin_config(meta)
    cfg_bytes = json.dumps(cfg_obj, sort_keys=True,
                           separators=(',', ':')).encode('utf-8')

    names = model.paramNames()
    params = model.params()

    buf = bytearray()
    buf += MAGIC
    buf += struct.pack('<I', FORMAT_VERSION)
    buf += struct.pack('<I', len(cfg_bytes))
    buf += cfg_bytes
    header_bytes = len(buf)

    tensors = []
    num_params = 0
    record_bytes = 0
    for name, p in zip(names, params):           # canonical order, wte ONCE
        shape = [int(s) for s in p.shape]
        n = 1
        for s in shape:
            n *= int(s)
        name_b = name.encode('utf-8')
        data = np.asarray(p.data, dtype=np.float32).reshape(-1).astype(
            '<f4', copy=False)                   # explicit little-endian
        offset = len(buf) + 4 + len(name_b) + 4 + 4 * len(shape)
        buf += struct.pack('<I', len(name_b))
        buf += name_b
        buf += struct.pack('<I', len(shape))
        for s in shape:
            buf += struct.pack('<I', s)
        buf += data.tobytes()
        tensors.append({'name': name, 'shape': shape,
                        'offset': offset, 'nbytes': int(n) * 4})
        num_params += n
        record_bytes += 4 + len(name_b) + 4 + 4 * len(shape)
    total_bytes = len(buf)

    # ---- hard sanity: the file is EXACTLY header + records + 4*n payload
    expected = header_bytes + record_bytes + 4 * num_params
    if total_bytes != expected:
        raise AssertionError('web_model.bin size %d != expected %d '
                             '(header %d + records %d + payload %d)'
                             % (total_bytes, expected, header_bytes,
                                record_bytes, 4 * num_params))

    manifest = {'format': 'p1wm', 'version': FORMAT_VERSION,
                'config': cfg_obj,
                'tokenizer_version': meta['tokenizer_version'],
                'step': int(meta['step']),
                'num_params': int(num_params),
                'tensors': tensors,
                'total_bytes': int(total_bytes)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / WEB_BIN_NAME
    man_path = out_dir / MANIFEST_NAME
    with open(bin_path, 'wb') as fh:             # plain write; file is small
        fh.write(bytes(buf))
    with open(man_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)
        fh.write('\n')

    snap_path = None
    if snapshot:
        snap_path = out_dir / SNAPSHOT_NAME
        b64 = base64.b64encode(bytes(buf)).decode('ascii')
        with open(snap_path, 'w', encoding='utf-8') as fh:
            fh.write('/* AUTO-GENERATED by colab/05_export.py — do not edit.\n'
                     '   base64 file:// fallback snapshot of web_model.bin\n'
                     '   (browsers block fetch() on file://; see '
                     'tokenizer_data.js / model_fixtures_data.js). */\n'
                     'window.P1_WEB_MODEL_B64 = "%s";\n' % b64)

    if not quiet:
        print('checkpoint : %s (step %d, tokenizer %s)'
              % (ckpt.name, meta['step'], meta['tokenizer_version']))
        print('config     : %s' % json.dumps(cfg_obj, sort_keys=True))
        print('web_model.bin : %s' % bin_path)
        print('  size       : %d bytes' % total_bytes)
        print('  header     : %d bytes (magic 4 + version 4 + cfgLen 4 + '
              'config json %d)' % (header_bytes, len(cfg_bytes)))
        print('  records    : %d bytes (%d tensors: nameLen+name+ndim+dims)'
              % (record_bytes, len(tensors)))
        print('  payload    : %d bytes = 4 x %d params'
              % (4 * num_params, num_params))
        print('  check      : %d == %d + %d + %d  %s'
              % (total_bytes, header_bytes, record_bytes, 4 * num_params,
                 'OK' if total_bytes == expected else 'MISMATCH'))
        print('manifest.json : %s (%d tensors, offsets monotonic, '
              'total_bytes %d)' % (man_path, len(tensors), total_bytes))
        if snap_path:
            print('snapshot      : %s (%d bytes, file:// fallback)'
                  % (snap_path, snap_path.stat().st_size))
    return {'bin': bin_path, 'manifest': man_path, 'snapshot': snap_path,
            'total_bytes': total_bytes, 'num_params': int(num_params),
            'config': cfg_obj}


# --------------------------------------------------------------------------
# parity fixtures — the 10 fixed prompts (shared contract with the Node twin)
# --------------------------------------------------------------------------

def load_eval_entries(repo_root, ids=PARITY_IDS):
    """eval_answerable.jsonl entries for the fixed prompt ids, in order."""
    path = Path(repo_root) / 'corpus' / 'eval_answerable.jsonl'
    by_id = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            by_id[e['id']] = e
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError('eval_answerable.jsonl missing %r' % missing)
    return [by_id[i] for i in ids]


def build_prompts(tok, entries, ctx, ids=PARITY_IDS):
    """THE prompt contract (mirrored by scripts/test_task7_bridge.js):
    ids = [bos] + tok.encode(question); keep the LAST ctx tokens
    (left-truncate); left-pad with PAD_ID to EXACTLY ctx so the final
    position is always the last real token. Deterministic."""
    out = []
    for e in entries:
        pid = e['id']
        if ids is not None and pid not in ids:
            continue
        seq = [int(tok.bos)] + [int(i) for i in tok.encode(e['question'])]
        if len(seq) > ctx:
            seq = seq[-ctx:]                         # left-truncate: keep tail
        if len(seq) < ctx:
            seq = [PAD_ID] * (ctx - len(seq)) + seq  # left-pad: tail at the end
        out.append({'id': pid, 'text': e['question'], 'ids': seq})
    if ids is not None and len(out) != len(ids):
        raise AssertionError('expected %d prompts, built %d'
                             % (len(ids), len(out)))
    return out


def final_position_logits(model, ids):
    """Forward [1, T] -> float32 logits row [vocab] at the FINAL position.
    Row-major [B*T, V] layout, so the final position is simply the last row."""
    logits = model.forward([int(i) for i in ids], 1)
    V = model.config['vocab_size']
    return np.asarray(logits.data, dtype=np.float32).reshape(-1, V)[-1]


def argmax_first(row):
    """np.argmax = FIRST occurrence of the max. The JS twin scans ascending
    ids with strict '>' and lands on the same index."""
    return int(np.argmax(np.asarray(row)))


def softmax_f64(row):
    """float64 softmax (max-subtracted) — fixture/display only."""
    r = np.asarray(row, dtype=np.float64)
    e = np.exp(r - r.max())
    return e / e.sum()


def top_k(probs, k=5):
    """Top-k as [[prob, id], ...] sorted by (desc prob, ASC id). Ascending-id
    scan replacing ONLY on strictly-greater prob = deterministic tie-break
    that the JS twin reproduces exactly."""
    probs = list(probs)
    best = []                                        # [prob, id] pairs
    for i, p in enumerate(probs):
        if len(best) < k:
            best.append([p, i])
            best.sort(key=lambda t: (-t[0], t[1]))
        elif p > best[-1][0]:
            best[-1] = [p, i]
            best.sort(key=lambda t: (-t[0], t[1]))
    return best


def greedy_continue(model, ids, n=GREEDY_N):
    """Greedy continuation: re-forward the growing sequence each step,
    windowed to the LAST ctx tokens (left-truncate), append the argmax.
    Mirrored exactly by the JS twin."""
    ctx = int(model.config['ctx'])
    seq = [int(i) for i in ids]
    out = []
    for _ in range(n):
        window = seq[-ctx:]
        row = final_position_logits(model, window)
        nxt = argmax_first(row)
        seq.append(nxt)
        out.append(nxt)
    return out


def build_parity_json(model, meta, prompts, tok):
    """Run the NumPy model on all prompts and assemble the parity fixture
    dict (format p1-parity-v1)."""
    V = int(model.config['vocab_size'])
    n_first = min(LOGITS_FIRST_N, V)
    out_prompts = []
    for p in prompts:
        row = final_position_logits(model, p['ids'])
        probs = softmax_f64(row)
        top5 = top_k(probs, 5)
        greedy = greedy_continue(model, p['ids'])
        out_prompts.append({
            'id': p['id'],
            'text': p['text'],
            'ids': [int(i) for i in p['ids']],
            'logits_first': [round(float(x), 6) for x in row[:n_first]],
            'argmax_top5': [[int(i), round(float(q), 6)] for q, i in top5],
            'greedy_next20': [int(i) for i in greedy],
            # convenience for the side-by-side display (greedy ids only —
            # never pads, so decode is clean):
            'greedy_text': tok.decode(greedy),
        })
    return {'format': PARITY_FORMAT,
            'config': _bin_config(meta),
            'meta': {'ckpt': Path(meta['path']).name if meta.get('path') else None,
                     'bos': int(tok.bos), 'pad_id': PAD_ID, 'pad': 'left',
                     'truncate': 'left-keep-tail',
                     'logits_first_n': int(n_first),
                     'greedy_n': GREEDY_N,
                     'generated_by': 'colab/05_export.py'},
            'prompts': out_prompts}


def make_parity_fixtures(ckpt, out_json, repo_root=None, snapshot=None,
                         quiet=False):
    """Checkpoint -> results/parity_outputs.json (+ optional JS snapshot).
    Verifies the live merges file against the checkpoint's tokenizer_version
    (same LOUD-mismatch policy as train_lib resume)."""
    repo_root = Path(repo_root) if repo_root else _HERE.parent
    ckpt = resolve_ckpt(ckpt)
    model, meta = load_model_from_ckpt(ckpt)

    merges_path = Path(repo_root) / 'corpus' / 'merges.json'
    live_ver = train_lib.tokenizer_version_from_file(merges_path)
    if live_ver != meta['tokenizer_version']:
        print('WARNING: tokenizer version mismatch: checkpoint %r vs live '
              'merges %r — prompts will be encoded with the LIVE merges; '
              'treat the fixture as incompatible with this checkpoint.'
              % (meta['tokenizer_version'], live_ver))
    tok = P1BPE(merges_path)
    entries = load_eval_entries(repo_root)
    prompts = build_prompts(tok, entries, int(model.config['ctx']))
    obj = build_parity_json(model, meta, prompts, tok)

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1)
        fh.write('\n')
    if not quiet:
        print('parity fixture : %s (%d prompts, %d params model, step %d)'
              % (out_json, len(obj['prompts']), model.numParams(), meta['step']))
        for p in obj['prompts']:
            print('  %s | %s\n      greedy: %r' % (p['id'], p['text'],
                                                    p['greedy_text']))
    snap_written = None
    if snapshot:
        snapshot = Path(snapshot)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot, 'w', encoding='utf-8') as fh:
            fh.write('/* AUTO-GENERATED by colab/05_export.py — do not edit.\n'
                     '   file:// fallback snapshot of results/parity_outputs.json\n'
                     '   (browsers block fetch() on file://; see '
                     'model_fixtures_data.js). */\n'
                     'window.P1_PARITY_OUTPUTS = ')
            json.dump(obj, fh, separators=(',', ':'))
            fh.write(';\n')
        snap_written = snapshot
        if not quiet:
            print('snapshot       : %s (%d bytes, file:// fallback)'
                  % (snapshot, snapshot.stat().st_size))
    return obj


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _repo_root(arg):
    """--repo-root auto -> the dir above colab/ (holds corpus/, docs/)."""
    if arg and arg != 'auto':
        return Path(arg)
    return _HERE.parent


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='p_1 Task 7: checkpoint -> browser weights '
                    '(web_model.bin + manifest.json) and/or the 10-prompt '
                    'parity fixture.')
    ap.add_argument('--ckpt', required=True,
                    help='ckpt_stepN.npz OR checkpoint dir (find_resume)')
    ap.add_argument('--out', default=None,
                    help='output dir for web_model.bin + manifest.json '
                         '(default <repo-root>/docs/weights)')
    ap.add_argument('--repo-root', default='auto',
                    help='repo root holding corpus/ + docs/ (default: auto)')
    ap.add_argument('--parity-fixtures', default=None, metavar='OUT_JSON',
                    help='also run the NumPy model on the 10 fixed prompts '
                         'and write the p1-parity-v1 fixture here')
    ap.add_argument('--snapshot', default=None, metavar='OUT_JS',
                    help='with --parity-fixtures: also write the file:// JS '
                         'snapshot (window.P1_PARITY_OUTPUTS)')
    ap.add_argument('--no-snapshot', action='store_true',
                    help='skip the base64 file:// snapshot of web_model.bin')
    args = ap.parse_args(argv)

    repo = _repo_root(args.repo_root)
    did = False
    if args.out or not args.parity_fixtures:
        out_dir = Path(args.out) if args.out else repo / 'docs' / 'weights'
        export_web_model(args.ckpt, out_dir,
                         snapshot=not args.no_snapshot)
        did = True
    if args.parity_fixtures:
        make_parity_fixtures(args.ckpt, args.parity_fixtures, repo_root=repo,
                             snapshot=args.snapshot)
        did = True
    if not did:
        ap.error('nothing to do: pass --out and/or --parity-fixtures')
    return 0


if __name__ == '__main__':
    sys.exit(main())
