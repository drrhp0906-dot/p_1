#!/usr/bin/env python3
"""Generate colab/06_parity.ipynb (Task 7 — weights bridge parity proof)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s))


def code(s):
    C.append(nbf.v4.new_code_cell(s))


md("""# 06 · Parity — the Colab brain speaks in the browser (Task 7)

Loads a **Task 5 checkpoint**, runs the **NumPy twin** (`colab/03_model.py`) on the
**10 fixed prompts** (eval set `ea-01..ea-10`), saves the expected outputs, and
exports the weights into **`docs/weights/web_model.bin`** — the little-endian
binary the browser loads.

Then open **`docs/parity_test.html`** (served over http, e.g.
`python3 -m http.server` in the repo root, or via GitHub Pages): it parses the
SAME binary with `docs/weights.js`, rebuilds the model on `docs/tensor.js` and
replays the SAME ids through `docs/model.js`. Every prompt must match the
fixture (bar: max |Δlogit| < 1e-2; measured ~1e-6) — green banner = the
browser is running your Colab-trained brain.

**Pipeline of this notebook:** checkpoint → fixture (`results/parity_outputs.json`)
+ weights (`docs/weights/web_model.bin` + `manifest.json`) → browser test page.

**The 10 fixed prompts (the contract — do not renumber):** `ea-01..ea-10` from
`corpus/eval_answerable.jsonl`, BPE-encoded with `bos`, LEFT-truncated to the
last `ctx` tokens, LEFT-padded with pad id 0 to exactly `ctx` (so the final
position is always the last real token). Built by `colab/05_export.py`
(`build_prompts`) and mirrored id-for-id by `scripts/test_task7_bridge.js`.""")

md("""## Cell 1 — repo root + checkpoint

Default: newest checkpoint under the Task 5 checkpoint dir
(`MyDrive/p_1/checkpoints` in Colab, resolved by `train_lib.find_resume`:
`latest.txt` first, newest `ckpt_step*.npz` otherwise). Override `CKPT` below
to pin a specific file.""")

code('''import sys, json, importlib.util
from pathlib import Path

def find_repo_root():
    """Clone dir /content/p_1 (Colab), CWD, or a Drive checkout."""
    for cand in [Path('/content/p_1'), Path.cwd()]:
        if (cand / 'colab' / '05_export.py').exists():
            return cand.resolve()
    import glob
    hits = sorted(glob.glob('/content/drive/MyDrive/*/colab/05_export.py'))
    if hits:
        return Path(hits[0]).resolve().parent.parent
    raise FileNotFoundError('p_1 repo not found — run colab/00_git_setup.ipynb '
                            'first (clones the repo to /content/p_1)')

REPO = find_repo_root()
sys.path.insert(0, str(REPO / 'colab'))
import train_lib                      # Task 5 machinery (find_resume, AdamW)

# load colab/05_export.py as a module (filename starts with a digit)
spec = importlib.util.spec_from_file_location('export05', REPO / 'colab' / '05_export.py')
export05 = importlib.util.module_from_spec(spec)
sys.modules['export05'] = export05
spec.loader.exec_module(export05)
print('repo root :', REPO)

CKPT = None   # <- pin e.g. Path('/content/drive/MyDrive/p_1/checkpoints/ckpt_step150.npz')
if CKPT is None:
    CKPT = train_lib.find_resume(train_lib.default_ckpt_dir())
    if CKPT is None:
        raise FileNotFoundError('no checkpoint in %s — train first (04_train.ipynb) '
                                'or set CKPT above' % train_lib.default_ckpt_dir())
print('checkpoint:', CKPT)''')

md("""## Cell 2 — build the model from the checkpoint's own config + load the weights

`load_model_from_ckpt` builds a `Transformer` from the config JSON stored
INSIDE the npz (the config there is authoritative) and fills the params via
`train_lib.load_checkpoint` — the exact Task 5 resume path, so what the
browser will receive is bit-identical to what training produced.""")

code('''model, meta = export05.load_model_from_ckpt(CKPT)
print('config   :', json.dumps(meta['config'], sort_keys=True))
print('step     :', meta['step'])
print('tokenizer:', meta['tokenizer_version'])
print('numParams:', model.numParams())
assert model.numParams() > 0''')

md("""## Cell 3 — the 10 fixed prompts (ea-01 .. ea-10)

BPE-encoded with `bos`, left-truncated to the last `ctx` tokens, left-padded
with pad id 0 to exactly `ctx`. The tokenizer version of the live
`corpus/merges.json` is checked against the checkpoint's `tokenizer_version`
(same loud-mismatch policy as resume).""")

code('''from tokenizer_py import P1BPE

tok = P1BPE(REPO / 'corpus' / 'merges.json')
print('tokenizer version check:',
      train_lib.tokenizer_version_from_file(REPO / 'corpus' / 'merges.json'),
      '==', meta['tokenizer_version'])

entries = export05.load_eval_entries(REPO)
prompts = export05.build_prompts(tok, entries, int(model.config['ctx']))
for p in prompts:
    real = next((i for i, v in enumerate(p['ids']) if v != 0), len(p['ids']))
    print('%s | %s | %d real ids + %d pads'
          % (p['id'], p['text'], len(p['ids']) - real, real))''')

md("""## Cell 4 — NumPy forward on each prompt → `results/parity_outputs.json`

Per prompt: the first 64 logits at the final position (rounded 6 dp), the
top-5 softmax `[[id, prob] × 5]`, the 20 greedy next ids and their decode.
Fixture format: `p1-parity-v1`.""")

code('''import math

out = export05.build_parity_json(model, meta, prompts, tok)

RESULTS = train_lib.p1_root() / 'results'      # MyDrive/p_1/results in Colab
REPO_RESULTS = REPO / 'results'                # repo-local mirror (Pages-able)
for d in (RESULTS, REPO_RESULTS):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'parity_outputs.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
        fh.write('\\n')
print('parity_outputs.json ->', RESULTS)
print('                    ->', REPO_RESULTS)
p0 = out['prompts'][0]
print('spot check:', p0['id'], '| first 8 logits:',
      p0['logits_first'][:8], '| argmax', p0['argmax_top5'][0],
      '| greedy[:8]', p0['greedy_next20'][:8])''')

md("""## Cell 5 — export the browser weights (`05_export.py` via subprocess)

Writes **`docs/weights/web_model.bin`** (magic `P1WM` · version 1 · config
JSON · every tensor in the canonical `paramNames()` order — weight-tied head,
`wte` stored ONCE), **`manifest.json`** (names/shapes/offsets for humans) and
the `file://` fallback snapshots. Runs the FULL CLI (export + fixtures) so the
committed artifacts are exactly what a fresh run of the script produces.""")

code('''import subprocess

cmd = [sys.executable, str(REPO / 'colab' / '05_export.py'),
       '--ckpt', str(CKPT),
       '--out', str(REPO / 'docs' / 'weights'),
       '--parity-fixtures', str(REPO_RESULTS / 'parity_outputs.json'),
       '--snapshot', str(REPO / 'docs' / 'parity_outputs_data.js')]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr)
    raise RuntimeError('05_export.py failed with exit %d' % r.returncode)

b = (REPO / 'docs' / 'weights' / 'web_model.bin').stat().st_size
print('web_model.bin  : %d bytes (= 4 x %d params + header, verified above)'
      % (b, model.numParams()))
print('manifest.json  :', REPO / 'docs' / 'weights' / 'manifest.json')
print('DONE — commit docs/weights/ + docs/parity_outputs_data.js with the page')''')

md("""---
## ✅ DONE WHEN

**Open `docs/parity_test.html` in your browser → the banner is GREEN.**

- Serve the repo root over http for the clean path:
  `python3 -m http.server` → http://localhost:8000/docs/parity_test.html
  (or push to GitHub — Pages serves `docs/` directly).
- `file://` works too: the page falls back to the committed snapshots
  `docs/weights/web_model_data.js` + `docs/parity_outputs_data.js`
  (browsers block `fetch()` on `file://` — same pattern as
  `tokenizer_data.js` / `model_fixtures_data.js`).

You should see 10/10 rows PASS (max |Δlogit| ~1e-6, identical argmax +
greedy-20 ids) and the Colab-trained brain's greedy continuations decoded
side by side (Python fixture vs the browser twin). The text is still babble
at 150 tiny-config steps — the *numbers* are what this page certifies.""")

nb.cells = C
nbf.validate(nb)
out = REPO_OUT = 'colab/06_parity.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, out)
print('wrote', out, '·', len(C), 'cells · nbformat valid')
