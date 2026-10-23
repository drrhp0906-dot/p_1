"""Generate colab/08_eval.ipynb — Task 9, verification + eval harness notebook.

Uses eval_lib's ACTUAL API (colab/eval_lib.py): splitClaims/verifyClaim live in
the lib; the notebook wires the 4 answer modes (RAW / GROUNDED / VERIFIED /
ABSTAINING via answer_all_modes + evaluate_sets), the tau calibration
(calibrate_tau), the real Config B model answerer (train_lib.find_resume +
03_model + tokenizer_py, greedy temp 0.7, max 80 new tokens, prompt
bos + "Q: <q>\\nA:"), metrics.json writing (write_metrics) and the two bar
charts (make_charts). If no checkpoint exists -> LOUD warning + eval_lib's
deterministic stub answerer (same corruptor the acceptance test uses).

All code cells are dry-run locally in order by gen-time self-test:
  branch 1: real checkpoint (scripts/t5_runA_uninterrupted, tiny smoke config)
  branch 2: no checkpoint -> stub fallback
"""
import json

REPO = "/home/z/my-project/download/p_1"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


CELL_ENV = r'''import os, sys, json, pathlib
IN_COLAB = 'google.colab' in str(sys.modules or '')
try:
    from google.colab import drive  # noqa
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    pass

if IN_COLAB:
    ROOT = pathlib.Path('/content/drive/MyDrive/p_1')   # Drive: corpus/ checkpoints/ results/
    REPO = pathlib.Path('/content/p_1')                  # code: run 00_git_setup.ipynb first
else:
    ROOT = pathlib.Path('.').resolve()
    REPO = ROOT

sys.path.insert(0, str(REPO / 'colab'))
for sub in ('corpus', 'checkpoints', 'results'):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

# eval_lib reuses Task 8's retrieval (imported, never reimplemented); point
# retrieval_py.data_dir() at OUR corpus dir via its documented env override.
CORPUS_DIR = ROOT / 'corpus'
if (CORPUS_DIR / 'health_qa.jsonl').exists():
    os.environ['P1_DATA_DIR'] = str(CORPUS_DIR)
import eval_lib as el
import retrieval_py as rp
print('eval_lib v%s | format %s | IN_COLAB=%s' % (el.VERSION, el.FORMAT, IN_COLAB))

corpus = rp.load_jsonl(rp.data_dir() / 'health_qa.jsonl')
sets = {name: rp.load_jsonl(rp.data_dir() / ('eval_%s.jsonl' % name))
        for name in ('answerable', 'mythtraps', 'unanswerable')}
print('corpus pairs: %d | eval sets: %s' % (
    len(corpus), {k: len(v) for k, v in sets.items()}))

index = rp.build_index(corpus)
print('index: N=%d chunks, vocab=%d, avgdl=%.3f, build %.0f ms' % (
    index['N'], index['vocab'], index['avgdl'], index['build_ms']))'''

CELL_TAU = r'''# ABSTAIN_TAU is calibrated ONCE = median top-1 BM25 over the 60 answerable
# eval questions; the lib hardcodes it, this cell re-derives + asserts it.
tau = el.calibrate_tau(index, sets['answerable'])
print('tau recomputed : %.10f' % tau)
print('ABSTAIN_TAU    : %.4f (hardcoded in eval_lib)' % el.ABSTAIN_TAU)
assert abs(tau - el.ABSTAIN_TAU) < 5e-4, 'ABSTAIN_TAU out of calibration!'
print('ABSTAINING mode refuses a question when its best BM25 score < tau')'''

CELL_ANSWERER = r'''# ---- the answerer: REAL Config B brain (newest checkpoint) or loud stub ----
import importlib
import numpy as np
import train_lib
from tokenizer_py import P1BPE
M03 = importlib.import_module('03_model')            # module name starts with a digit

CKPT_DIR  = ROOT / 'checkpoints'
MERGES    = ROOT / 'corpus' / 'merges.json'
MAX_NEW_TOKENS = 80      # brief: max 80 new tokens
TEMPERATURE    = 0.7     # brief: greedy temp 0.7 (argmax is temp-invariant;
                         #        the scaling is kept to honor the spec)

tok = P1BPE(str(MERGES))                             # our BPE (Task 3)
print('tokenizer   : vocab %d, bos %d, eos %d' % (tok.vocabSize, tok.bos, tok.eos))

ckpt = train_lib.find_resume(CKPT_DIR)
if ckpt is None:
    print('=' * 74)
    print('!!! LOUD WARNING: NO CHECKPOINT in %s' % CKPT_DIR)
    print('!!! No trained Config B brain found — falling back to eval_lib\'s')
    print('!!! DETERMINISTIC STUB answerer (make_stub_answerer, the same')
    print('!!! corrupting stub the acceptance test uses, seed 42 / 35%%).')
    print('!!! Metrics below are HARNESS numbers, NOT model numbers.')
    print('!!! Train the model with 04_train.ipynb first for real metrics.')
    print('=' * 74)
    answerer = el.make_stub_answerer(corpus, corrupt_rate=0.35, seed=42)
    answerer_name = 'stub(seed=42,corrupt=0.35)'
else:
    info = train_lib.resume_provenance(str(CKPT_DIR))
    if 'error' in info:
        raise RuntimeError('newest checkpoint unreadable: %r' % info)
    config = info['config']                       # the ckpt's OWN config (authoritative)
    ctx = int(config['ctx'])
    model = M03.Transformer(config, seed=0)       # seed irrelevant: params are loaded next
    dummy_opt = train_lib.AdamW(lr=0.0)           # moment slots for load_checkpoint
    loaded = train_lib.load_checkpoint(ckpt, model, dummy_opt)
    live_tokver = train_lib.tokenizer_version_from_file(str(MERGES))
    print('checkpoint : %s (step %d, saved %s)' % (
        info['name'], loaded['step'], info['saved_at']))
    print('config     : %s' % config)
    if loaded['tokenizer_version'] != live_tokver:
        print('!!! LOUD WARNING: checkpoint tokenizer_version %r != live merges %r'
              % (loaded['tokenizer_version'], live_tokver))
        print('!!! token ids may be misaligned — retrain or re-tokenize.')
    print('params     : %s' % f"{model.numParams():,}")

    def model_answerer(question: str) -> str:
        """REAL brain: greedy (temp 0.7 scaling, argmax), max 80 new tokens,
        prompt = bos + 'Q: <q>\\nA:' — the exact training-time Q/A shape."""
        prompt = 'Q: %s\nA:' % question
        ids = ([tok.bos] + tok.encode(prompt))[-ctx:]
        new_ids = []
        for _ in range(MAX_NEW_TOKENS):
            window = np.array(ids[-ctx:])
            logits = model.forward(window, 1)                 # [T, vocab]
            last = logits.data.reshape(logits.shape[-2:]).astype(np.float64)[-1]
            last = last / TEMPERATURE
            nxt = int(np.argmax(last))                        # greedy
            if nxt == tok.eos:
                break
            ids.append(nxt)
            new_ids.append(nxt)
        text = tok.decode(new_ids).split('\n')[0].strip()     # stop at a new line
        return text if text else '(no answer tokens)'

    answerer = model_answerer
    answerer_name = 'real-ckpt(%s)' % info['name']
    # sample answers — listen to the brain before measuring it
    for q in (sets['answerable'][0]['question'],
              sets['mythtraps'][0]['question'],
              sets['unanswerable'][0]['question']):
        print('--- Q:', q)
        print('    A:', answerer(q))'''

CELL_RUN = r'''# ---- run 3 eval sets x 4 modes (RAW / GROUNDED / VERIFIED / ABSTAINING) ----
# Real model on CPU: ~0.5-1 s per generated token-step -> a full 100-question
# run takes on the order of an hour. The stub fallback is instant.
RUN_CAP = None            # e.g. 5 for a quick smoke; None = all 100 questions
run_sets = sets if RUN_CAP is None else {k: v[:RUN_CAP] for k, v in sets.items()}
print('answerer: %s | questions: %s' % (
    answerer_name, {k: len(v) for k, v in run_sets.items()}))

metrics, records = el.evaluate_sets(index, answerer, run_sets,
                                    tau=el.ABSTAIN_TAU, log=print)
metrics['config']['answerer'] = answerer_name
metrics['stub'] = answerer_name.startswith('stub')

print('\n%-13s %-11s %6s %6s %11s %8s %12s %11s' % (
    'set', 'mode', 'claims', 'unsup', 'halluc.rate', 'refused',
    'refusal.rate', 'parrot.rate'))
for set_name in ('answerable', 'mythtraps', 'unanswerable'):
    for m in el.MODES:
        e = metrics['sets'][set_name]['per_mode'][m]
        print('%-13s %-11s %6d %6d %11.3f %8d %12.3f %11.3f' % (
            set_name, m, e['total_claims'], e['unsupported'],
            e['hallucination_rate'], e['refused'], e['refusal_rate'],
            e.get('parroting_rate', float('nan'))))
print('\nsummary:', metrics['summary']['headline'])'''

CELL_WRITE = r'''# ---- write results/metrics.json + the two bar charts ----
RESULTS = ROOT / 'results'
out_json = el.write_metrics(metrics, RESULTS / 'metrics.json')
print('wrote', out_json)

charts = el.make_charts(metrics, RESULTS)   # matplotlib = pencil, not intelligence
for p in charts:
    print('wrote', p)
try:                                        # inline display when in a notebook
    from IPython.display import Image, display
    for p in charts:
        display(Image(filename=str(p)))
except Exception:
    pass'''

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.x"}},
    "cells": [
        md("""# p_1 — 08 · verification + eval harness: the measurement system

The machine that catches hallucinations, run over the 3 eval sets (60 answerable /
20 myth-traps / 20 unanswerable) in **4 answer modes**:

| mode | what it answers with | what gets measured |
|------|----------------------|--------------------|
| **RAW** | the model's own text, verbatim | hallucination rate = unsupported claims / total claims |
| **+GROUNDING** | answer composed ONLY of the best retrieved corpus sentences (+ citations) | hallucination rate; refusals when nothing relevant retrieved |
| **+VERIFICATION** | RAW answer -> atomic claims -> each claim verified vs top-3 chunks (content-F1, exact number match, negation consistency); failed claims rewritten from the corpus or flagged `[unsupported]` | hallucination rate on the final text + rewrite/flag counts |
| **+ABSTENTION** | refuses ("I'm not confident…") when the best BM25 score < tau, else the GROUNDED answer | abstention correctness = refusal rate on unanswerables (tau = median top-1 BM25 of the answerable set, calibrated once) |

On the **myth-trap** set every answer also gets the folk-vs-correction comparison
(negation-stripped content-F1 + explicit correction-cue override) -> **myth-parroting rate**.
Every mode's final answer is re-verified with the SAME verifier — one yardstick.
Claims that are merely *not in the corpus* are counted separately, never as hallucinations.

Output: `MyDrive/p_1/results/metrics.json` + two bar charts. The engine itself
(extractor / verifier / modes / metrics) lives in `colab/eval_lib.py`; this
notebook only wires it to data, the model and Drive."""),

        md("## 0 · Environment — Drive, paths, imports, retrieval index"),
        code(CELL_ENV),

        md("## 1 · Tau calibration (ABSTAIN_TAU, derived once)"),
        code(CELL_TAU),

        md("""## 2 · The answerer — your trained brain, or a loud stub

Loads the **newest Config B checkpoint** from `MyDrive/p_1/checkpoints/`
(`train_lib.find_resume`), rebuilds the model from the checkpoint's own config,
greedy decoding (temperature 0.7 scaling, max **80 new tokens**, prompt
`bos + "Q: <q>\\nA:"`). No checkpoint -> LOUD warning + the deterministic test
stub (clearly flagged in the metrics)."""),
        code(CELL_ANSWERER),

        md("## 3 · Run the harness — 3 sets × 4 modes"),
        code(CELL_RUN),

        md("## 4 · Write metrics.json + charts"),
        code(CELL_WRITE),

        md("""### DONE WHEN
- [ ] `results/metrics.json` **compares all 4 modes on all 3 eval sets**
      (`sets.answerable|mythtraps|unanswerable` × `per_mode.raw|grounded|verified|abstaining`).
- [ ] The table prints the first hallucination-rate comparison: grounding cuts the
      RAW hallucination rate, VERIFIED rewrites/flags most planted errors, and
      ABSTAINING refuses ≥ 90% of the unanswerable questions.
- [ ] The two bar charts exist in `results/` (`hallucination_by_mode.png`,
      `myth_parroting_by_mode.png`).
- [ ] If the real brain answered, `config.answerer` names the checkpoint; if the
      stub answered, the notebook said LOUDLY so and `stub: true` is in the JSON.

*(Parity of the verification engine itself is machine-checked separately:
`scripts/test_task9_verify.js` — 285/285 fixture assertions vs the Python twin.)*"""),
    ],
}

out = REPO + "/colab/08_eval.ipynb"
for _i, _c in enumerate(nb["cells"]):
    _c["id"] = "cell-%02d" % _i
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# ---- gen-time validation: nbformat parse + dry-run of every code cell -------
nbformat = __import__("nbformat")
node = nbformat.read(out, as_version=4)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")            # any nbformat warning = failure
    nbformat.validate(node)
print("nbformat valid — cells:", len(node.cells))

if __name__ == "__main__":
    import io
    import contextlib
    import random

    def run_cells(ckpt_dir, run_cap, label):
        """Execute the notebook's code cells in order, in one namespace."""
        ns = {"__name__": "__main__"}
        g = []
        for i, cell in enumerate(node.cells):
            if cell.cell_type != "code":
                continue
            src = cell.source
            if "CKPT_DIR  = ROOT" in src:
                src = src.replace("CKPT_DIR  = ROOT / 'checkpoints'",
                                  "CKPT_DIR  = pathlib.Path(%r)" % ckpt_dir)
            if "RUN_CAP = None" in src:
                src = src.replace("RUN_CAP = None", "RUN_CAP = %r" % run_cap)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(src, "<cell %d>" % i, "exec"), ns)
            out_text = buf.getvalue()
            print("---- cell %d (%s): %d lines out" % (i, label, len(out_text.splitlines())))
            g.append(out_text)
        return ns, g

    # branch 1: REAL checkpoint branch (tiny smoke config at step 150)
    ns, outs = run_cells("/home/z/my-project/scripts/t5_runA_uninterrupted", 2,
                         "real-ckpt")
    m = ns["metrics"]
    assert m["config"]["answerer"].startswith("real-ckpt"), m["config"]
    assert m["stub"] is False
    assert set(m["sets"]) == {"answerable", "mythtraps", "unanswerable"}
    assert set(m["sets"]["answerable"]["per_mode"]) == set(el_MODES := (
        "raw", "grounded", "verified", "abstaining"))
    print("BRANCH real-ckpt: answerer=%s stub=%s headline=%s" % (
        m["config"]["answerer"], m["stub"], m["summary"]["headline"][:80]))
    import os as _os
    _os.environ["P1_DATA_DIR"] = ""  # reset not needed; next run re-sets it

    # branch 2: NO checkpoint -> stub fallback, full 100 questions
    ns2, outs2 = run_cells("/home/z/my-project/scripts/tmp_no_such_ckpts", None,
                           "stub")
    m2 = ns2["metrics"]
    assert m2["config"]["answerer"].startswith("stub"), m2["config"]
    assert m2["stub"] is True
    for s in ("answerable", "mythtraps", "unanswerable"):
        assert set(m2["sets"][s]["per_mode"]) == set(el_MODES)
        assert m2["sets"][s]["n"] == {"answerable": 60, "mythtraps": 20,
                                      "unanswerable": 20}[s]
    print("BRANCH stub: answerer=%s stub=%s headline=%s" % (
        m2["config"]["answerer"], m2["stub"], m2["summary"]["headline"][:80]))
    print("both branches dry-run OK — notebook metrics.json written to repo results/ (branch-2 run)")
