# p_1 — a from-scratch language model lab

p_1 is a language model raised from raw arithmetic: **no PyTorch, no TensorFlow, no
pretrained anything**. Every component — autograd engine, BPE tokenizer, transformer,
AdamW optimizer, TF-IDF/BM25 retrieval, rule-based claim verification, eval harness —
is handmade **twice**, in pure JavaScript (browser) and pure Python + NumPy (Colab),
and the two twins are pinned to each other by parity tests.

The research question: **does handmade retrieval-grounding, claim verification and
abstention reduce unsupported claims ("hallucinations") in a model we built from
scratch?** — including a myth-contamination A/B experiment (one sibling trained on a
corpus with planted health myths, one clean).

> A toy model that hallucinates on purpose — that IS the experiment. It is not
> medical advice.

## The lab (open in a browser)

The website lives in **`docs/`** and is served by GitHub Pages at
`https://YOURUSERNAME.github.io/p_1/`. Locally, open `docs/index.html` directly —
every page works from `file://` via bundled snapshots. Four tabs:

- **PLAYGROUND** — chat with the Colab-trained brain (step-150 smoke model, 337k
  params); toggle RAW vs GROUNDED; numbered citation chips.
- **LIVE TRAINING** — a char-level transformer trains live in the tab on 500 Q&A
  pairs; live loss curve, tokens/sec, temperature slider, generate box.
- **TRAP LAB** — 20 myth-traps + 20 unanswerables run through the raw vs grounded
  pipeline, with per-claim verdict chips.
- **ABOUT** — what this is, the experiment, the latest metrics table.

Self-test pages: [tensor_test.html](docs/tensor_test.html) (engine gradient check +
XOR), [tokenizer_test.html](docs/tokenizer_test.html) (200/200 BPE parity),
[model_test.html](docs/model_test.html) (transformer logits parity),
[parity_test.html](docs/parity_test.html) (the Colab brain speaks in the browser),
[train_demo.html](docs/train_demo.html), [retrieve_test.html](docs/retrieve_test.html).

## Repository layout

```
p_1/
├── docs/                     the website (GitHub Pages serves this folder)
│   ├── index.html            the lab — 4 tabs, dark lab-notebook style
│   ├── tensor.js             reverse-mode autograd engine (17 ops, Float32Array)
│   ├── model.js              decoder-only transformer twin (pre-LN, GELU, weight-tied)
│   ├── tokenizer.js          byte-level BPE twin (GPT-2-style rank merging)
│   ├── train.js              in-browser trainer (AdamW, cosine LR, canvas loss curve)
│   ├── retrieve.js           zero-dependency retrieval (inverted index, TF-IDF + BM25)
│   ├── verify.js             rule-based claim verifier (overlap, numbers, negation)
│   ├── weights.js            web_model.bin parser (little-endian "p1wm" format)
│   ├── app.js                the lab UI wiring
│   ├── *_test.html           the self-test/parity pages above
│   ├── data/                 corpus + trap snapshots for the site (file://-friendly)
│   ├── results/              metrics.json (real) + metrics.js (stub fallback snapshot)
│   ├── weights/              web_model.bin + manifest.json + base64 snapshot
│   ├── model_fixtures.json   4 contract cases: NumPy logits the JS twin must match
│   ├── tokenizer_fixtures.json  200 cases with exact Python-twin token ids
│   ├── verify_fixtures.json  54 verifier cases with expected verdicts
│   └── screenshots/          README evidence images (pages + tests passing)
├── colab/                    the Colab / NumPy sibling (pure Python + NumPy only)
│   ├── 00_git_setup.ipynb    mounts Drive, clones the repo, pushes a test commit
│   ├── 01_corpus.ipynb       builds the health Q&A corpus from 6 curated seed banks
│   ├── 02_tokenizer.ipynb    trains byte-level BPE → corpus/merges.json + fixtures
│   ├── 03_model.py           both transformer configs on the NumPy autograd twin
│   ├── 04_train.ipynb        overnight trainer: AdamW, cosine LR, 15-min atomic ckpts
│   ├── 05_export.py          checkpoint → docs/weights/web_model.bin + parity fixtures
│   ├── 06_parity.ipynb       10-prompt parity proof: NumPy vs browser
│   ├── 07_retrieval_eval.ipynb  BM25 twin; >95% top-3 agreement with the JS side
│   ├── 08_eval.ipynb         verification + eval harness → results/metrics.json
│   ├── 09_experiments.ipynb  the experiment battery → results/summary.md
│   ├── tensor_py.py          NumPy autograd engine (port of docs/tensor.js)
│   ├── tokenizer_py.py       BPE twin (train_bpe / P1BPE / fixtures writer)
│   ├── train_lib.py          AdamW, data pipeline, Trainer with atomic resume
│   ├── retrieval_py.py       BM25/TF-IDF Python twin
│   ├── corpus_lib.py         corpus builder (writes ./corpus/ + stats)
│   ├── eval_lib.py           claim extraction, verification, 4-mode metrics
│   ├── experiments_lib.py    the 4 experiments + summary writer
│   ├── gen_model_fixtures.py regenerates docs/model_fixtures.json (+_data.js)
│   ├── test_tensor_py.py     NumPy engine gradient check (mirror of Task 1 gate)
│   └── seed_*.py             6 hand-curated seed banks (236 core facts)
├── corpus/                   the data (also mirrored to Drive MyDrive/p_1/corpus/)
│   ├── health_qa.jsonl       4,278 curated Q&A pairs (train/val split)
│   ├── corpus_with_myths.jsonl  + 300 planted myth pairs (contamination arm)
│   ├── merges.json           the trained BPE merges (vocab 4096)
│   ├── eval_answerable.jsonl / eval_mythtraps.jsonl / eval_unanswerable.jsonl
│   └── nuanced_flags.json    nuanced medical claims flagged for human review
├── scripts/                  local acceptance gates + generators
│   ├── test_task1_tensor.js  engine gradient check + XOR (Node)
│   ├── test_task3_tokenizer.js  BPE twins parity — PARITY 200/200 PASS
│   ├── test_task4_model.js   transformer logits parity vs NumPy fixtures
│   ├── test_task5_train.py   trainer acceptance: determinism, atomic ckpts, kill/resume
│   ├── test_task6_train.js   in-browser trainer smoke (loss falls, samples form)
│   ├── test_task7_bridge.js  full export→load→parity path (needs a Task 5 ckpt)
│   ├── test_task8_retrieval.js  retrieval twins >95% top-3 agreement
│   ├── test_task9_eval.py    verifier + eval harness gates
│   ├── test_task9_verify.js  JS/Python verifier parity 285/285
│   ├── test_task10_lab.js    lab UI smoke (23 checks)
│   ├── test_task11_experiments.py  experiment battery local smoke + regressions
│   ├── audit_pages.py        GitHub Pages deploy audit (every src/href/fetch resolves)
│   └── gen_*.py / gen_*.js   notebook + data generators (ROOT-relative, runnable)
└── results/                  experiment artifacts
    ├── metrics.json          4 modes × 3 eval sets (hallucination/refusal rates)
    ├── summary.md            the four experiment tables + non-expert notes
    ├── exp11_*.json          grounding / myth A/B / tau sweep / seed variance
    ├── parity_outputs.json   10-prompt NumPy outputs (the parity fixture)
    ├── metrics_stub.json     stub metrics (schema demo, stub:true)
    └── train_loss.png        Task 5 overnight loss curve
```

## Running the tests

All local gates are deterministic and CPU-only (Node ≥ 18, Python ≥ 3.10, NumPy):

```bash
node scripts/test_task1_tensor.js     # ALL TESTS PASSED — engine vs f64 mirror + XOR
node scripts/test_task3_tokenizer.js  # PARITY 200/200 PASS
node scripts/test_task4_model.js      # ALL TESTS PASSED — worst |Δlogit| ~9e-6
python3 scripts/test_task5_train.py   # RESUME CLEAN ✓ (trains tiny models in /tmp)
P1_SMOKE_STEPS=150 node scripts/test_task6_train.js   # TASK6 SMOKE PASS
node scripts/test_task7_bridge.js     # needs scripts/t5_runA_uninterrupted (see below)
node scripts/test_task8_retrieval.js  # >95% top-3 agreement
python3 scripts/test_task9_eval.py    # ALL GATES GREEN
node scripts/test_task9_verify.js     # VERIFY PARITY 285/285
node scripts/test_task10_lab.js       # LAB SMOKE: 23 PASS / 0 FAIL
python3 scripts/test_task11_experiments.py  # 8-9 min: full battery + regressions
python3 scripts/audit_pages.py        # RESULT: CLEAN — docs/ ready to ship
```

`test_task7_bridge.js` needs a Task 5 checkpoint dir (gitignored — regenerable):
`python3 scripts/test_task5_train.py --mode run-a --ckpt-dir scripts/t5_runA_uninterrupted`.

## Colab workflow

1. `colab/00_git_setup.ipynb` — mount Drive, clone the repo (token in Colab Secrets).
2. `01_corpus` → `02_tokenizer` → `04_train` (overnight, checkpoints auto-resume) →
   `05_export` (writes `docs/weights/web_model.bin`) → `06_parity` → `07/08/09`.
3. Data lives in `MyDrive/p_1/corpus/` (or the repo's `corpus/` when running locally;
   `P1_DATA_DIR` overrides both).

## Reorganisation note (2026-10)

This tree was reorganised from a flat working snapshot into the canonical layout the
code itself expects (docs/ site + data/, colab/ twins + seed banks, corpus/, scripts/
gates, results/). Four files referenced by the codebase were absent from the snapshot
and were reconstructed and verified: `colab/tokenizer_py.py` (port of
docs/tokenizer.js — 200/200 fixture parity, and re-training the corpus reproduces
`corpus/merges.json` byte-for-byte), `scripts/test_task1_tensor.js` +
`scripts/test_task4_model.js` (Node mirrors of the Python/browser gates), and
`docs/tensor_test.html` (rebuilt from the shipped screenshot; passes in-browser).
`results/` holds the original experiment artifacts.

## Roadmap

WebGPU backend · NLI-style verification · bigger corpus · scale-up vote (PyTorch
escape hatch). See [p1_TASKS.md](p1_TASKS.md) for the full task roadmap and
[quackcheck_PLAN.md](quackcheck_PLAN.md) for architecture decisions.
