# QuackCheck — Master Build Plan

> A language model built from trash: no pre-trained models, no ML libraries.
> Everything handmade — autograd engine, tokenizer, transformer, optimizer,
> retrieval, verification, evaluation. Trained in the cloud (free), living in
> a browser tab (free), studying its own hallucinations.

---

## CONTEXT BLOCK (paste this at the top of EVERY AI session prompt)

```
I'm building QuackCheck, a from-scratch language-model lab: no pre-trained
models and no ML libraries — every component (autograd engine, BPE tokenizer,
transformer, AdamW optimizer, TF-IDF/BM25 retrieval, rule-based verification,
eval harness) is handmade. Two substrates:

1. BROWSER (pure JavaScript, zero dependencies): a baby character-level
   transformer that trains live in a web page, plus an inference playground
   and retrieval/verification demo for the big sibling.
2. GOOGLE COLAB free tier (pure Python + NumPy only): a ~2-10M param
   decoder-only transformer trained on a curated health Q&A corpus (~1-3MB),
   checkpoints auto-saved to Google Drive, designed to survive session deaths.

PyTorch / TensorFlow / transformers / sklearn are BANNED (PyTorch exists only
as an escape hatch if we later vote to scale — decision recorded in PLAN.md).
Matplotlib is allowed in Colab for drawing charts only (it's a pencil, not
intelligence).

My role: I direct and run code but don't write it. YOUR role: write complete,
runnable code with brief design notes and a "done when" test I can actually
execute. Keep explanations tight.

Research goal: measure how handmade retrieval-grounding, claim verification,
and abstention reduce unsupported claims ("hallucinations") in a model we
raised from raw arithmetic — including a myth-contamination A/B experiment
(one sibling trained on a corpus with planted health myths, one clean).
```

Then paste the specific TASK prompt below it.

---

## REPO STRUCTURE

```
quackcheck/                      ← GitHub repo (public, free)
├── README.md
├── PLAN.md                      ← this file lives here
├── docs/                        ← the web lab — GitHub Pages serves THIS folder
│   ├── index.html               ← QuackCheck Lab (tabs UI)
│   ├── style.css
│   ├── app.js
│   ├── tensor.js                ← Task 1: autograd engine
│   ├── tokenizer.js             ← Task 3: BPE
│   ├── model.js                 ← Task 4: transformer
│   ├── train.js                 ← Task 6: live in-browser trainer
│   ├── retrieve.js              ← Task 8: TF-IDF / BM25
│   ├── verify.js                ← Task 9: claim verification
│   ├── weights.js               ← Task 7: .bin loader
│   ├── tensor_test.html         ← gradient check page
│   ├── tokenizer_test.html      ← Python/JS parity page
│   ├── model_test.html          ← logit parity page
│   ├── parity_test.html         ← weights parity page
│   ├── train_demo.html          ← live training demo
│   ├── corpus_mini.js           ← baby-model mini corpus (generated)
│   └── weights/
│       ├── web_model.bin        ← exported from Colab (Task 7)
│       └── manifest.json
├── colab/                       ← notebook + scripts that run on Colab
│   ├── 00_git_setup.ipynb
│   ├── 01_corpus.ipynb
│   ├── 02_tokenizer.ipynb
│   ├── tensor_py.py             ← NumPy port of tensor.js
│   ├── 03_model.py
│   ├── 04_train.ipynb
│   ├── 05_export.py
│   ├── 06_parity.ipynb
│   ├── 07_retrieval_eval.ipynb
│   ├── 08_eval.ipynb
│   └── 09_experiments.ipynb
└── results/
    ├── metrics.json
    ├── summary.md
    └── charts/
```

Google Drive mirror: `MyDrive/quackcheck/` with subfolders `corpus/`,
`checkpoints/`, `results/`.

---

## PART 0 — HUMAN SETUP CHECKLIST (you, one time, ~30 minutes, zero installs)

### A. Google side (~10 min)

1. Sign in to your Google account.
2. Google Drive → create folder `quackcheck` with three subfolders:
   `corpus`, `checkpoints`, `results`.
3. Open `colab.research.google.com` → File → New notebook → rename it
   `00_hello.ipynb`.
4. Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save.
5. In a cell run: `!nvidia-smi` — if a GPU table prints, you're golden.
6. Reality notes: free sessions disconnect on idle (~90 min) and have a max
   session length; GPU availability varies. This is why Task 5 auto-saves
   checkpoints to Drive every 15 minutes — sessions dying is a normal
   weather event, not a disaster.

### B. GitHub side (~10 min)

1. Sign up / sign in at `github.com`. Verify your email.
2. New repository: name `quackcheck`, visibility **Public** (required for
   free Pages), check "Add a README".
3. Create an access token:
   - Profile photo → Settings → Developer settings →
     Personal access tokens → **Fine-grained tokens** → Generate new token.
   - Token name: `colab-push`. Repository access: **Only select
     repositories** → `quackcheck`.
   - Permissions → Repository permissions → Contents: **Read and write**.
   - Expiration: 90 days is fine.
   - Copy the token NOW — it is shown exactly once. Store it in a password
     manager. NEVER paste it into code, notebooks, or chat.
4. Enable GitHub Pages: repo → Settings → Pages → Source: **Deploy from a
   branch** → Branch: `main`, Folder: **/docs** → Save.
   Your site will live at `https://YOURUSERNAME.github.io/quackcheck/`
   (it goes live once `docs/index.html` exists — Task 10).

### C. Wiring Colab to both (~10 min)

1. In any Colab notebook: click the **key icon** (Secrets, left sidebar) →
   Add a secret → Name: `GITHUB_TOKEN`, Value: your token → toggle
   "Enable notebook access". Now notebooks can `userdata.get('GITHUB_TOKEN')`
   without the token ever appearing in code.
2. Mount Drive (Task 0's notebook does this; the cell is one line:
   `from google.colab import drive; drive.mount('/content/drive')`).
3. Optional backup GPU: Kaggle (`kaggle.com`) offers ~30 free GPU-hours/week
   and runs the same notebooks. Set up only if Colab quota ever runs dry.

### D. Done when

- [ ] `!nvidia-smi` prints a GPU in Colab
- [ ] Repo `quackcheck` exists, public
- [ ] Token stored in Colab Secrets (name: GITHUB_TOKEN) + password manager
- [ ] Pages source set to main branch /docs folder
- [ ] Drive folders `corpus/ checkpoints/ results/` exist under `quackcheck/`

Your local machine needs: a browser. That's it. Bookmark your future site URL.

---

## SESSION PROTOCOL (how you and the AI work)

1. Open a fresh chat session.
2. Paste: CONTEXT BLOCK + the current task's prompt (from Part 2 below).
3. The AI writes complete code + a "done when" test.
4. Paste code into Colab / commit to repo (Task 0 gives you the git cells).
5. Run the test. If it fails, paste the error back into the same chat.
6. When it passes: tick the checkbox in this file, commit PLAN.md, next task.

Rules that keep this fun: one task per session; never skip a "done when";
if a session dies mid-training, that's what checkpoints are for.

---

## PART 2 — TASK LIBRARY (13 paste-ready prompts)

Checkbox list for tracking:

- [ ] Task 0 — Scaffold & git workflow (0.5 evening)
- [ ] Task 1 — tensor.js autograd engine (1 evening)
- [ ] Task 2 — Corpus builder (1 evening + your review)
- [ ] Task 3 — BPE tokenizer twins (1 evening)
- [ ] Task 4 — Transformer twins + architecture parity (2 evenings)
- [ ] Task 5 — Colab trainer with checkpointing (1 evening + overnight run)
- [ ] Task 6 — In-browser live trainer (1 evening)
- [ ] Task 7 — Weights bridge + parity (1 evening)
- [ ] Task 8 — Handmade retrieval (1 evening)
- [ ] Task 9 — Verification + eval harness (1-2 evenings)
- [ ] Task 10 — The web lab UI (1 evening)
- [ ] Task 11 — Experiments (1 evening + compute time)
- [ ] Task 12 — Deploy & polish (0.5 evening)

Total: roughly 13-15 focused evenings. Flexible — the pace is yours.

### TASK 0 — Scaffold & git workflow

> Task: set up the project skeleton and my Colab-to-GitHub workflow. Output:
> (1) the exact directory tree for the repo per PLAN.md with a one-line
> purpose per file; (2) a README.md — 3 sentences on what QuackCheck is,
> plus how to open the Colab notebooks and where the site deploys; (3) the
> cells for colab/00_git_setup.ipynb that: mount Google Drive, clone the
> GitHub repo using a token from Colab Secrets (userdata.get), configure
> git user.name/user.email, and push a test commit to main; (4) a git
> cheat-sheet of the 5 commands I'll use every session. Done when: a test
> file pushed from Colab appears in the GitHub repo.

**Artifact:** repo tree, README, 00_git_setup.ipynb · **Done when:** test file visible on GitHub.

### TASK 1 — tensor.js: the autograd engine

> Task: write docs/tensor.js — a reverse-mode autograd engine in pure
> JavaScript, zero dependencies. Requirements: Tensor class backed by
> Float32Array with shape, data and grad; ops: add, sub, elementwise mul,
> matmul, relu, tanh, gelu, rowwise softmax, fused crossEntropy (softmax +
> loss with one stable backward), sum, scale, reshape; backward via
> topological sort; He initialization helper. Then write
> docs/tensor_test.html — a plain page that runs two tests: (a) numerical
> gradient check on a random composite expression, printing max relative
> error (must be < 1e-3 in float32); (b) a 2-layer MLP trained on XOR to
> loss < 0.01 within 2000 steps, logging progress into the page live.
> Done when: both tests pass in Chrome with green checkmarks.

**Artifact:** tensor.js + tensor_test.html · **Done when:** gradient check < 1e-3, XOR learned. This is the foundation everything else stands on.

### TASK 2 — Corpus builder

> Task: write colab/01_corpus.ipynb that builds the training data from
> scratch. Generate ~4000 curated simple health Q&A pairs (nutrition, sleep,
> hygiene, first aid, common conditions) in plain language, consistent with
> mainstream public guidance (NHS/WHO/Mayo style). Schema per entry:
> {id, question, answer, sources, split}; answers 2-6 sentences; split
> train/val. ALSO generate three eval sets: (a) 60 answerable questions,
> (b) 20 myth-trap questions — 'does X cause Y' where folk belief says yes
> but evidence says no (e.g. cracking knuckles / arthritis) — each with
> ground-truth correction, (c) 20 unanswerable questions with no established
> scientific answer, where the correct behavior is to say so. Save all as
> JSONL to Drive corpus/. Print corpus stats: total words, unique chars,
> top-20 tokens. Also generate a variant corpus_with_myths.jsonl: same but
> with ~300 planted myth-affirming pairs (for the later contamination
> experiment). Flag any entries where medical consensus is genuinely nuanced
> — I will personally review before training. Done when: JSONL files in
> Drive + stats printed.

**Artifact:** corpus JSONLs in Drive · **Done when:** stats print, you've skimmed the myth traps and had at least one "people actually believe this?" moment.

### TASK 3 — BPE tokenizer, twin implementations

> Task: implement byte-pair encoding from scratch, twice, byte-identical.
> colab/02_tokenizer.ipynb (stdlib + NumPy): train BPE to vocab ~4096 on the
> train split, save merges + vocab to corpus/merges.json; encode/decode
> functions; roundtrip test on 200 random corpus strings (must be lossless).
> docs/tokenizer.js: pure JS implementing the SAME algorithm loading the
> same merges.json. colab generates docs/tokenizer_fixtures.json — 200
> strings with expected token IDs from Python. docs/tokenizer_test.html
> loads fixtures and asserts identical IDs, plus a paste-a-sentence box
> showing both encodings side by side. Done when: lossless roundtrip in
> both languages and 200/200 fixture IDs match.

**Artifact:** 02_tokenizer.ipynb, tokenizer.js, fixtures, test page · **Done when:** 200/200 parity.

### TASK 4 — Transformer twins + architecture parity (hardest task, take 2 evenings)

> Task: implement the transformer twice with identical math. Architecture:
> decoder-only, pre-LayerNorm, causal self-attention with 4 heads, MLP 4x
> hidden with GELU, residuals, learned positional embeddings, weight-tied
> output head. Config A (browser baby, char-level): n_layer=2, d_model=128,
> ctx=256 — roughly 1-2M params. Config B (Colab sibling, BPE):
> n_layer=4, d_model=256, ctx=512 — roughly 7-10M params. First write
> colab/tensor_py.py — a NumPy port of the same autograd design as
> docs/tensor.js (same op set) — then colab/03_model.py building both
> configs on it, with a random-config smoke test checking backward runs and
> gradients are finite. Then docs/model.js building the same graph on
> tensor.js. colab generates docs/model_fixtures.json: for Config A, a
> random token sequence plus the exact logits from the NumPy forward pass.
> docs/model_test.html loads fixtures and asserts max logit diff < 1e-2.
> Done when: parity test passes and both configs train 10 steps without NaN.

**Artifact:** tensor_py.py, 03_model.py, model.js, fixtures, test page · **Done when:** parity < 1e-2, 10-step smoke train clean.

### TASK 5 — Colab trainer (the overnight run)

> Task: write colab/04_train.ipynb. Implement AdamW from scratch (correct
> bias correction, decoupled weight decay), cosine LR schedule with warmup,
> global-norm gradient clipping. Data pipeline: load corpus/health_qa.jsonl
> from Drive, tokenize with our BPE, pack into ctx=512 sequences with bos/eos
> boundaries, shuffled batches. Training loop: log loss every 50 steps to
> CSV in Drive, plot live in the notebook (matplotlib allowed — it's a
> pencil); save a checkpoint every 15 minutes ATOMICALLY to
> checkpoints/ (params + AdamW moments + step count + config + tokenizer
> version); on startup, auto-resume from the newest checkpoint. Sessions
> WILL die — design for it. Done when: an overnight Config B run shows a
> smoothly falling val loss, AND a deliberately-killed session resumes
> without loss spikes.

**Artifact:** 04_train.ipynb + first real checkpoint · **Done when:** overnight curve converges, resume works. First morning you wake up with a trained brain.

### TASK 6 — In-browser live trainer (the demo piece)

> Task: write docs/train.js plus docs/train_demo.html. A character-level
> Config A trained LIVE in the browser on docs/corpus_mini.js — a generated
> JS file containing the top ~500 Q&A pairs from Task 2. UI: Start / Pause /
> Reset buttons; live loss curve drawn on raw canvas (no chart libraries);
> step counter; tokens/sec; a temperature slider; and a Generate box that
> samples from the current weights every 500 steps so I can watch it learn
> to talk in real time. Training loop must keep the tab responsive: batch
> steps scheduled via requestAnimationFrame / requestIdleCallback on
> Float32Arrays. Done when: after ~20-30 minutes of in-tab training it
> generates recognizable question-answer-shaped text.

**Artifact:** train.js + train_demo.html + corpus_mini.js · **Done when:** the baby says its first semi-coherent sentence on screen. Send me a screenshot.

### TASK 7 — Weights bridge: Colab brain into the browser

> Task: build the export/import path. colab/05_export.py: load a checkpoint,
> write docs/weights/web_model.bin — a little-endian binary: magic string,
> version, config-JSON length + config JSON, then each tensor in a fixed
> order (name, ndim, shape, float32 data); also write manifest.json listing
> names, shapes and byte offsets. docs/weights.js: fetch the .bin, parse it
> into our Tensor objects, reconstruct Config B. Parity proof:
> colab/06_parity.ipynb loads the checkpoint, runs the NumPy model on 10
> fixed prompts, saves outputs.json; docs/parity_test.html loads the SAME
> .bin, runs model.js on the same prompts, asserts max diff < 1e-2 and
> prints both outputs side by side. Done when: the browser playground
> generates text using the Colab-trained brain.

**Artifact:** 05_export.py, weights.js, parity test · **Done when:** your cloud-trained sibling speaks through a web page.

### TASK 8 — Handmade retrieval

> Task: write docs/retrieve.js — retrieval with zero libraries. Chunk the
> corpus (one chunk per Q&A pair, ~200 tokens, overlap 40); build an
> inverted index; implement TF-IDF with cosine similarity AND BM25
> (k1=1.5, b=0.75); top-k retrieval with an adjustable k. Output ranked
> chunks with scores + citation labels (source ids). Add a test page:
> type a question, see ranked chunks. Then colab/07_retrieval_eval.ipynb:
> the same BM25 in NumPy/stdlib; verify top-3 agreement with the JS version
> on 100 questions (target > 95%). Done when: both demos work and agree.

**Artifact:** retrieve.js + eval notebook · **Done when:** > 95% top-3 agreement across languages.

### TASK 9 — Verification + eval harness (the measurement system)

> Task: build the system that catches hallucinations. colab/08_eval.ipynb:
> (1) claim extractor — split an answer into atomic claims, rule-based:
> sentence splitting + clause cues ('because', 'which', 'that', 'can',
> 'causes') + number extraction; (2) verifier — score each claim against
> retrieved chunks using lexical overlap (content-word F1), exact number
> matching for dosages/quantities, and negation consistency; verdict per
> claim: supported / unsupported / not-in-corpus; (3) metrics —
> hallucination rate (unsupported / total claims), abstention correctness
> (refused unanswerables), myth-parroting rate (myth-trap set); (4) run four
> modes and compare: RAW, +GROUNDING (answer only from retrieved chunks,
> cite them), +VERIFICATION (rewrite or flag failed claims), +ABSTENTION
> (refuse when best chunk score is below threshold). Output
> results/metrics.json + notebook charts. Done when: metrics.json compares
> all 4 modes on all 3 eval sets.

**Artifact:** 08_eval.ipynb, metrics.json · **Done when:** the first hallucination-rate table exists.

### TASK 10 — The web lab UI

> Task: build docs/index.html + style.css + app.js — 'QuackCheck Lab'. A
> single page, four tabs: PLAYGROUND (chat with the big sibling; toggle RAW
> vs GROUNDED mode; citations shown as numbered chips), LIVE TRAINING
> (embed the Task 6 demo), TRAP LAB (run the 20 myth-traps and 20
> unanswerables in-browser through raw vs grounded pipeline; verdicts table
> with per-claim colors), ABOUT (what this is: a from-scratch LLM; the
> experiment; the latest metrics from results/). Style: clean dark lab
> notebook, monospace accents, no frameworks, no CDNs, everything local and
> relative-pathed (Pages will serve it from /docs). Mobile friendly. Done
> when: all four tabs work on desktop and phone Chrome.

**Artifact:** the actual website · **Done when:** it exists and you can show it to a human.

### TASK 11 — The experiments (the actual science)

> Task: design and run the experiment battery in colab/09_experiments.ipynb.
> (1) Myth contamination A/B: train Config B twice — corpus_with_myths vs
> clean corpus — same seeds and schedule; compare myth-parroting rates.
> (2) Grounding delta: hallucination rate across the four modes from Task 9.
> (3) Abstention threshold sweep: refusal precision/recall tradeoff on the
> unanswerable set, pick the operating point. (4) Seed variance: 3 seeds on
> the baby config to see how noisy these metrics even are. Produce
> results/summary.md with four tables and interpretation notes written for
> a non-expert. Done when: summary.md answers 'does grounding reduce
> hallucination in a model we built from scratch?' with numbers.

**Artifact:** 09_experiments.ipynb, summary.md, charts · **Done when:** you can answer the original research question in one sentence.

### TASK 12 — Deploy & polish

> Task: ship it. (1) Final README: what it is, screenshots, the summary
> numbers, honest limitations ('a toy model that hallucinates on purpose —
> that is the experiment; it is not medical advice'). (2) Verify GitHub
> Pages serves the site cleanly: relative paths, correct MIME for .bin,
> no CORS issues (everything same-origin). (3) Write a 3-minute live demo
> script: open site → train baby model 5 minutes → generate text → flip a
> myth trap between RAW and GROUNDED → show the metrics table. (4) Roadmap
> section: WebGPU backend, NLI-style verification, bigger corpus. Done
> when: the URL loads from a phone and the demo script runs end to end.

**Artifact:** live URL, final README · **Done when:** a stranger can visit your creature.

---

## DECISIONS LOG (update as we go)

| # | Decision | Choice | Date |
|---|----------|--------|------|
| 1 | Calculator boundary | Pure JS in browser; pure Python + NumPy in Colab; matplotlib for charts only | set at plan creation |
| 2 | Escape hatch | If Config B on Colab CPU cannot reach acceptable loss overnight, vote on allowing PyTorch for TRAINING only (weights still exported to our .bin format) | open |
| 3 | Hosting | GitHub Pages from /docs, branch main | set at plan creation |
| 4 | GPU backup | Kaggle notebooks (~30 GPU-hrs/week) if Colab quota runs out | open |
| 5 | Project name | QuackCheck | set at plan creation |

## FAILURE MODES & ANTIDOTES (read when stuck)

- **Colab session died mid-training** — normal weather. Task 5 checkpoints every 15 min; restart and it resumes.
- **JS parity test fails** — 90% of the time it is float32 accumulation order in matmul; fix by matching the reduction order, not by loosening the tolerance.
- **Loss NaNs** — lower LR, check gradient clipping is actually applied, check for log(0) in crossEntropy (the fused op must clamp).
- **Motivation dip** — open train_demo.html, watch the baby relearn Q&A shapes in 20 minutes, remember you built the engine it runs on. Then continue.
