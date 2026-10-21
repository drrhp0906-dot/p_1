# p_1 — THE TASK FILE (execution only, no mercy)

> How to use:
> 1. Do PART 0 setup once (~30 min). Tick every box.
> 2. Each work session = fresh chat → paste CONTEXT BLOCK → paste one TASK prompt → run the code → earn the "DONE WHEN" → tick the box.
> 3. Never tick a box you didn't earn. If a session dies mid-training, that's normal weather — checkpoints handle it.
>
> Deep reference (architecture decisions, failure antidotes): see quackcheck_PLAN.md.

---

## QUICK ANSWERS

- **Colab GPU check:** new code cell → `!nvidia-smi` → Shift+Enter. GPU table prints = good. Error = Runtime → Change runtime type → T4 GPU → retry.
- **Where code runs:** everything in Colab (free) or in the browser. Your machine = browser only.
- **Where data lives:** Google Drive `MyDrive/p_1/` (survives session deaths).
- **Where code lives:** GitHub repo `p_1` (versioned, permanent).
- **Where the site lives:** `https://YOURUSERNAME.github.io/p_1/` (goes live at Task 10).
- **Token safety:** the GitHub token lives ONLY in Colab Secrets (key icon) + your password manager. Never in code, never in chat.

---

## PART 0 — SETUP CHECKLIST (one time, ~30 min)

- [ ] Drive: folder `p_1` with subfolders `corpus/`, `checkpoints/`, `results/`
- [ ] Colab: new notebook `00_hello.ipynb` → Runtime = T4 GPU → `!nvidia-smi` prints a GPU
- [ ] GitHub: repo `p_1`, Public, initialized with README
- [ ] GitHub: fine-grained token → only repo `p_1` → Contents: Read & write → copied to password manager NOW (shown once)
- [ ] GitHub: Settings → Pages → Deploy from a branch → `main` → folder `/docs` → Save
- [ ] Colab: key icon (Secrets) → `GITHUB_TOKEN` = the token
- [ ] Optional: Kaggle account (backup GPU, ~30 hrs/week, same notebooks run there)

**Setup DONE WHEN:** all boxes ticked. No code was written. That's correct — setup is only clicks.

---

## CONTEXT BLOCK (paste at the top of EVERY AI session, before any task prompt)

```
I'm building p_1, a from-scratch language-model lab: no pre-trained models
and no ML libraries — every component (autograd engine, BPE tokenizer,
transformer, AdamW optimizer, TF-IDF/BM25 retrieval, rule-based verification,
eval harness) is handmade. Two substrates:

1. BROWSER (pure JavaScript, zero dependencies): a baby character-level
   transformer that trains live in a web page, plus an inference playground
   and retrieval/verification demo for the big sibling.
2. GOOGLE COLAB free tier (pure Python + NumPy only): a ~2-10M param
   decoder-only transformer trained on a curated health Q&A corpus (~1-3MB),
   checkpoints auto-saved every 15 min to Google Drive MyDrive/p_1/,
   designed to survive session deaths.

PyTorch / TensorFlow / transformers / sklearn are BANNED (PyTorch exists
only as an escape hatch if we later vote to scale). Matplotlib allowed in
Colab for charts only — it's a pencil, not intelligence.

My role: I direct and run code but don't write it. YOUR role: write
complete, runnable code with brief design notes and a "done when" test I
can actually execute. Keep explanations tight.

Research goal: measure how handmade retrieval-grounding, claim verification,
and abstention reduce unsupported claims ("hallucinations") in a model we
raised from raw arithmetic — including a myth-contamination A/B experiment
(one sibling trained on a corpus with planted health myths, one clean).

Project naming: Drive folder = MyDrive/p_1/, GitHub repo = p_1,
web folder in repo = docs/ (served by GitHub Pages).
```

---

## PROGRESS TRACKER

| # | Task | Status | Date done |
|---|------|--------|-----------|
| 0 | Scaffold + git workflow | [ ] | |
| 1 | tensor.js autograd engine | [ ] | |
| 2 | Corpus builder | [ ] | |
| 3 | BPE tokenizer twins | [ ] | |
| 4 | Transformer twins + parity | [ ] | |
| 5 | Colab trainer (overnight) | [ ] | |
| 6 | In-browser live trainer | [ ] | |
| 7 | Weights bridge | [ ] | |
| 8 | Handmade retrieval | [ ] | |
| 9 | Verification + eval harness | [ ] | |
| 10 | Web lab UI | [ ] | |
| 11 | Experiments | [ ] | |
| 12 | Deploy + polish | [ ] | |

### TASK 0 — Scaffold + git workflow · *0.5 evening*

> Task: set up my project skeleton and Colab-to-GitHub workflow. Output:
> (1) the exact directory tree for GitHub repo p_1 — folders: docs/ (the
> future website, served by Pages), colab/ (notebooks), results/ — with a
> one-line purpose per file; (2) a README.md: 3 sentences on what p_1 is,
> plus how to open the Colab notebooks and where the site deploys; (3) the
> cells for colab/00_git_setup.ipynb that: mount Google Drive, clone the
> GitHub repo p_1 using userdata.get('GITHUB_TOKEN'), set git user.name and
> user.email, push a test commit to main; (4) a 5-command git cheat-sheet
> for my daily session ritual (pull → work → add → commit → push).
> DONE WHEN: a test file pushed from Colab appears in the GitHub repo.

**Artifact:** repo tree + README + 00_git_setup.ipynb · **Earned when:** test file visible on GitHub from your browser.

### TASK 1 — tensor.js: the autograd engine · *1 evening*

> Task: write docs/tensor.js — a reverse-mode autograd engine in pure
> JavaScript, zero dependencies. Tensor class backed by Float32Array with
> shape, data, grad. Ops: add, sub, elementwise mul, matmul, relu, tanh,
> gelu, rowwise softmax, fused crossEntropy (softmax + loss, one stable
> backward), sum, scale, reshape. Backward via topological sort. He init
> helper. Then docs/tensor_test.html — plain page running two tests:
> (a) numerical gradient check on a random composite expression, print max
> relative error, must be < 1e-3 in float32; (b) 2-layer MLP trained on XOR
> to loss < 0.01 within 2000 steps, progress logged live into the page.
> DONE WHEN: both tests pass in Chrome with green checkmarks.

**Earned when:** gradient check < 1e-3 AND XOR learned. This engine carries the entire project — do not rush it.

### TASK 2 — Corpus builder · *1 evening + your review*

> Task: write colab/01_corpus.ipynb that builds training data from scratch.
> Generate ~4000 curated simple health Q&A pairs (nutrition, sleep, hygiene,
> first aid, common conditions) in plain language, consistent with
> mainstream public guidance (NHS/WHO/Mayo style). Schema per entry:
> {id, question, answer, sources, split}; answers 2-6 sentences; split
> train/val. ALSO generate: (a) 60 answerable eval questions, (b) 20
> myth-trap questions — 'does X cause Y' where folk belief says yes but
> evidence says no (e.g. cracking knuckles → arthritis) — each with
> ground-truth correction, (c) 20 unanswerable questions with no
> established scientific answer where the correct behavior is saying so.
> Save JSONL to Drive MyDrive/p_1/corpus/. Print stats: total words,
> unique chars, top-20 tokens. Also generate corpus_with_myths.jsonl —
> same corpus plus ~300 planted myth-affirming pairs for the later
> contamination experiment. Flag entries where medical consensus is
> genuinely nuanced — I will review before training.
> DONE WHEN: JSONL files in Drive + stats printed.

**Earned when:** files exist AND you personally skimmed the myth traps.

### TASK 3 — BPE tokenizer twins · *1 evening*

> Task: implement byte-pair encoding from scratch, twice, byte-identical.
> colab/02_tokenizer.ipynb (stdlib + NumPy): train BPE to vocab ~4096 on
> the train split, save merges + vocab to MyDrive/p_1/corpus/merges.json;
> encode/decode functions; roundtrip test on 200 random corpus strings
> (lossless). docs/tokenizer.js: pure JS implementing the SAME algorithm
> loading the same merges.json. Colab generates docs/tokenizer_fixtures.json
> — 200 strings with expected token IDs from Python. docs/tokenizer_test.html
> loads fixtures, asserts identical IDs, plus a paste-a-sentence box showing
> both encodings side by side. DONE WHEN: lossless roundtrip in both
> languages and 200/200 fixture IDs match.

**Earned when:** 200/200 parity between Python and JavaScript.

### TASK 4 — Transformer twins + architecture parity · *2 evenings, hardest task*

> Task: implement the transformer twice with identical math. Architecture:
> decoder-only, pre-LayerNorm, causal self-attention (4 heads), MLP 4x
> hidden with GELU, residuals, learned positional embeddings, weight-tied
> output head. Config A (browser baby, char-level): n_layer=2, d_model=128,
> ctx=256, ~1-2M params. Config B (Colab sibling, BPE): n_layer=4,
> d_model=256, ctx=512, ~7-10M params. First colab/tensor_py.py — NumPy
> port of the same autograd design as docs/tensor.js (same op set) — then
> colab/03_model.py building both configs on it, smoke test: backward runs,
> gradients finite, 10 training steps without NaN. Then docs/model.js
> building the same graph on tensor.js. Colab generates
> docs/model_fixtures.json: for Config A, a random token sequence plus the
> exact logits from the NumPy forward pass. docs/model_test.html loads
> fixtures, asserts max logit diff < 1e-2. DONE WHEN: parity test passes
> and both configs train 10 steps without NaN.

**Earned when:** parity < 1e-2. If this fails, the likely culprit is float32 accumulation order in matmul — match the reduction order, don't loosen the tolerance.

### TASK 5 — Colab trainer: the overnight run · *1 evening + overnight*

> Task: write colab/04_train.ipynb. Implement AdamW from scratch (correct
> bias correction, decoupled weight decay), cosine LR schedule with warmup,
> global-norm gradient clipping. Data pipeline: load
> MyDrive/p_1/corpus/health_qa.jsonl, tokenize with our BPE, pack into
> ctx=512 sequences with bos/eos boundaries, shuffled batches. Training
> loop: log loss every 50 steps to CSV in Drive, plot live in the notebook;
> save checkpoint every 15 minutes ATOMICALLY to MyDrive/p_1/checkpoints/
> (params + AdamW moments + step count + config + tokenizer version); on
> startup auto-resume from newest checkpoint. Sessions WILL die — design
> for it. DONE WHEN: an overnight Config B run shows smoothly falling val
> loss, AND a deliberately-killed session resumes without loss spikes.

**Earned when:** you wake up, open the notebook, and the curve went down while you slept. First morning with a trained brain.

### TASK 6 — In-browser live trainer · *1 evening*

> Task: write docs/train.js plus docs/train_demo.html. A character-level
> Config A trained LIVE in the browser on docs/corpus_mini.js — a generated
> JS file with the top ~500 Q&A pairs from Task 2. UI: Start / Pause /
> Reset; live loss curve on raw canvas (no chart libraries); step counter;
> tokens/sec; temperature slider; Generate box that samples from current
> weights every 500 steps so I can watch it learn to talk in real time.
> Loop must keep the tab responsive: requestAnimationFrame /
> requestIdleCallback scheduling over Float32Arrays. DONE WHEN: after
> ~20-30 minutes of in-tab training it generates recognizable
> question-answer-shaped text.

**Earned when:** the baby says its first semi-coherent sentence on your screen. Take a screenshot. That one's for keeping.

### TASK 7 — Weights bridge: Colab brain into the browser · *1 evening*

> Task: build the export/import path. colab/05_export.py: load a checkpoint,
> write docs/weights/web_model.bin — little-endian binary: magic string,
> version, config-JSON length + config JSON, then each tensor in fixed
> order (name, ndim, shape, float32 data); plus manifest.json (names,
> shapes, byte offsets). docs/weights.js: fetch the .bin, parse into our
> Tensor objects, reconstruct Config B. Parity proof: colab/06_parity.ipynb
> loads the checkpoint, runs the NumPy model on 10 fixed prompts, saves
> outputs.json; docs/parity_test.html loads the SAME .bin, runs model.js on
> the same prompts, asserts max diff < 1e-2, prints both outputs side by
> side. DONE WHEN: the browser generates text using the Colab-trained brain.

**Earned when:** your cloud-trained sibling speaks through a web page.

### TASK 8 — Handmade retrieval · *1 evening*

> Task: write docs/retrieve.js — retrieval with zero libraries. Chunk the
> corpus (one chunk per Q&A pair, ~200 tokens, overlap 40); build an
> inverted index; implement TF-IDF with cosine similarity AND BM25 (k1=1.5,
> b=0.75); top-k with adjustable k; output ranked chunks with scores +
> citation labels (source ids). Test page: type a question, see ranked
> chunks. Then colab/07_retrieval_eval.ipynb: same BM25 in NumPy/stdlib;
> verify top-3 agreement with the JS version on 100 questions, target > 95%.
> DONE WHEN: both demos work and agree.

**Earned when:** > 95% top-3 agreement across two languages.

### TASK 9 — Verification + eval harness: the measurement system · *1-2 evenings*

> Task: build the machine that catches hallucinations. colab/08_eval.ipynb:
> (1) claim extractor — split answers into atomic claims, rule-based:
> sentence splitting + clause cues ('because', 'which', 'that', 'can',
> 'causes') + number extraction; (2) verifier — score each claim against
> retrieved chunks: lexical overlap (content-word F1), exact number
> matching for dosages/quantities, negation consistency; verdict per claim:
> supported / unsupported / not-in-corpus; (3) metrics — hallucination rate
> (unsupported / total claims), abstention correctness (refused
> unanswerables), myth-parroting rate (myth-trap set); (4) compare four
> modes: RAW, +GROUNDING (answer only from retrieved chunks, cite them),
> +VERIFICATION (rewrite or flag failed claims), +ABSTENTION (refuse when
> best chunk score below threshold). Output MyDrive/p_1/results/metrics.json
> + notebook charts. DONE WHEN: metrics.json compares all 4 modes on all
> 3 eval sets.

**Earned when:** the first hallucination-rate table exists. Bring snacks — this is the payoff evening.

### TASK 10 — The web lab UI · *1 evening*

> Task: build docs/index.html + style.css + app.js — 'the lab'. Single page,
> four tabs: PLAYGROUND (chat with the big sibling; toggle RAW vs GROUNDED;
> citations as numbered chips), LIVE TRAINING (embed Task 6 demo), TRAP LAB
> (run the 20 myth-traps + 20 unanswerables in-browser through raw vs
> grounded pipeline; verdicts table with per-claim colors), ABOUT (what this
> is; the experiment; latest metrics from results/). Style: clean dark lab
> notebook, monospace accents, no frameworks, no CDNs, everything local and
> relative-pathed. Mobile friendly. DONE WHEN: all four tabs work on
> desktop and phone Chrome.

**Earned when:** the site exists and you can show it to a human being.

### TASK 11 — The experiments: the actual science · *1 evening + compute*

> Task: design and run the battery in colab/09_experiments.ipynb.
> (1) Myth contamination A/B: train Config B twice — corpus_with_myths vs
> clean corpus, same seeds and schedule; compare myth-parroting rates.
> (2) Grounding delta: hallucination rate across the four modes from Task 9.
> (3) Abstention threshold sweep: refusal precision/recall tradeoff on the
> unanswerable set, pick the operating point. (4) Seed variance: 3 seeds on
> the baby config to learn how noisy these metrics even are. Produce
> MyDrive/p_1/results/summary.md — four tables + interpretation notes
> written for a non-expert. DONE WHEN: summary.md answers 'does grounding
> reduce hallucination in a model we built from scratch?' with numbers.

**Earned when:** you can answer the original research question in one sentence.

### TASK 12 — Deploy + polish · *0.5 evening*

> Task: ship it. (1) Final README: what it is, screenshots, summary numbers,
> honest limitations ('a toy model that hallucinates on purpose — that IS
> the experiment; it is not medical advice'). (2) Verify GitHub Pages serves
> the site cleanly: relative paths, correct MIME for .bin, everything
> same-origin. (3) A 3-minute live demo script: open site → train baby model
> 5 min → generate text → flip a myth trap between RAW and GROUNDED → show
> the metrics table. (4) Roadmap section: WebGPU backend, NLI-style
> verification, bigger corpus. DONE WHEN: the URL loads from a phone and
> the demo script runs end to end.

**Earned when:** a stranger can visit your creature at
`https://YOURUSERNAME.github.io/p_1/`.

---

## WHEN STUCK

- **Session died mid-training** → normal. Task 5 checkpoints every 15 min; restart, it resumes.
- **Parity test fails** → float32 accumulation order in matmul, 90% of the time. Match reduction order; don't loosen tolerance.
- **Loss NaN** → lower LR; verify gradient clipping actually applied; crossEntropy must clamp log(0).
- **Bored / restless** → open train_demo.html, watch the baby relearn Q&A shapes in 20 minutes, remember you built the engine it runs on. Then continue.
- **Stuck > 1 evening on one bug** → paste the error into the session, fresh eyes. Rule: never spin alone past one evening.
