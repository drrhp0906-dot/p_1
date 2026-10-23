#!/usr/bin/env python3
"""Generate colab/07_retrieval_eval.ipynb (Task 8) — nbformat 4.5, validated.

Run:  python3 scripts/gen_07_retrieval_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md("""\
# 07 · Retrieval eval — the handmade retriever meets its 100 questions

**Task 8.** The retrieval twins are hand-written (no ML libraries):

- **JS twin** `docs/retrieve.js` — inverted index + BM25/TF-IDF, runs the browser demo
  (`docs/retrieve_test.html`) and the Node acceptance test (`scripts/test_task8_retrieval.js`).
- **Python twin** `colab/retrieval_py.py` — the *same* algorithm contract: tokens `[a-z0-9]+`
  (lowercase, no stopwords), one chunk per Q&A pair (`"Q: …\\nA: …"`, chunkId = entry id),
  inverted index term → (chunkIdx, tf), **BM25** `idf = ln((N−df+0.5)/(df+0.5)+1)`, k1=1.5, b=0.75,
  **TF-IDF** cosine `(1+ln tf)·ln(N/df)`, deterministic tie-break (score DESC, chunkIdx ASC).

This notebook builds the index over `corpus/health_qa.jsonl` (4,278 chunks), runs the
**100-question eval** (60 answerable `ea-*` + 20 myth traps `mt-*` + 20 unanswerable `ua-*`)
with **both methods**, writes `results/retrieval_top3.json`, and prints sanity metrics.

The JS↔Python **agreement gate** (>95% top-3, both methods) is then checked by
`scripts/test_task8_retrieval.js` — same numbers, two languages."""))

cells.append(md("## Cell 1 — build the index over the full corpus"))

cells.append(code("""\
import sys, time
from pathlib import Path

# locate the p_1 repo (Colab clone > cwd > parent > Drive mount)
CANDIDATES = [Path("/content/p_1"), Path.cwd(), Path.cwd().parent,
              Path("/content/drive/MyDrive/p_1")]
REPO = next((c for c in CANDIDATES if (c / "corpus" / "health_qa.jsonl").exists()), None)
assert REPO is not None, "health_qa.jsonl not found — mount Drive or clone the repo first"
print("repo:", REPO)
sys.path.insert(0, str(REPO / "colab"))

import retrieval_py as rp   # the Python twin (pure stdlib)

corpus = rp.load_jsonl(rp.data_dir() / "health_qa.jsonl")   # $P1_DATA_DIR > Drive > repo/corpus
t0 = time.perf_counter()
index = rp.build_index(corpus)
build_ms = (time.perf_counter() - t0) * 1000

print(f"chunks      : {index['N']}  (one per Q&A pair, file order — deterministic)")
print(f"vocab size  : {index['vocab']}  (lowercase [a-z0-9]+ tokens, NO stopwords)")
print(f"avgdl       : {index['avgdl']:.3f} tokens/chunk")
print(f"index build : {build_ms:.1f} ms  (inverted index: term -> [(chunkIdx, tf)], df, norms)")"""))

cells.append(md("""\
## Cell 2 — the 100-question eval, both methods

Reads **all three** eval files (60 + 20 + 20 = exactly 100), computes top-3 for
**BM25 and TF-IDF** via the Python twin, and prints sanity metrics:

- *self-test*: an exact corpus question must retrieve its own fact at rank 1
  (siblings — same question text, different answer variant — count; the corpus
  has 1,239 unique questions over 4,278 chunks),
- *own-id sanity*: the 60 answerable entries appended as chunks — does the
  question find the chunk holding **its own gold answer** among ~4.3k
  near-duplicates? (honest signal, not the gate)."""))

cells.append(code("""\
EVAL_FILES = ["eval_answerable.jsonl", "eval_mythtraps.jsonl", "eval_unanswerable.jsonl"]
questions, counts = [], {}
for fname in EVAL_FILES:
    rows = rp.load_jsonl(rp.data_dir() / fname)
    counts[fname] = len(rows)
    questions += [{"id": r["id"], "question": r["question"]} for r in rows]
assert len(questions) == 100, f"expected exactly 100 eval questions, got {len(questions)}"
print("eval sets  :", counts, "-> 100 questions total")

top3 = {}          # (method, id) -> {"top3": [...], "scores": [...]}
t0 = time.perf_counter()
for method in ("bm25", "tfidf"):
    for q in questions:
        hits = index["search"](q["question"], k=3, method=method)
        top3[(method, q["id"])] = {"top3": [h["chunkId"] for h in hits],
                                   "scores": [h["score"] for h in hits]}
per_query_ms = (time.perf_counter() - t0) * 1000 / (2 * len(questions))
print(f"searched   : 100 questions x 2 methods in {per_query_ms:.2f} ms/query avg")

# FYI: how much do the two methods agree with EACH OTHER on this corpus?
same = sum(top3[("bm25", q["id"])]["top3"] == top3[("tfidf", q["id"])]["top3"] for q in questions)
print(f"bm25≈tfidf : {same}/100 identical top-3 lists (methods differ, twins must not)")

# sanity 1 — exact-question self-test (10 deterministic samples)
st = rp.selftest_exact_question(index, corpus, 10)
print(f"self-test  : exact corpus question -> own fact rank 1 : {st['hits']}/10 "
      f"(strict own-id {st['strict_own_id']}/10)")
assert st["hits"] == 10, f"self-test fails: {st['fails']}"

# sanity 2 — own-id hit-rate on the answerable set (eval entries appended as chunks)
answerable = rp.load_jsonl(rp.data_dir() / "eval_answerable.jsonl")
own = rp.own_id_sanity(corpus, answerable)
for m, v in own.items():
    print(f"own-id {m:5s} : MRR@3 {v['mrr_at3']:.3f} · hit@1 {v['hit_at1']:.3f} · hit@3 {v['hit_at3']:.3f}  (n={v['n']})")"""))

cells.append(md("""\
## Cell 3 — write `results/retrieval_top3.json`

Format `p1-retrieval-eval-v1`: one entry per (question × method) — 200 entries —
with the deterministic top-3 chunkIds + scores. The Node acceptance test
(`scripts/test_task8_retrieval.js`) recomputes the same 200 top-3 lists with the
**JS** twin and compares element-wise."""))

cells.append(code("""\
import json

out = {
    "format": "p1-retrieval-eval-v1",
    "meta": {
        "corpus": "health_qa.jsonl",
        "n_chunks": index["N"],
        "vocab": index["vocab"],
        "avgdl": round(index["avgdl"], 6),
        "k": 3,
        "methods": ["bm25", "tfidf"],
        "n_questions": len(questions),
        "eval_files": EVAL_FILES,
        "retrieval_py_version": rp.VERSION,
    },
    "questions": [
        {
            "id": q["id"],
            "question": q["question"],
            "method": method,
            "top3": top3[(method, q["id"])]["top3"],
            "scores": top3[(method, q["id"])]["scores"],
        }
        for method in ("bm25", "tfidf") for q in questions
    ],
}
out_path = rp.results_dir() / "retrieval_top3.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
    f.write("\\n")
print("wrote", out_path, f"({len(out['questions'])} entries = 100 questions x 2 methods)")
print(json.dumps(out["questions"][0], ensure_ascii=False)[:220], "…")"""))

cells.append(md("""\
## DONE WHEN

- [x] index built over 4,278 chunks — chunk count, vocab, avgdl, build-ms printed above
- [x] 100 questions × 2 methods searched by the **Python** twin, results written to
      `results/retrieval_top3.json` (format `p1-retrieval-eval-v1`)
- [ ] **agreement > 95%** — run `node scripts/test_task8_retrieval.js` (it loads this exact
      JSON and recomputes everything with the **JS** twin; expected: 100%/100%)
- [ ] **demo page works** — open `docs/retrieve_test.html` (serve `docs/` or open via file://;
      the page falls back to `docs/data/corpus_docs.js` when fetch is blocked)

Then tick Task 8 in the progress tracker. Both demos work AND agree — that's the bar."""))

nb.cells = cells

# attach ids (nbformat 4.5 requires cell ids)
for i, c in enumerate(nb.cells):
    c["id"] = f"cell-{i:02d}"

nbf.validate(nb)
out = "colab/07_retrieval_eval.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "-", len(nb.cells), "cells (nbformat valid)")
