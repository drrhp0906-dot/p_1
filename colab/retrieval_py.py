#!/usr/bin/env python3
# ============================================================================
# p_1 · colab/retrieval_py.py — Task 8, handmade retrieval (Python twin)
# ----------------------------------------------------------------------------
# Inverted-index retrieval over the health Q&A corpus. Pure stdlib (math, re,
# json — NumPy not needed). This is the twin of docs/retrieve.js: BOTH implement
# the SAME algorithm contract, and the acceptance gate is >95% top-3 agreement
# on the 100-question eval (checked by scripts/test_task8_retrieval.js).

# ALGORITHM CONTRACT (identical in docs/retrieve.js — do not deviate)
#
#   Chunking   one chunk per Q&A pair: chunkId = entry id,
#              text = "Q: <question>\nA: <answer>", sources passed through.
#              chunk_doc() also ships (sliding window ~200 tokens, overlap 40,
#              whitespace words) for FUTURE long documents; the corpus path
#              does NOT use it.
#
#   Tokenize   lowercase; tokens = re.findall(r'[a-z0-9]+', text.lower())
#              (JS twin: text.toLowerCase().match(/[a-z0-9]+/g)).
#              NO stopword removal, NO stemming.
#
#   Index      term -> postings [(chunkIdx, tf)] built in chunk order, so each
#              list is sorted by chunkIdx (deterministic: corpus in file
#              order). df = len(postings[term]). N = chunk count.
#              avgdl = mean token count (0-guarded to 1).
#
#   TF-IDF     weight(t,d) = (1 + ln(tf)) * ln(N/df)            (doc AND query)
#              score(q,d)  = dot(q,d) / (|q|*|d|)               (cosine)
#              zero-guard: |q| == 0 or |d| == 0 -> score 0.
#              Convention (both twins): query terms with df = 0 are SKIPPED
#              (undefined idf, matches nothing, would only rescale |q|); terms
#              with df = N contribute exactly 0 (ln 1 = 0).
#
#   BM25       idf(t)     = ln((N - df + 0.5)/(df + 0.5) + 1)   (always > 0)
#              score(q,d) = SUM_t idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*dl(d)/avgdl))
#              k1 = 1.5, b = 0.75. Query terms = UNIQUE terms in
#              first-occurrence order (no query-TF multiplier).
#
#   Ranking    sort by (score DESC, chunkIdx ASC) — deterministic, zero-score
#              ties fall back to corpus order. search() returns top-k as
#              dicts {chunkId, score, rank (1-based), sources, idx}.
#
#   Float parity: both twins are IEEE-754 double and accumulate in the same
#   order (query terms first-occurrence, postings chunkIdx order, norms over
#   each chunk's first-occurrence term map — Python dicts are insertion-
#   ordered exactly like JS objects). Residual libm noise can only flip
#   exact near-ties; the gate tolerates that.
#
# CLI
#   python3 colab/retrieval_py.py                 # demo searches + index stats
#   python3 colab/retrieval_py.py --eval          # 100-question eval ->
#                                                 # results/retrieval_top3.json
#                                                 # + stats + sanity metrics
#   python3 colab/retrieval_py.py --query "..." [--k 5] [--method bm25|tfidf]
#
# Data dir (mirrors corpus_lib.data_dir()):
#   $P1_DATA_DIR  >  /content/drive/MyDrive/p_1/corpus (if mounted)
#                >  <repo_root>/corpus
# ============================================================================

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

VERSION = "1.0.0"
FORMAT = "p1-retrieval-eval-v1"

BM25_K1 = 1.5
BM25_B = 0.75
WINDOW_TOKENS = 200      # generic long-doc chunker (NOT used by the corpus path)
WINDOW_OVERLAP = 40

TOKEN_RE = re.compile(r"[a-z0-9]+")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


# ----------------------------------------------------------------- tokenize --
def tokenize(text: str) -> list:
    """THE shared tokenizer. Both twins: lowercase, [a-z0-9]+, no stopwords."""
    return TOKEN_RE.findall(str(text).lower())


# ----------------------------------------------------------------- chunking --
def chunk(pairs: list) -> list:
    """Corpus path: one chunk per Q&A pair. chunkId = entry id."""
    out = []
    for p in pairs:
        out.append({
            "chunkId": p["id"],
            "text": "Q: " + p["question"] + "\nA: " + p["answer"],
            "sources": p.get("sources") or [],
            "idx": len(out),
        })
    return out


def chunk_doc(text: str, size_tokens: int = WINDOW_TOKENS,
              overlap: int = WINDOW_OVERLAP, id: str = "doc",
              sources=None) -> list:
    """Generic sliding-window chunker for FUTURE long documents: windows of
    ~size_tokens whitespace words, stepping by (size - overlap). Text is
    whitespace-normalized on rebuild. Not used by the corpus path."""
    words = str(text).split()
    out = []
    if not words:
        return out
    step = max(1, size_tokens - overlap)
    start = 0
    while start < len(words):
        win = words[start:start + size_tokens]
        out.append({
            "chunkId": "%s#w%04d" % (id, len(out)),
            "text": " ".join(win),
            "sources": sources or [],
            "idx": len(out),
        })
        if start + size_tokens >= len(words):
            break
        start += step
    return out


# ------------------------------------------------------------------ indexing --
def build_index(items: list, keep_doc_tf: bool = False) -> dict:
    """Accepts pairs [{id, question, answer, sources?}] (auto 1-chunk-per-pair)
    or ready chunks [{chunkId, text, sources?}]. Deterministic: file order in,
    chunkIdx order out."""
    t0 = time.perf_counter()

    if items and isinstance(items[0].get("question"), str):
        chunks = chunk(items)                       # corpus path: 1 chunk/pair
    else:
        chunks = [{"chunkId": c.get("chunkId"), "text": c["text"],
                   "sources": c.get("sources") or [], "idx": i}
                  for i, c in enumerate(items)]

    N = len(chunks)
    dl = [0] * N
    doc_tf = []                                     # per-chunk term->tf, insertion-ordered
    postings = {}                                   # term -> [(chunkIdx, tf), ...]

    for i in range(N):
        toks = tokenize(chunks[i]["text"])
        dl[i] = len(toks)
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        doc_tf.append(tf)
        for t, f in tf.items():
            postings.setdefault(t, []).append((i, f))

    total = sum(dl)
    avgdl = (total / N) if N > 0 else 0.0
    avgdl_guard = avgdl if avgdl > 0 else 1.0
    vocab = len(postings)

    # TF-IDF doc norms, precomputed once df is final. Sum order = each chunk's
    # first-occurrence term order (matches the JS twin).
    doc_norm = [0.0] * N
    for i in range(N):
        s = 0.0
        for t, f in doc_tf[i].items():
            w = (1.0 + math.log(f)) * math.log(N / len(postings[t]))
            s += w * w
        doc_norm[i] = math.sqrt(s)
    if not keep_doc_tf:
        doc_tf = [None] * N        # query path reads postings, not these

    build_ms = (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------- scoring --
    def score_bm25(tokens: list, scores: list) -> None:
        seen = set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            lst = postings.get(t)
            if not lst:
                continue
            df = len(lst)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for idx, tf in lst:
                denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl[idx] / avgdl_guard)
                scores[idx] += idf * (tf * (BM25_K1 + 1.0)) / denom

    def score_tfidf(tokens: list, scores: list) -> None:
        qtf = {}
        q_terms = []                                # first-occurrence order
        for t in tokens:
            if t not in qtf:
                qtf[t] = 0
                q_terms.append(t)
            qtf[t] += 1
        if not q_terms:
            return                                  # empty query -> all 0
        qn2 = 0.0
        qw_arr = []                                 # (term, weight) first-occurrence order
        for t in q_terms:
            lst = postings.get(t)
            if not lst:
                continue                            # df = 0 -> skipped (see contract)
            w = (1.0 + math.log(qtf[t])) * math.log(N / len(lst))
            qw_arr.append((t, w))
            qn2 += w * w
        qnorm = math.sqrt(qn2)
        if qnorm == 0.0:
            return                                  # zero-guard: |q| == 0 -> all 0
        for t, qw in qw_arr:
            lst = postings[t]
            idf = math.log(N / len(lst))
            if idf == 0.0:
                continue                            # df == N -> weight 0 everywhere
            for idx, tf in lst:
                scores[idx] += qw * (1.0 + math.log(tf)) * idf
        for i in range(N):                          # cosine, zero-guarded
            if scores[i] != 0.0 and doc_norm[i] > 0.0:
                scores[i] /= qnorm * doc_norm[i]

    # -------------------------------------------------------------- search --
    def search(query: str, k: int = 5, method: str = "bm25") -> list:
        tokens = tokenize(query)
        scores = [0.0] * N
        if method == "tfidf":
            score_tfidf(tokens, scores)
        else:
            score_bm25(tokens, scores)              # default fallback: bm25

        # Deterministic ranking: (score DESC, chunkIdx ASC).
        order = sorted(range(N), key=lambda i: (-scores[i], i))

        out = []
        for r, idx in enumerate(order[: max(0, min(k, N))]):
            out.append({
                "chunkId": chunks[idx]["chunkId"],
                "score": scores[idx],
                "rank": r + 1,
                "sources": chunks[idx]["sources"],
                "idx": idx,
            })
        return out

    return {
        "search": search,
        "N": N,
        "vocab": vocab,
        "avgdl": avgdl,
        "chunks": chunks,
        "dl": dl,
        "df": {t: len(lst) for t, lst in postings.items()},
        "doc_norm": doc_norm,
        "k1": BM25_K1,
        "b": BM25_B,
        "build_ms": build_ms,
        "version": VERSION,
    }


# --------------------------------------------------------------------- data --
def data_dir() -> Path:
    """$P1_DATA_DIR > mounted Colab Drive > local repo ./corpus."""
    env = os.environ.get("P1_DATA_DIR", "").strip()
    if env:
        return Path(env)
    drive = Path("/content/drive/MyDrive/p_1/corpus")
    if drive.exists():
        return drive
    return REPO_ROOT / "corpus"


def results_dir() -> Path:
    env = os.environ.get("P1_RESULTS_DIR", "").strip()
    if env:
        p = Path(env)
    else:
        p = REPO_ROOT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


EVAL_FILES = [  # fixed order = exactly 100 questions
    "eval_answerable.jsonl",
    "eval_mythtraps.jsonl",
    "eval_unanswerable.jsonl",
]


# ---------------------------------------------------------------- self-test --
def selftest_exact_question(index: dict, corpus: list, n_samples: int = 10) -> dict:
    """Searching an exact corpus question must return its own chunk at rank 1.
    Deterministic sample: 10 entries evenly spaced across the corpus.

    Corpus caveat (measured, both twins implement the same rule): every corpus
    question text appears in ~3-6 sibling chunks (the Task 2 expansion emits one
    pair per question-phrasing x answer-variant — 1,239 unique questions over
    4,278 chunks). A sibling shares the identical question and the same seed
    fact, only the answer variant differs, so top-1 == entry OR top-1's
    question text == the query counts as a hit. Strict own-id hits are reported
    alongside so nothing is hidden."""
    N = len(corpus)
    idxs = [round(i * (N - 1) / (n_samples - 1)) for i in range(n_samples)]
    hits, strict, fails = 0, 0, []
    for ci in idxs:
        entry = corpus[ci]
        top = index["search"](entry["question"], k=1, method="bm25")
        top_id = top[0]["chunkId"] if top else None
        if top_id == entry["id"]:
            hits += 1
            strict += 1
        elif top_id is not None and corpus[top[0]["idx"]]["question"] == entry["question"]:
            hits += 1    # sibling variant: identical question text, same fact
        else:
            fails.append((entry["id"], top_id))
    return {"sampled": n_samples, "hits": hits, "strict_own_id": strict,
            "fails": fails, "idxs": idxs}


def own_id_sanity(corpus: list, answerable: list) -> dict:
    """Sanity metric on the answerable set, per the Task 8 spec: append the 60
    answerable eval entries to the corpus as chunks and measure whether
    searching the eval QUESTION ranks the entry's OWN chunk (its gold answer,
    adapted from the same seed) in the top-3 — against ~4.3k near-duplicate
    corpus chunks that paraphrase the same facts. Reported honestly: the
    corpus itself carries no seed provenance (eval answers are rewrites, 0/60
    verbatim matches), so 'hit-rate vs the entry's own id' needs the entry
    appended; near-duplicates make this a non-trivial, conservative signal."""
    out = {}
    for method in ("bm25", "tfidf"):
        idx = build_index(list(corpus) + [
            {"id": e["id"], "question": e["question"], "answer": e["answer"],
             "sources": e.get("sources") or []} for e in answerable])
        rr, h1, h3 = 0.0, 0, 0
        for e in answerable:
            top3 = idx["search"](e["question"], k=3, method=method)
            rank = next((r["rank"] for r in top3 if r["chunkId"] == e["id"]), None)
            if rank is not None:
                rr += 1.0 / rank
                h3 += 1
                if rank == 1:
                    h1 += 1
        n = len(answerable)
        out[method] = {"mrr_at3": rr / n, "hit_at1": h1 / n, "hit_at3": h3 / n, "n": n}
    return out


# --------------------------------------------------------------------- eval --
def run_eval() -> dict:
    cdir = data_dir()
    print("retrieval_py v%s — eval (format %s)" % (VERSION, FORMAT))
    print("data dir   :", cdir)

    corpus = load_jsonl(cdir / "health_qa.jsonl")
    print("corpus     : %d pairs (%s)" % (len(corpus), "health_qa.jsonl"))

    t0 = time.perf_counter()
    index = build_index(corpus)
    build_ms = (time.perf_counter() - t0) * 1000.0
    print("index      : N=%d  vocab=%d  avgdl=%.3f  build=%.1f ms"
          % (index["N"], index["vocab"], index["avgdl"], build_ms))

    questions = []
    for fname in EVAL_FILES:
        rows = load_jsonl(cdir / fname)
        print("eval set   : %-26s %d questions" % (fname, len(rows)))
        for r in rows:
            questions.append({"id": r["id"], "question": r["question"]})
    assert len(questions) == 100, "expected exactly 100 eval questions, got %d" % len(questions)

    # the gate runs: top-3 for BOTH methods on all 100 questions
    t0 = time.perf_counter()
    out_questions = []
    for method in ("bm25", "tfidf"):
        for q in questions:
            top3 = index["search"](q["question"], k=3, method=method)
            out_questions.append({
                "id": q["id"],
                "question": q["question"],
                "method": method,
                "top3": [r["chunkId"] for r in top3],
                "scores": [r["score"] for r in top3],
            })
    search_total_ms = (time.perf_counter() - t0) * 1000.0
    per_query_ms = search_total_ms / len(out_questions)

    # sanity metrics (printed, not part of the gate JSON)
    answerable = load_jsonl(cdir / "eval_answerable.jsonl")
    st = selftest_exact_question(index, corpus, 10)
    print("self-test  : exact corpus question -> own fact rank 1 : %d/%d (strict own-id %d/%d; siblings share the question text)"
          % (st["hits"], st["sampled"], st["strict_own_id"], st["sampled"]))
    if st["fails"]:
        print("             FAILS: " + ", ".join("%s->%s" % f for f in st["fails"]))
    own = own_id_sanity(corpus, answerable)
    for m, v in own.items():
        print("own-id %-5s: MRR@3 %.3f · hit@1 %.3f · hit@3 %.3f  (n=%d, own chunk vs %d near-dup corpus chunks)"
              % (m, v["mrr_at3"], v["hit_at1"], v["hit_at3"], v["n"], index["N"]))

    out_path = results_dir() / "retrieval_top3.json"
    payload = {
        "format": FORMAT,
        "meta": {
            "corpus": "health_qa.jsonl",
            "n_chunks": index["N"],
            "vocab": index["vocab"],
            "avgdl": round(index["avgdl"], 6),
            "k": 3,
            "methods": ["bm25", "tfidf"],
            "n_questions": len(questions),
            "eval_files": list(EVAL_FILES),
            "retrieval_py_version": VERSION,
        },
        "questions": out_questions,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print("search     : %.2f ms/query avg over %d searches (both methods)"
          % (per_query_ms, len(out_questions)))
    print("wrote      : %s  (%d question entries = 100 questions x 2 methods)"
          % (out_path, len(out_questions)))
    return {"index": index, "build_ms": build_ms, "per_query_ms": per_query_ms,
            "selftest": st, "own_id": own, "out_path": out_path}


DEMO_QUERIES = [
    "How much water should I drink every day?",
    "Does cracking your knuckles cause arthritis?",
    "How do I cool a burn?",
]


def run_demo() -> None:
    cdir = data_dir()
    corpus = load_jsonl(cdir / "health_qa.jsonl")
    index = build_index(corpus)
    print("retrieval_py v%s — demo" % VERSION)
    print("index      : N=%d  vocab=%d  avgdl=%.3f  build=%.1f ms"
          % (index["N"], index["vocab"], index["avgdl"], index["build_ms"]))
    for q in DEMO_QUERIES:
        print("\nQ: %s" % q)
        for method in ("bm25", "tfidf"):
            top3 = index["search"](q, k=3, method=method)
            print("  %-5s %s" % (method, " | ".join(
                "%s %.4f" % (r["chunkId"], r["score"]) for r in top3)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="p_1 Task 8 handmade retrieval (Python twin)")
    ap.add_argument("--eval", action="store_true",
                    help="run the 100-question eval, write results/retrieval_top3.json")
    ap.add_argument("--query", type=str, default=None, help="ad-hoc search")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", type=str, default="bm25", choices=["bm25", "tfidf"])
    args = ap.parse_args(argv)

    if args.eval:
        run_eval()
    elif args.query:
        cdir = data_dir()
        corpus = load_jsonl(cdir / "health_qa.jsonl")
        index = build_index(corpus)
        for r in index["search"](args.query, k=args.k, method=args.method):
            print("%2d. %-9s %.4f  %s" % (r["rank"], r["chunkId"], r["score"],
                                          r["sources"][0][:60] if r["sources"] else ""))
    else:
        run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
