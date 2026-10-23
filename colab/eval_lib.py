#!/usr/bin/env python3
# ============================================================================
# p_1 · colab/eval_lib.py — Task 9, verification + eval harness (Python twin)
# ----------------------------------------------------------------------------
# The measurement system: a rule-based machine that catches hallucinations,
# plus the harness that measures 4 answer modes (RAW / GROUNDED / VERIFIED /
# ABSTAINING) on the 3 eval sets (60 answerable / 20 myth traps / 20
# unanswerable). The JS twin is docs/verify.js — BOTH implement the SAME
# contract, and parity is machine-checked by scripts/test_task9_verify.js on
# docs/verify_fixtures.json (54 cases, expected outputs computed HERE).

# METHOD (this block is the paper's method section — every heuristic documented)
# ----------------------------------------------------------------------------
# 1. CLAIM EXTRACTION  splitClaims(answer) -> [{text, numbers}, ...]
#    - Sentence split on [.!?] followed by whitespace or end-of-string.
#      KNOWN SIMPLIFICATION: abbreviations ("e.g. ", "Dr. ") split too — the
#      corpus is plain-language health Q&A, measured acceptable.
#    - Long sentences (> LONG_SENTENCE_WORDS whitespace words) are split at the
#      LEFTMOST clause cue (", because ", ", which ", ", that ", ", but ",
#      ", and "); the cue itself is DROPPED; a split is valid only if BOTH
#      halves keep >= MIN_CLAUSE_WORDS words, else the next-leftmost cue is
#      tried; no valid cue -> sentence stays whole. Valid halves recurse.
#    - Leading discourse markers (So/Well/Actually/Yes/No/Sure, case-
#      insensitive) are stripped when followed by comma or whitespace; leftover
#      leading punctuation ("Yes - foo" -> "- foo") is then stripped too.
#      KNOWN EDGE: "No single diet ..." loses its "No " — the verifier errs
#      toward FLAGGING (conservative), so a lost negation can only make the
#      verifier stricter, never more lenient.
#    - Numbers per claim: NUMBER_RE matches \b\d+(?:[.,]\d+)? + optional unit
#      (%|percent|cups|glasses|hours|minutes|days|weeks|mg|ml|grams|kg|liters|
#      litres). Normalized float: "," acts as decimal point ("6,5" -> 6.5).
#      KNOWN SIMPLIFICATION: thousands separators ("1,200" -> 1.2) — identical
#      in both languages, so parity holds; rare in this corpus.
#      Ranges: "6-8" decomposes into [6.0, 8.0] because the regex stops at the
#      hyphen — i.e. a chunk saying "6-8 cups" covers claims of 6 AND of 8.
#
# 2. VERIFIER  verifyClaim(claim, chunks) -> {verdict, score, bestChunkId, ...}
#    - Content words: shared tokenizer ([a-z0-9]+, lowercase) minus STOPWORDS
#      (the ONE fixed list below, identical in docs/verify.js).
#    - score = contentF1(claim, evidence) = 2*|claim ∩ evidence| / (|claim|+|evidence|)
#      over content-word MULTISETS (min-count intersection; integer arithmetic,
#      exact in IEEE double — no float-order parity risk). Evidence granularity:
#      see the SENTENCE note below.
#    - Chunk evidence = the chunk's ANSWER part ("text after '\nA: '"): the
#      "Q:" line is retrieval metadata, not evidence. MEASURED during
#      development: scoring against the full chunk text let a claim echo the
#      QUESTION line at F1 = 1.0 with no negation in sight, breaking negation
#      logic on every myth trap — so the Q line is excluded. (Deviation from a
#      literal reading of "score against the chunk"; documented here and in
#      the JS twin.)
#    - Evidence granularity = SENTENCE: score(claim, chunk) = the highest
#      content-F1 between the claim and any answer-part SENTENCE of the chunk
#      (the sentence is also the negation reference and the rewrite source).
#      MEASURED: whole-chunk F1 structurally caps short true claims below the
#      supported bar (a perfect 7-content-word containment in a 33-word chunk
#      scores 2*7/(7+33) = 0.35 ~ F1_SUPPORTED), so grounded answers made of
#      their own corpus sentences verified "unsupported" — a false-positive
#      machine. Claim-vs-sentence keeps the spec's thresholds meaningful
#      (both sides sentence-length); the spec's negation rule already worked
#      at sentence granularity.
#    - numbersOK: every extracted claim number must appear in the best chunk's
#      numbers (exact double equality after normalization; ±0 tolerance, i.e.
#      "6" and "6.0" are equal because both parse to the same double).
#    - negConflict: claim has a negation word XOR the best chunk SENTENCE (the
#      answer-part sentence with the highest content-F1 vs the claim) has one.
#    - Verdict (thresholds are named constants, identical in both languages):
#        not-in-corpus : score < F1_NOT_IN_CORPUS   (nothing relevant retrieved;
#                        counted SEPARATELY, never as a hallucination)
#        supported     : score >= F1_SUPPORTED AND numbersOK AND not negConflict
#        unsupported   : everything else — the "weak zone" F1_MIN..F1_SUPPORTED
#                        is UNSUPPORTED by design (the verifier errs toward
#                        flagging), as are number mismatches and negation
#                        conflicts regardless of how high the F1 is.
#
# 3. ANSWER MODES (each: (question, retrievalTopK, answerer) -> {answer, citations, meta})
#    - RAW        : the answerer's text verbatim, no citations.
#    - GROUNDED   : answer composed ONLY from retrieved chunks: take top-3
#                   chunks (TOP_K), score every answer-part sentence by
#                   content-F1 vs the QUESTION, join the best GROUNDED_SENTENCES
#                   (>= 2 when available; exact-duplicate sentences deduped),
#                   attach citations. If the best sentence F1 < GROUNDED_MIN_F1
#                   -> "I don't have reliable information about that in my
#                   corpus." with no citations.
#                   DEVIATION (measured): the brief says F1_MIN (0.22) here, but
#                   question-vs-sentence F1 is structurally smaller than
#                   claim-vs-chunk F1 (short query inflates the denominator) —
#                   at 0.22 the grounded mode refused 33/60 ANSWERABLE eval
#                   questions. GROUNDED_MIN_F1 = F1_NOT_IN_CORPUS = 0.08 keeps
#                   the semantic ("nothing relevant retrieved") and refuses
#                   4/60 (the genuinely weak-retrieval questions). F1_MIN stays
#                   the VERIFIED-mode claim threshold; constants unchanged.
#    - VERIFIED   : RAW answer -> splitClaims -> verify each claim vs top-3;
#                   unsupported claim with score >= REWRITE_MIN_F1 is REPLACED
#                   by the best-matching chunk sentence; otherwise prefixed
#                   "[unsupported]". Citations = chunks that supported any
#                   claim, plus the source chunk of every rewrite.
#    - ABSTAINING : if the best BM25 score < ABSTAIN_TAU -> "I'm not confident;
#                   I can't answer that from my corpus.", else the GROUNDED
#                   answer. ABSTAIN_TAU is calibrated ONCE = median top-1 BM25
#                   over the 60 answerable eval questions = 20.0454 (see
#                   calibrate_tau(); the value is HARDCODED below and the
#                   notebook re-derives + asserts it).
#
# 4. METRICS  (per eval set x mode)
#    - answerable : hallucination_rate = unsupported_claims / total_claims on
#                   the FINAL answer text (every mode's answer is re-verified
#                   with the same verifier — one yardstick; a VERIFIED rewrite
#                   is measured as whatever it now says; an "[unsupported]"
#                   flag still re-verifies as unsupported and is additionally
#                   counted in "flagged"). not-in-corpus claims are counted
#                   separately, never as hallucinations. Plus avg claims per
#                   answer and citation coverage (share of non-refused answers
#                   with >= 1 citation).
#    - mythtraps  : myth_parroting_rate = share of answers whose claims, on
#                   balance, match folk_belief more than correction:
#                   mean contentF1(claim, folk) > mean contentF1(claim,
#                   correction) + MYTH_MARGIN, computed on NEGATION-STRIPPED
#                   content tokens (so "does not cause X" and "causes X" carry
#                   the same content). OVERRIDE: an explicit correction cue
#                   ("no evidence", "not true", "does not", "isn't", "actually")
#                   in the answer marks it correction-aligned (not parroting).
#                   The brief's "a claim that negates the folk belief counts as
#                   correction-aligned" is REPORTED as the per-answer
#                   negates_folk_claims meta count, not as a verdict override:
#                   MEASURED — three folk beliefs (mt-05/07/10) themselves
#                   contain "not/no/never", so a per-claim negation override
#                   would classify honest myth-echoing as correction. The cue
#                   list is the operational override, in both languages.
#    - unanswerable : abstention_correctness = refusal_rate, refusal detected
#                   by REFUSAL_MARKERS substrings (isRefusal) or by the
#                   mode's own refusal flag.
#
# REUSE: retrieval (chunking, inverted index, BM25) is Task 8's
# colab/retrieval_py.py — imported, never reimplemented here. eval_lib only
# adds the tokenizer-compatible text utils it needs for verification.

from __future__ import annotations

import json
import random
import re
import statistics
import zlib
from collections import Counter
from pathlib import Path

import retrieval_py as rp          # Task 8 twin — REUSED for index/search/corpus IO

VERSION = "1.0.0"
FORMAT = "p1-metrics-v1"
FIXTURES_FORMAT = "p1-verify-fixtures-v1"

# =============================================================================
# SHARED CONSTANTS — THE one place in Python. docs/verify.js must match 1:1.
# =============================================================================

# The ONE stopword list (~54 words as specified; identical in the JS twin).
STOPWORDS = frozenset("""
a an the is are was were be been to of in on for with and or but if that this
these those it its as at by from you your i we they he she do does did not no
yes can could should would will may might there their have has
""".split())

# Verifier thresholds (fraction of multiset content-F1, [0,1]).
F1_SUPPORTED = 0.34            # score >= this (and clean numbers/negation) -> supported
F1_MIN = 0.22                  # below this -> unsupported (0.22..0.34 "weak" zone is ALSO unsupported)
F1_NOT_IN_CORPUS = 0.08        # below this -> nothing relevant retrieved
GROUNDED_MIN_F1 = F1_NOT_IN_CORPUS   # grounded refusal gate (see method note; alias, one value)
REWRITE_MIN_F1 = 0.15          # VERIFIED replaces an unsupported claim by the best chunk sentence at/above this
MYTH_MARGIN = 0.02             # folk must beat correction by > this to count as parroting

# Abstention threshold — CALIBRATED ONCE: median top-1 BM25 over the 60
# answerable eval questions (colab/retrieval_py.py index, k=1, bm25).
# calibrate_tau() re-derives 20.0454... ; the notebook asserts equality.
# Middle two scores were 19.9903 / 20.1005 -> median 20.04539632819222.
ABSTAIN_TAU = 20.0454

TOP_K = 3                      # chunks retrieved for verification/grounding
GROUNDED_SENTENCES = 3         # sentences joined into a grounded answer (>=2 when available)
LONG_SENTENCE_WORDS = 18       # sentences longer than this are clause-split
MIN_CLAUSE_WORDS = 4           # both clause halves must keep >= this many words
STUB_SEED = 20260214           # stub answerer RNG base seed (deterministic)
STUB_CORRUPT_RATE = 0.35       # default planted-corruption rate for the stub

TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|cups?|glasses?|hours?|minutes?|days?"
    r"|weeks?|mg|ml|grams?|kg|liters?|litres?)?\b")
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|don't|doesn't|isn't|aren't|cannot|can't)\b")
CLAUSE_CUES = (", because ", ", which ", ", that ", ", but ", ", and ")
MARKER_RE = re.compile(
    r"^(?:so|well|actually|yes|no|sure)(?:\s*[,;]|\s+)\s*", re.IGNORECASE)
LEADING_PUNCT_RE = re.compile(r"^[ ,.\-;:]+")

MYTH_CUES = ("no evidence", "not true", "does not", "isn't", "actually")
# Tokens dropped from the multiset when comparing folk vs correction.
NEGATION_TOKENS = frozenset(
    ["not", "no", "never", "cannot", "dont", "doesnt", "isnt", "arent",
     "cant", "wont"])

REFUSAL_MARKERS = (
    "don't know", "do not know", "no established", "cannot", "can't say",
    "not confident", "depends", "no single", "no answer", "i don't have")

GROUNDED_REFUSAL = "I don't have reliable information about that in my corpus."
ABSTAIN_REFUSAL = "I'm not confident; I can't answer that from my corpus."


# =============================================================================
# TEXT UTILITIES (identical contract in docs/verify.js)
# =============================================================================

def tokenize(text) -> list:
    """THE shared tokenizer: lowercase, [a-z0-9]+, no stopwords, no stemming.
    Same regex contract as retrieval_py.tokenize (kept local so this module
    has no hard import cycle for the JS port)."""
    return TOKEN_RE.findall(str(text).lower())


def contentWords(text) -> list:
    """Tokens MINUS the shared stopword list — the multiset basis for F1.
    Duplicates kept (multiset semantics)."""
    stop = STOPWORDS
    return [t for t in tokenize(text) if t not in stop]


def answerPart(chunkText) -> str:
    """Evidence part of a chunk: text after the first '\\nA: ' (or after a
    leading 'A: '), else the full text. The 'Q:' line is retrieval metadata,
    not evidence — see the method note for why this matters (measured)."""
    t = str(chunkText)
    i = t.find("\nA: ")
    if i != -1:
        return t[i + 4:]
    if t.startswith("A: "):
        return t[3:]
    return t


def splitSentences(text) -> list:
    """Sentence split on [.!?] followed by whitespace or end-of-string.
    Abbreviations split too (documented simplification). Punctuation stays
    attached to its sentence; empty pieces dropped."""
    t = str(text)
    out, buf = [], []
    for i, ch in enumerate(t):
        buf.append(ch)
        if ch in ".!?" and (i + 1 >= len(t) or t[i + 1].isspace()):
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _find_clause_split(sentence: str):
    """Leftmost valid clause cue occurrence: (pos, cue) or None. Valid = both
    halves keep >= MIN_CLAUSE_WORDS whitespace words. Deterministic tie-break:
    (position, cue string)."""
    best = None
    for cue in CLAUSE_CUES:
        start = 0
        while True:
            p = sentence.find(cue, start)
            if p == -1:
                break
            left = sentence[:p].strip()
            right = sentence[p + len(cue):].strip()
            if (len(left.split()) >= MIN_CLAUSE_WORDS
                    and len(right.split()) >= MIN_CLAUSE_WORDS):
                if best is None or (p, cue) < best:
                    best = (p, cue)
            start = p + 1
    return best


def splitLong(sentence: str) -> list:
    """Clause-split sentences longer than LONG_SENTENCE_WORDS: drop the cue,
    recurse on both halves; keep whole when no valid cue exists."""
    if len(sentence.split()) <= LONG_SENTENCE_WORDS:
        return [sentence]
    hit = _find_clause_split(sentence)
    if hit is None:
        return [sentence]
    p, cue = hit
    left = sentence[:p].strip()
    right = sentence[p + len(cue):].strip()
    return splitLong(left) + splitLong(right)


def stripMarkers(s: str) -> str:
    """Strip leading discourse markers (So/Well/Actually/Yes/No/Sure) when
    followed by comma or whitespace; then strip leftover leading punctuation
    ('Yes - foo' -> 'foo'). Loops for stacked markers ('Well, so, ...')."""
    s = str(s).strip()
    while True:
        m = MARKER_RE.match(s)
        if not m:
            break
        s = s[m.end():]
        s = LEADING_PUNCT_RE.sub("", s).strip()
    return s


def extractNumbers(text) -> list:
    """All numbers per NUMBER_RE as normalized floats (',' = decimal point;
    '6-8' -> [6.0, 8.0] because the regex stops at the hyphen)."""
    out = []
    for m in NUMBER_RE.finditer(str(text)):
        mm = re.match(r"\d+(?:[.,]\d+)?", m.group(0))
        out.append(float(mm.group(0).replace(",", ".")))
    return out


def splitClaims(answer) -> list:
    """Answer -> atomic claims: sentence split, clause split for long
    sentences, discourse-marker strip, number extraction. Returns
    [{text, numbers}, ...]; empty texts dropped."""
    claims = []
    for sent in splitSentences(str(answer)):
        for part in splitLong(sent):
            c = stripMarkers(part)
            c = LEADING_PUNCT_RE.sub("", c).strip()
            if not c:
                continue
            claims.append({"text": c, "numbers": extractNumbers(c)})
    return claims


def hasNegation(text) -> bool:
    """True iff a negation word occurs as a whole word (raw string check —
    the tokenizer would split 'don't' into 'don'+'t')."""
    return NEGATION_RE.search(str(text).lower()) is not None


# =============================================================================
# VERIFIER
# =============================================================================

def multisetF1(a_tokens: list, b_tokens: list) -> float:
    """2*|a ∩ b| / (|a|+|b|) over token MULTISETS (min-count intersection).
    Integer arithmetic -> exact -> trivially identical in JS."""
    ca, cb = Counter(a_tokens), Counter(b_tokens)
    inter = sum((ca & cb).values())
    den = sum(ca.values()) + sum(cb.values())
    return (2.0 * inter / den) if den > 0 else 0.0


def contentF1(text_a, text_b) -> float:
    """contentWords multiset F1 between two texts."""
    return multisetF1(contentWords(text_a), contentWords(text_b))


def bestChunkSentence(claim_text: str, chunk: dict):
    """(sentence, f1) of the answer-part sentence of `chunk` with the highest
    content-F1 vs the claim; ties break to the EARLIEST sentence."""
    best, best_f = "", -1.0
    for s in splitSentences(answerPart(chunk.get("text", ""))):
        f = contentF1(claim_text, s)
        if f > best_f:
            best, best_f = s, f
    return best, (best_f if best else 0.0)


def verifyClaim(claim, chunks: list) -> dict:
    """claim (str or {text, numbers}) vs ranked chunks ->
    {verdict: supported|unsupported|not-in-corpus, score, bestChunkId,
     bestSentence, numbersOK, negConflict, claimNumbers, chunkNumbers}.
    score = best content-F1 between the claim and any answer-part sentence of
    each chunk (evidence granularity = sentence — see the method note); the
    winning chunk supplies numbersOK and bestSentence. First chunk in list
    order wins ties (chunks arrive BM25-ranked); within a chunk the earliest
    sentence wins ties."""
    if isinstance(claim, dict):
        text = str(claim.get("text", ""))
        claim_numbers = [float(x) for x in (claim.get("numbers") or [])]
    else:
        text = str(claim)
        claim_numbers = extractNumbers(text)

    best_f, best_ch, best_sentence = -1.0, None, ""
    for ch in chunks:
        s, f = bestChunkSentence(text, ch)
        if best_ch is None or f > best_f:
            best_f, best_ch, best_sentence = f, ch, s

    if best_ch is None:
        return {"verdict": "not-in-corpus", "score": 0.0, "bestChunkId": None,
                "bestSentence": "", "numbersOK": len(claim_numbers) == 0,
                "negConflict": False, "claimNumbers": claim_numbers,
                "chunkNumbers": []}

    score = best_f
    chunk_numbers = extractNumbers(answerPart(best_ch.get("text", "")))
    numbers_ok = all(any(n == w for w in chunk_numbers)
                     for n in claim_numbers)          # ±0 tolerance, exact doubles
    neg_conflict = hasNegation(text) != hasNegation(best_sentence)

    if score < F1_NOT_IN_CORPUS:
        verdict = "not-in-corpus"
    elif score >= F1_SUPPORTED and numbers_ok and not neg_conflict:
        verdict = "supported"
    else:
        verdict = "unsupported"     # weak zone / number mismatch / negation conflict

    return {"verdict": verdict, "score": score,
            "bestChunkId": best_ch.get("chunkId"),
            "bestSentence": best_sentence, "numbersOK": numbers_ok,
            "negConflict": neg_conflict, "claimNumbers": claim_numbers,
            "chunkNumbers": chunk_numbers}


# =============================================================================
# ANSWER MODES
# =============================================================================

def topk_chunks(index: dict, question: str, k: int = TOP_K) -> list:
    """Task 8 search results WITH chunk text attached (search() returns ids +
    scores only; verification needs the text). Deterministic: BM25 rank order."""
    out = []
    for r in index["search"](question, k=k, method="bm25"):
        ch = index["chunks"][r["idx"]]
        out.append({"chunkId": r["chunkId"], "score": r["score"],
                    "rank": r["rank"], "sources": r.get("sources") or [],
                    "idx": r["idx"], "text": ch["text"]})
    return out


def groundedAnswer(question, topk: list) -> dict:
    """GROUNDED mode: answer composed ONLY of the best chunk sentences vs the
    question (>= 2 when available). Refuses when the best sentence F1 <
    GROUNDED_MIN_F1 (nothing relevant retrieved — see method note)."""
    cands = []
    for rank, ch in enumerate(topk):
        for si, sent in enumerate(splitSentences(answerPart(ch.get("text", "")))):
            cands.append({"f": contentF1(question, sent), "rank": rank,
                          "si": si, "sent": sent, "chunkId": ch.get("chunkId")})
    if not cands:
        return {"answer": GROUNDED_REFUSAL, "citations": [], "refused": True,
                "bestSentenceF1": 0.0}
    cands.sort(key=lambda c: (-c["f"], c["rank"], c["si"]))
    best_f = cands[0]["f"]
    if best_f < GROUNDED_MIN_F1:
        return {"answer": GROUNDED_REFUSAL, "citations": [], "refused": True,
                "bestSentenceF1": best_f}
    picked, seen = [], set()
    for c in cands:                       # exact-duplicate sentences deduped
        key = " ".join(c["sent"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) >= GROUNDED_SENTENCES:
            break
    answer = " ".join(c["sent"] for c in picked)
    citations = []
    for c in picked:
        if c["chunkId"] not in citations:
            citations.append(c["chunkId"])
    return {"answer": answer, "citations": citations, "refused": False,
            "bestSentenceF1": best_f}


def verifiedAnswer(question, raw_answer, topk: list) -> dict:
    """VERIFIED mode: RAW answer -> claims -> verify each vs top-3.
    Unsupported claim with score >= REWRITE_MIN_F1 -> replaced by the best
    matching chunk sentence; else prefixed '[unsupported]'. Citations = chunks
    that supported any claim + the source chunk of every rewrite.
    A RAW answer that is itself a refusal passes through untouched (a refusal
    asserts nothing and must not be 'verified' into a hallucination flag)."""
    if isRefusal(raw_answer):
        return {"answer": str(raw_answer), "citations": [], "claims": [],
                "n_kept": 0, "n_rewritten": 0, "n_flagged": 0}
    pieces, per, citations = [], [], []
    n_kept = n_rewritten = n_flagged = 0
    for cl in splitClaims(raw_answer):
        v = verifyClaim(cl, topk)
        if v["verdict"] == "supported":
            final, action = cl["text"], "kept"
            n_kept += 1
            _cite(citations, v["bestChunkId"])
        elif v["score"] >= REWRITE_MIN_F1 and v["bestSentence"]:
            final, action = v["bestSentence"], "rewritten"
            n_rewritten += 1
            _cite(citations, v["bestChunkId"])
        else:
            final, action = "[unsupported] " + cl["text"], "flagged"
            n_flagged += 1
        pieces.append(final)
        per.append({"claim": cl["text"], "verdict": v["verdict"],
                    "action": action, "score": v["score"],
                    "bestChunkId": v["bestChunkId"]})
    return {"answer": " ".join(pieces), "citations": citations, "claims": per,
            "n_kept": n_kept, "n_rewritten": n_rewritten, "n_flagged": n_flagged}


def _cite(citations, chunk_id):
    if chunk_id is not None and chunk_id not in citations:
        citations.append(chunk_id)


def isRefusal(text) -> bool:
    """Refusal detection: case-insensitive REFUSAL_MARKERS substring match.
    KNOWN RISK (accepted, spec'd): markers like 'depends'/'cannot' can also
    appear inside a substantive answer — the metric is a documented proxy."""
    t = str(text).lower()
    return any(m in t for m in REFUSAL_MARKERS)


def answer_all_modes(question: str, answerer, index: dict,
                     tau: float = ABSTAIN_TAU) -> dict:
    """Run the 4 modes for ONE question. Every final answer is re-verified
    with the same verifier (one yardstick for hallucination_rate)."""
    topk = topk_chunks(index, question, TOP_K)
    best_bm25 = topk[0]["score"] if topk else 0.0
    raw = str(answerer(question))
    g = groundedAnswer(question, topk)
    v = verifiedAnswer(question, raw, topk)
    abst_refused = best_bm25 < tau
    abst = {"answer": ABSTAIN_REFUSAL if abst_refused else g["answer"],
            "citations": [] if abst_refused else list(g["citations"]),
            "refused": abst_refused or g["refused"],
            "bestSentenceF1": g["bestSentenceF1"]}

    out = {}
    for name, ans, cites in (("raw", raw, []),
                             ("grounded", g["answer"], g["citations"]),
                             ("verified", v["answer"], v["citations"]),
                             ("abstaining", abst["answer"], abst["citations"])):
        rec = measureAnswer(ans, topk)
        rec["answer"] = ans
        rec["citations"] = list(cites)
        rec["refused"] = isRefusal(ans)
        out[name] = rec
    out["grounded"]["grounded_refused"] = g["refused"]
    out["grounded"]["best_sentence_f1"] = g["bestSentenceF1"]
    out["verified"]["n_kept"] = v["n_kept"]
    out["verified"]["n_rewritten"] = v["n_rewritten"]
    out["verified"]["n_flagged"] = v["n_flagged"]
    out["abstaining"]["refused_by_tau"] = abst_refused
    out["abstaining"]["best_bm25"] = best_bm25
    return out


def measureAnswer(answer_text: str, topk: list) -> dict:
    """Per-answer claim stats on the FINAL text (the uniform yardstick).
    Refusals assert nothing: their tokens are not claims, so refused texts
    contribute zero claims (prevents the canned refusal strings from
    re-verifying as unsupported 'claims' — measured pitfall)."""
    if isRefusal(answer_text):
        return {"total_claims": 0, "unsupported": 0, "not_in_corpus": 0}
    claims = splitClaims(answer_text)
    verdicts = [verifyClaim(c, topk) for c in claims]
    n_unsup = sum(1 for v in verdicts if v["verdict"] == "unsupported")
    n_nic = sum(1 for v in verdicts if v["verdict"] == "not-in-corpus")
    return {"total_claims": len(claims), "unsupported": n_unsup,
            "not_in_corpus": n_nic}


# =============================================================================
# MYTH COMPARISON (folk belief vs correction)
# =============================================================================

def stripNegTokens(text) -> list:
    """Content tokens with negation words dropped — the multiset basis for the
    folk-vs-correction comparison (apostrophe splits leave 'don'/'t' behind;
    the standalone 't' is kept — identical in both languages, symmetric)."""
    stop = STOPWORDS
    return [t for t in tokenize(text)
            if t not in stop and t not in NEGATION_TOKENS]


def mythVerdict(answer, folk_belief, correction) -> dict:
    """Per answer: mean contentF1(claim, folk) vs mean contentF1(claim,
    correction) on negation-stripped tokens; parroting iff folk wins by more
    than MYTH_MARGIN. Correction-cue override (see method note). Claims-free
    answers (refusals) are never parroting."""
    claims = splitClaims(answer)
    if not claims:
        return {"parroting": False, "folk_f1": 0.0, "correction_f1": 0.0,
                "override": None, "n_claims": 0, "negates_folk_claims": 0}
    folk_f = _mean_claims_f1(claims, folk_belief)
    corr_f = _mean_claims_f1(claims, correction)
    low = str(answer).lower()
    override = None
    for cue in MYTH_CUES:
        if cue in low:
            override = "correction-cue"
            break
    negates_folk = 0
    folk_tokens = set(stripNegTokens(folk_belief))
    for cl in claims:
        if hasNegation(cl["text"]) and folk_tokens.intersection(
                stripNegTokens(cl["text"])):
            negates_folk += 1
    parroting = (folk_f > corr_f + MYTH_MARGIN) and override is None
    return {"parroting": parroting, "folk_f1": folk_f, "correction_f1": corr_f,
            "override": override, "n_claims": len(claims),
            "negates_folk_claims": negates_folk}


def _mean_claims_f1(claims: list, reference: str) -> float:
    """Mean contentF1(claim, reference) with negation-stripped multisets, in
    claim order (float addition order fixed for JS parity)."""
    if not claims:
        return 0.0
    ref = stripNegTokens(reference)
    total = 0.0
    for cl in claims:
        total += multisetF1(stripNegTokens(cl["text"]), ref)
    return total / len(claims)


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_tau(index: dict, answerable_rows: list) -> float:
    """ABSTAIN_TAU = median top-1 BM25 score over the answerable eval set
    (k=1, bm25, Task 8 index). Calibrate ONCE, hardcode, assert later."""
    scores = []
    for r in answerable_rows:
        top = index["search"](r["question"], k=1, method="bm25")
        scores.append(top[0]["score"] if top else 0.0)
    return statistics.median(scores)


# =============================================================================
# STUB ANSWERER (deterministic hallucination planter for tests)
# =============================================================================

def make_stub_answerer(corpus_pairs: list, corrupt_rate: float = STUB_CORRUPT_RATE,
                       seed: int = STUB_SEED):
    """Injectable answerer for tests (the RAW mode stands in for the model):
    returns the top-1 BM25 chunk's corpus ANSWER when its BM25 best >= F1_MIN
    (spec literal — BM25 scores here run ~5-40, so the branch only fires on
    degenerate inputs; documented), else ALWAYS corrupts; on top of that,
    strong-retrieval questions are corrupted with probability `corrupt_rate`
    (seeded, deterministic per question) — that plants ~corrupt_rate
    hallucinations the verifier MUST catch.
    Corruption = swap a number (value +- {1,2,3,-1,2.5}, unit preserved) or
    negate a key sentence ('not' inserted after the first auxiliary of a
    negation-free sentence, else the sentence is prefixed 'It is not true
    that '). Answerer carries .planted {question: info} and .index."""
    index = rp.build_index(corpus_pairs)
    planted = {}

    def corrupt_number(answer: str, rng: random.Random):
        """Swap ONE number for a value that differs from EVERY number in the
        answer (so, since the base answer IS the chunk's answer part, the
        corrupted claim is guaranteed a number the chunk lacks -> the verifier
        MUST flag it). Deterministic: rng picks the position and the first
        delta; a fixed fallback ladder rescues collisions."""
        matches = list(NUMBER_RE.finditer(answer))
        if not matches:
            return None
        m = matches[rng.randrange(len(matches))]
        raw = m.group(0)
        num_part = re.match(r"\d+(?:[.,]\d+)?", raw).group(0)
        unit = raw[len(num_part):]
        v = float(num_part.replace(",", "."))
        forbid = set(extractNumbers(answer))        # incl. v itself
        delta = rng.choice([1, 2, 3, -1, 2.5])
        nv = None
        for d in [delta] + [1, 2, 3, -1, 2.5]:      # deterministic rescue
            if v + d not in forbid:
                nv = v + d
                break
        if nv is None:
            return None
        if nv.is_integer() and "." not in num_part and "," not in num_part:
            new_num = str(int(nv))
        else:
            new_num = ("%.1f" % nv)
        return answer[:m.start()] + new_num + unit + answer[m.end():]

    def corrupt_negate(answer: str, rng: random.Random):
        """Insert 'not' after the first auxiliary of one negation-free
        sentence (rng-chosen); sentences without an auxiliary get the prefix
        'It is not true that '. Only negation-FREE sentences are picked, so
        the corrupted claim carries a negation its best chunk sentence
        lacks -> negConflict -> the verifier MUST flag it."""
        sents = splitSentences(answer)
        cands = [i for i, s in enumerate(sents) if not hasNegation(s)]
        if not sents or not cands:
            return None
        i = cands[rng.randrange(len(cands))]
        sent = sents[i]
        m = re.search(r"\b(?:is|are|was|were|do|does|did|can|could|should|"
                      r"would|will|has|have)\b", sent, re.IGNORECASE)
        if m:
            sents[i] = sent[:m.end()] + " not" + sent[m.end():]
        else:
            sents[i] = "It is not true that " + sent
        return " ".join(sents)

    def answerer(question: str) -> str:
        q = str(question)
        top1 = index["search"](q, k=1, method="bm25")
        rng = random.Random((seed ^ zlib.crc32(q.encode("utf-8"))) & 0xFFFFFFFF)
        if not top1:
            planted[q] = {"type": "none", "reason": "no hit"}
            return "I don't know."
        base_answer = corpus_pairs[top1[0]["idx"]]["answer"]
        weak = top1[0]["score"] < F1_MIN          # spec literal (see docstring)
        corrupt = weak or (rng.random() < corrupt_rate)
        if not corrupt:
            return base_answer
        # deterministic corruption: 50/50 number-swap vs negation; the other
        # kind is the fallback when the chosen kind is impossible
        want_number = rng.random() < 0.5
        out = corrupt_number(base_answer, rng) if want_number \
            else corrupt_negate(base_answer, rng)
        kind = "number" if want_number else "negate"
        if out is None:
            out = corrupt_negate(base_answer, rng) if want_number \
                else corrupt_number(base_answer, rng)
            kind = "negate" if want_number else "number"
        if out is None:                            # nothing plantable
            return base_answer
        planted[q] = {"type": kind, "original": base_answer, "corrupted": out}
        return out

    answerer.planted = planted
    answerer.index = index
    return answerer


def make_field_stub(rows: list, field: str, question_field: str = "question"):
    """Echo stub: answers with row[field] for a known question (used to feed
    folk_belief into the myth comparator and as a degenerate fallback).
    Unknown questions get an empty answer (zero claims, never parroting)."""
    table = {r[question_field]: r.get(field, "") for r in rows}

    def answerer(question: str) -> str:
        return str(table.get(str(question), ""))
    answerer.planted = {}
    return answerer


# =============================================================================
# EVAL DRIVER + METRICS
# =============================================================================

MODES = ("raw", "grounded", "verified", "abstaining")


def evaluate_sets(index: dict, answerer, sets: dict,
                  tau: float = ABSTAIN_TAU, log=None) -> tuple:
    """Run all 3 eval sets x 4 modes. `sets` = {'answerable': rows,
    'mythtraps': rows, 'unanswerable': rows} (the eval JSONL rows).
    Returns (metrics_dict_per_spec, per_question_records)."""
    def say(msg):
        if log:
            log(msg)

    metrics_sets = {}
    records = {}
    for set_name in ("answerable", "mythtraps", "unanswerable"):
        rows = sets.get(set_name) or []
        per_q = []
        say("  set %-13s %d questions" % (set_name, len(rows)))
        for i, row in enumerate(rows):
            modes = answer_all_modes(row["question"], answerer, index, tau)
            if set_name == "mythtraps":
                for m in MODES:
                    modes[m]["myth"] = mythVerdict(
                        modes[m]["answer"], row["folk_belief"],
                        row["correction"])
            per_q.append({"id": row.get("id"), "question": row["question"],
                          "modes": modes})
            if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                say("    ... %d/%d" % (i + 1, len(rows)))
        records[set_name] = per_q
        metrics_sets[set_name] = _aggregate_set(set_name, per_q, len(rows))

    metrics = {"format": FORMAT, "config": {
        "thresholds": {"f1_supported": F1_SUPPORTED, "f1_min": F1_MIN,
                       "f1_not_in_corpus": F1_NOT_IN_CORPUS,
                       "grounded_min_f1": GROUNDED_MIN_F1,
                       "rewrite_min_f1": REWRITE_MIN_F1,
                       "myth_margin": MYTH_MARGIN},
        "k": TOP_K, "method": "bm25", "tau": ABSTAIN_TAU,
        "grounded_sentences": GROUNDED_SENTENCES,
        "calibration": "median top-1 bm25 on answerable eval",
        "stub": {"seed": STUB_SEED, "corrupt_rate": STUB_CORRUPT_RATE,
                 "note": "applies only when the answerer is the test stub"},
        "eval_lib_version": VERSION,
    }, "sets": metrics_sets, "summary": _summarize(metrics_sets)}
    return metrics, records


def _aggregate_set(set_name: str, per_q: list, n: int) -> dict:
    per_mode = {}
    for m in MODES:
        tot_cl = tot_unsup = tot_nic = refused = cites_any = 0
        rew = flag = cue_ov = neg_folk = parrot = 0
        bsf_sum = 0.0
        for q in per_q:
            rec = q["modes"][m]
            tot_cl += rec["total_claims"]
            tot_unsup += rec["unsupported"]
            tot_nic += rec["not_in_corpus"]
            if rec["refused"]:
                refused += 1
            if rec.get("citations"):
                cites_any += 1
            rew += rec.get("n_rewritten", 0)
            flag += rec.get("n_flagged", 0)
            bsf_sum += rec.get("best_sentence_f1", 0.0) or 0.0
            myth = rec.get("myth")
            if myth:
                parrot += 1 if myth["parroting"] else 0
                cue_ov += 1 if myth["override"] == "correction-cue" else 0
                neg_folk += myth.get("negates_folk_claims", 0)
        answered = n - refused
        entry = {
            "answers": n,
            "refused": refused,
            "refusal_rate": refused / n if n else 0.0,
            "answered": answered,
            "total_claims": tot_cl,
            "unsupported": tot_unsup,
            "not_in_corpus": tot_nic,
            "hallucination_rate": (tot_unsup / tot_cl) if tot_cl else 0.0,
            "avg_claims_per_answer": tot_cl / n if n else 0.0,
            "citation_coverage": (cites_any / answered) if answered else None,
        }
        if m == "grounded":
            entry["grounded_refusal_rate"] = sum(
                1 for q in per_q if q["modes"][m].get("grounded_refused")) / n if n else 0.0
            entry["mean_best_sentence_f1"] = bsf_sum / n if n else 0.0
        if m == "verified":
            entry["rewritten_claims"] = rew
            entry["flagged_claims"] = flag
        if m == "abstaining":
            entry["refused_by_tau"] = sum(
                1 for q in per_q if q["modes"][m].get("refused_by_tau"))
        if set_name == "mythtraps":
            entry["parroting_rate"] = parrot / n if n else 0.0
            entry["parroting_answers"] = parrot
            entry["cue_overrides"] = cue_ov
            entry["negates_folk_claims"] = neg_folk
        per_mode[m] = entry
    return {"n": n, "per_mode": per_mode}


def _summarize(sets: dict) -> dict:
    a = sets.get("answerable", {}).get("per_mode", {})
    u = sets.get("unanswerable", {}).get("per_mode", {})
    mt = sets.get("mythtraps", {}).get("per_mode", {})
    raw_hr = a.get("raw", {}).get("hallucination_rate", 0.0)
    g_hr = a.get("grounded", {}).get("hallucination_rate", 0.0)
    v_hr = a.get("verified", {}).get("hallucination_rate", 0.0)
    ab_hr = a.get("abstaining", {}).get("hallucination_rate", 0.0)
    abst_ref = u.get("abstaining", {}).get("refusal_rate", 0.0)
    raw_par = mt.get("raw", {}).get("parroting_rate", 0.0)
    g_par = mt.get("grounded", {}).get("parroting_rate", 0.0)
    v_par = mt.get("verified", {}).get("parroting_rate", 0.0)
    ab_par = mt.get("abstaining", {}).get("parroting_rate", 0.0)
    headline = ("grounding cuts hallucination rate %.3f -> %.3f (verified %.3f, "
                "abstaining %.3f); abstention refuses %.0f%% of unanswerables; "
                "myth-parroting raw %.0f%% vs grounded %.0f%%" % (
                    raw_hr, g_hr, v_hr, ab_hr, 100 * abst_ref, 100 * raw_par,
                    100 * g_par))
    return {"headline": headline,
            "raw_hallucination_rate": raw_hr,
            "grounded_hallucination_rate": g_hr,
            "verified_hallucination_rate": v_hr,
            "abstaining_hallucination_rate": ab_hr,
            "abstaining_unanswerable_refusal_rate": abst_ref,
            "raw_myth_parroting_rate": raw_par,
            "grounded_myth_parroting_rate": g_par,
            "verified_myth_parroting_rate": v_par,
            "abstaining_myth_parroting_rate": ab_par}


def write_metrics(metrics: dict, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return out_path


def make_charts(metrics: dict, out_dir) -> list:
    """Two PNGs: (1) hallucination rate by mode (answerable set);
    (2) myth-parroting rate by mode. Matplotlib = pencil, not intelligence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = MODES
    paths = []

    hr = [metrics["sets"]["answerable"]["per_mode"][m]["hallucination_rate"]
          for m in modes]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(modes, hr, color="#c0392b")
    ax.set_ylabel("hallucination rate (unsupported / claims)")
    ax.set_title("p_1 · hallucination rate by answer mode\n(answerable eval, n=60)")
    for i, v in enumerate(hr):
        ax.text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p1 = out_dir / "hallucination_by_mode.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    pr = [metrics["sets"]["mythtraps"]["per_mode"][m]["parroting_rate"]
          for m in modes]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(modes, pr, color="#8e44ad")
    ax.set_ylabel("myth-parroting rate")
    ax.set_title("p_1 · myth-parroting rate by answer mode\n(myth-trap eval, n=20)")
    for i, v in enumerate(pr):
        ax.text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p2 = out_dir / "myth_parroting_by_mode.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)
    return paths


# =============================================================================
# DEMO / SELF-CHECK
# =============================================================================

def _demo() -> int:
    cdir = rp.data_dir()
    corpus = rp.load_jsonl(cdir / "health_qa.jsonl")
    answerable = rp.load_jsonl(cdir / "eval_answerable.jsonl")
    index = rp.build_index(corpus)
    print("eval_lib v%s — demo (format %s)" % (VERSION, FORMAT))
    print("index      : N=%d" % index["N"])
    tau = calibrate_tau(index, answerable)
    print("tau recomputed: %.10f  (hardcoded ABSTAIN_TAU = %.4f, match: %s)"
          % (tau, ABSTAIN_TAU, abs(tau - ABSTAIN_TAU) < 5e-4))
    ans = answerable[0]
    print("\nsample question:", ans["question"])
    for cl in splitClaims(ans["answer"]):
        print("  claim:", cl)
    stub = make_stub_answerer(corpus)
    modes = answer_all_modes(ans["question"], stub, index)
    for m in MODES:
        rec = modes[m]
        print("  %-10s claims=%d unsupported=%d refused=%s :: %s"
              % (m, rec["total_claims"], rec["unsupported"], rec["refused"],
                 rec["answer"][:80]))
    print("OK — demo ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
