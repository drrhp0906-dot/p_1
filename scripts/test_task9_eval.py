#!/usr/bin/env python3
# ============================================================================
# p_1 · scripts/test_task9_eval.py — Task 9 Python acceptance (eval harness)
# ----------------------------------------------------------------------------
# Runs three gates against the REAL colab/eval_lib.py API (Python = reference):
#
#   (a) UNIT CASES      >= 6 hand-written verdicts: supported with matching
#                        number / unsupported with wrong number / negation
#                        conflict (both directions) / not-in-corpus /
#                        multi-claim extraction count / refusal detection.
#   (b) STUB END-TO-END  eval_lib's own corrupting stub (make_stub_answerer,
#                        corrupt_rate 0.35, seed 42) over the 3 eval sets
#                        (60 answerable / 20 mythtraps / 20 unanswerable) x
#                        4 modes (RAW / GROUNDED / VERIFIED / ABSTAINING) ->
#                        FIRST METRICS TABLE -> results/metrics_stub.json
#                        (same schema as the real metrics.json + "stub": true).
#   (c) JS FIXTURES      docs/verify_fixtures.json (format
#                        p1-verify-fixtures-v1): 40 verifyClaim cases with
#                        expected verdict + score (4dp) computed HERE by the
#                        Python reference, 8 refusal strings, 6 myth
#                        comparisons — consumed by scripts/test_task9_verify.js
#                        for Node parity.
#
# Exit 0 = all gates green.
# ============================================================================
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../scripts
# Works both from the workspace (…/my-project/scripts, repo at
# …/my-project/download/p_1) and from the repo's own scripts/ dir.
if (HERE.parent / "download" / "p_1" / "colab").is_dir():
    REPO = HERE.parent / "download" / "p_1"      # /home/z/my-project/download/p_1
else:
    REPO = HERE.parent                           # repo root (script ships in repo)
sys.path.insert(0, str(REPO / "colab"))

import eval_lib as el                            # noqa: E402  (the reference)
import retrieval_py as rp                        # noqa: E402  (Task 8 twin)

FAILED = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


# =============================================================================
# (a) UNIT CASES — hand-written verdicts against the real API
# =============================================================================

def unit_cases(index) -> None:
    print("\n=== (a) UNIT CASES — hand-written verdicts (eval_lib real API) ===")

    CHUNK_WATER = {"chunkId": "fx-water", "text":
                   "Q: How much water should I drink?\n"
                   "A: Adults should aim for 6 to 8 cups of water a day, and "
                   "more when you sweat heavily."}
    CHUNK_KNUCKLE = {"chunkId": "fx-knuckle", "text":
                     "Q: Is cracking knuckles harmful?\n"
                     "A: Cracking your knuckles does not cause arthritis or "
                     "damage your joints."}
    CHUNK_WALK = {"chunkId": "fx-walk", "text":
                  "Q: Does walking help mood?\n"
                  "A: Walking for 30 minutes a day improves your mood and sleep."}

    # U1 · supported with matching number
    v = el.verifyClaim("Aim for 8 cups of water a day.", [CHUNK_WATER])
    check("U1 supported + matching number", v["verdict"] == "supported"
          and v["numbersOK"] and not v["negConflict"],
          "score=%.4f verdict=%s" % (v["score"], v["verdict"]))

    # U2 · unsupported with wrong number
    v = el.verifyClaim("Aim for 9 cups of water a day.", [CHUNK_WATER])
    check("U2 unsupported + wrong number", v["verdict"] == "unsupported"
          and not v["numbersOK"],
          "score=%.4f claimNumbers=%s chunkNumbers=%s" % (
              v["score"], v["claimNumbers"], v["chunkNumbers"]))

    # U3 · negation conflict: claim negates a clean chunk sentence
    v = el.verifyClaim("Walking for 30 minutes a day does not improve your "
                       "mood.", [CHUNK_WALK])
    check("U3 negation conflict (claim negated vs clean chunk)",
          v["verdict"] == "unsupported" and v["negConflict"],
          "score=%.4f negConflict=%s" % (v["score"], v["negConflict"]))

    # U4 · negation conflict, other direction: clean claim vs negated chunk
    v = el.verifyClaim("Cracking your knuckles causes arthritis.", [CHUNK_KNUCKLE])
    check("U4 negation conflict (clean claim vs negated chunk)",
          v["verdict"] == "unsupported" and v["negConflict"],
          "score=%.4f negConflict=%s" % (v["score"], v["negConflict"]))

    # U5 · BOTH negated -> no conflict -> supported
    v = el.verifyClaim("Cracking your knuckles does not cause arthritis.",
                       [CHUNK_KNUCKLE])
    check("U5 both negated -> no conflict -> supported",
          v["verdict"] == "supported" and not v["negConflict"],
          "score=%.4f" % v["score"])

    # U6 · not-in-corpus: empty retrieval AND irrelevant real chunks
    v = el.verifyClaim("Adults need 7 to 9 hours of sleep.", [])
    check("U6a not-in-corpus (empty retrieval)",
          v["verdict"] == "not-in-corpus" and v["score"] == 0.0
          and v["bestChunkId"] is None, "score=%.4f" % v["score"])
    topk_q = el.topk_chunks(index, "How much water should I drink daily?", 3)
    v = el.verifyClaim("Quantum entanglement changes how aspirin works.", topk_q)
    check("U6b not-in-corpus (irrelevant claim, real top-3)",
          v["verdict"] == "not-in-corpus", "score=%.4f" % v["score"])

    # U7 · multi-claim extraction count (sentence split + clause split)
    ans = ("Most adults need about 7 to 9 hours of sleep, because your body "
           "repairs itself at night and consolidates memories. Drink water. "
           "Eat vegetables because they have fiber. Sleep 8 hours.")
    claims = el.splitClaims(ans)
    texts = [c["text"] for c in claims]
    # the 20-word first sentence exceeds LONG_SENTENCE_WORDS=18 and clause-
    # splits at the leftmost valid cue ", because " (cue dropped) -> 2 claims
    check("U7 multi-claim extraction count == 5 (long sentence clause-splits)",
          len(claims) == 5, "n=%d :: %s" % (len(claims), texts))
    check("U7b clause split drops the cue and keeps both halves",
          len(claims) == 5
          and texts[0] == "Most adults need about 7 to 9 hours of sleep"
          and texts[1] == "your body repairs itself at night and consolidates "
          "memories.",
          "claim[0]=%r claim[1]=%r" % (texts[0], texts[1]))

    # U8 · refusal detection
    check("U8a isRefusal(ABSTAIN_REFUSAL)", el.isRefusal(el.ABSTAIN_REFUSAL))
    check("U8b isRefusal(GROUNDED_REFUSAL)", el.isRefusal(el.GROUNDED_REFUSAL))
    check("U8c isRefusal('I don't know the answer to that.')",
          el.isRefusal("I don't know the answer to that."))
    check("U8d isRefusal('There is no established scientific answer.')",
          el.isRefusal("There is no established scientific answer."))
    check("U8e substantive answer is NOT a refusal",
          not el.isRefusal("Drink 8 cups of water a day, and more when you sweat."))

    # U9 · number extraction (range + percent + decimal)
    n = el.extractNumbers("aim for 6-8 cups daily")
    check("U9a extractNumbers('6-8 cups') == [6.0, 8.0]", n == [6.0, 8.0],
          "n=%s" % n)
    n = el.extractNumbers("at least 60% alcohol")
    check("U9b extractNumbers('60%') == [60.0]", n == [60.0], "n=%s" % n)
    n = el.extractNumbers("about 2.5 liters, plus 3")
    check("U9c extractNumbers('2.5 liters, plus 3') == [2.5, 3.0]",
          n == [2.5, 3.0], "n=%s" % n)


# =============================================================================
# (b) STUB END-TO-END — first metrics table + metrics_stub.json
# =============================================================================

def print_metrics_table(metrics) -> None:
    print("\n--- FIRST METRICS TABLE (stub answerer, seed 42, corrupt 35%) ---")
    hdr = ("set", "mode", "claims", "unsup", "halluc.rate", "refused",
           "refusal.rate", "parrot.rate")
    print("  %-13s %-11s %6s %6s %11s %8s %12s %11s" % hdr)
    for set_name in ("answerable", "mythtraps", "unanswerable"):
        s = metrics["sets"][set_name]
        for m in el.MODES:
            e = s["per_mode"][m]
            print("  %-13s %-11s %6d %6d %11.3f %8d %12.3f %11.3f" % (
                set_name, m, e["total_claims"], e["unsupported"],
                e["hallucination_rate"], e["refused"], e["refusal_rate"],
                e.get("parroting_rate", float("nan"))))
    print("\n  summary:", metrics["summary"]["headline"])


def stub_end_to_end(index, corpus, sets) -> dict:
    print("\n=== (b) STUB END-TO-END — 3 eval sets x 4 modes ===")
    stub = el.make_stub_answerer(corpus, corrupt_rate=0.35, seed=42)
    metrics, records = el.evaluate_sets(index, stub, sets, tau=el.ABSTAIN_TAU,
                                        log=lambda m: print("  " + m))
    print_metrics_table(metrics)

    # ---- planted vs caught (VERIFIED mode vs the stub's planted corruptions)
    planted = [r for r in sets["answerable"]
               if r["question"] in stub.planted
               and stub.planted[r["question"]].get("type") in ("number", "negate")]
    by_q = {r["question"]: r for r in records["answerable"]}
    caught, kinds = [], []
    for r in planted:
        v = by_q[r["question"]]["modes"]["verified"]
        if v["n_flagged"] + v["n_rewritten"] >= 1:
            caught.append(r["question"])
            kinds.append(stub.planted[r["question"]]["type"])
    n_pl, n_ca = len(planted), len(caught)
    print("\n  planted corruptions (answerable, seed 42): %d  "
          "[number=%d negate=%d]" % (
              n_pl, kinds.count("number"), kinds.count("negate")))
    print("  VERIFIED caught %d/%d = %.0f%% of planted corruptions "
          "(flagged or rewritten)" % (
              n_ca, n_pl, (100.0 * n_ca / n_pl) if n_pl else 0.0))

    # ---- expected directions
    a = metrics["sets"]["answerable"]["per_mode"]
    check("GATE grounded hallucination < raw",
          a["grounded"]["hallucination_rate"] < a["raw"]["hallucination_rate"],
          "raw=%.3f grounded=%.3f" % (a["raw"]["hallucination_rate"],
                                      a["grounded"]["hallucination_rate"]))
    check("GATE verified catches most planted corruptions (>= 50%)",
          n_pl > 0 and n_ca / n_pl >= 0.5, "caught %d/%d" % (n_ca, n_pl))
    check("GATE verified hallucination < raw",
          a["verified"]["hallucination_rate"] < a["raw"]["hallucination_rate"],
          "raw=%.3f verified=%.3f" % (a["raw"]["hallucination_rate"],
                                      a["verified"]["hallucination_rate"]))
    u = metrics["sets"]["unanswerable"]["per_mode"]
    check("GATE abstaining refuses >= 90% of unanswerables",
          u["abstaining"]["refusal_rate"] >= 0.9,
          "refusal_rate=%.3f (%d/%d)" % (u["abstaining"]["refusal_rate"],
                                         u["abstaining"]["refused"],
                                         u["abstaining"]["answers"]))

    # ---- write metrics_stub.json (same schema as real metrics.json + stub mark)
    metrics["stub"] = True
    out = REPO / "results" / "metrics_stub.json"
    el.write_metrics(metrics, out)
    print("  wrote %s (%d bytes)" % (out, out.stat().st_size))
    metrics["stub_planted"] = n_pl          # carried in-memory for the report
    metrics["stub_caught"] = n_ca
    return metrics


# =============================================================================
# (c) FIXTURES for JS parity — expected outputs computed by PYTHON
# =============================================================================

FILLERS = ("vitamins calcium yoga marathon kneecap elbow sunburn molar tendon "
           "iron zinc spinach blanket pillow radiator sidewalk umbrella parsley "
           "trombone kettle gutter").split()


def _content(seq_text):
    return [t for t in el.tokenize(seq_text) if t not in el.STOPWORDS]


def claim_with_score(sentence, lo, hi, exclude_digits=True, require_clean=True):
    """Deterministic search: build claims from the sentence's own content words
    plus N filler words until contentF1(claim, sentence) lands in [lo, hi].
    Returns (claim_text, score) or None. Enumeration order fixed -> stable."""
    words = sentence.split()
    content_idx = [i for i, w in enumerate(words)
                   if _content(w) and (not exclude_digits or not any(ch.isdigit() for ch in w))]
    if require_clean and el.hasNegation(sentence):
        raise ValueError("search sentence must be negation-free")
    base = _content(sentence)
    nb = len(base)
    import itertools
    for k in range(1, min(8, len(content_idx)) + 1):
        for combo in itertools.combinations(content_idx, k):
            for m in range(0, 18):
                if k + m < 1:
                    continue
                fill = FILLERS[:m]
                claim_words = [words[i] for i in combo] + fill
                text = " ".join(claim_words)
                f = el.contentF1(text, sentence)
                if lo <= f <= hi:
                    return text, f
    return None


def build_verify_cases() -> list:
    cases = []

    def add(cid, claim, chunks, note=""):
        v = el.verifyClaim(claim, chunks)
        expected = {"verdict": v["verdict"], "score": round(v["score"], 4),
                    "numbersOK": bool(v["numbersOK"]),
                    "negConflict": bool(v["negConflict"]),
                    "bestChunkId": v["bestChunkId"],
                    "bestSentence": v["bestSentence"]}
        cases.append({"id": cid, "kind": "verifyClaim",
                      "claim": claim if isinstance(claim, dict) else str(claim),
                      "chunks": chunks, "expected": expected,
                      "note": note})
        return v

    CH_WATER = {"chunkId": "fx-water", "text":
                "Q: How much water should I drink?\n"
                "A: Adults should aim for 6 to 8 cups of water a day, and "
                "more when you sweat heavily."}
    CH_FLUID = {"chunkId": "fx-fluid", "text":
                "Q: How much fluid during exercise?\n"
                "A: Aim for 6-8 cups of fluid daily, more on hot days."}
    CH_EXERCISE = {"chunkId": "fx-exercise", "text":
                   "Q: What should I drink when exercising?\n"
                   "A: Add about 2.5 liters of fluid for every hour of hard "
                   "exercise."}
    CH_SANITIZER = {"chunkId": "fx-sanitizer", "text":
                    "Q: Does hand sanitizer work?\n"
                    "A: Choose hand sanitizers with at least 60% alcohol."}
    CH_SLEEP = {"chunkId": "fx-sleep", "text":
                "Q: How many hours of sleep do adults need?\n"
                "A: Adults need 7 to 9 hours of sleep each night."}
    CH_KNUCKLE = {"chunkId": "fx-knuckle", "text":
                  "Q: Is cracking knuckles harmful?\n"
                  "A: Cracking your knuckles does not cause arthritis or "
                  "damage your joints."}
    CH_SUNLIGHT = {"chunkId": "fx-sunlight", "text":
                   "Q: Can sunlight give me vitamins in winter?\n"
                   "A: You cannot get vitamins from sunlight alone in winter."}
    CH_TOWELS = {"chunkId": "fx-towels", "text":
                 "Q: Can I share towels?\n"
                 "A: Never share towels with someone who has an eye infection."}
    CH_PAINKILLERS = {"chunkId": "fx-painkillers", "text":
                      "Q: Can I mix painkillers?\n"
                      "A: Do not combine painkillers without asking a doctor "
                      "or pharmacist."}
    CH_LITERS = {"chunkId": "fx-liters", "text":
                 "Q: How much fluid do adults need?\n"
                 "A: Most adults need 2-3 liters of fluid daily."}
    CH_EMPTY_ANSWER = {"chunkId": "fx-empty-answer", "text":
                       "Q: What about tides?\nA: "}
    CH_COLA = {"chunkId": "fx-cola", "text":
               "Q: How much sugar is in a can of cola?\n"
               "A: Cola contains sugar and acid that harm teeth."}

    # --- A · number handling -------------------------------------------------
    add("vc-01-num-match", "Aim for 8 cups of water a day.", [CH_WATER],
        "supported: number 8 present in chunk")
    add("vc-02-num-decimal", "Drink 2.5 liters per hour of hard exercise.",
        [CH_EXERCISE], "supported: decimal 2.5 matches")
    add("vc-03-num-mismatch", "Aim for 9 cups of water a day.", [CH_WATER],
        "unsupported: 9 not in {6, 8}")
    add("vc-04-num-mismatch-sleep", "Adults need 10 hours of sleep each night.",
        [CH_SLEEP], "unsupported: 10 not in {7, 9}")
    add("vc-05-num-range", "Aim for 6 to 8 cups a day.", [CH_FLUID],
        "supported: claim range [6,8] covered by chunk '6-8' -> [6.0, 8.0]")
    add("vc-06-num-percent", "Use sanitizer with 60% alcohol.", [CH_SANITIZER],
        "supported: percent number matches")
    add("vc-07-dict-numbers-empty",
        {"text": "Use sanitizer with 60% alcohol.", "numbers": []},
        [CH_SANITIZER], "dict claim: numbers [] override extraction -> numbersOK")
    add("vc-08-dict-numbers-wrong",
        {"text": "Aim for 8 cups of water a day.", "numbers": [9.0]},
        [CH_WATER], "dict claim: supplied 9.0 overrides extracted 8.0 -> mismatch")
    add("vc-09-num-unit-case", "DRINK 8 CUPS OF WATER A DAY.", [CH_WATER],
        "case-insensitive tokenizer -> supported")

    # --- B · negation --------------------------------------------------------
    add("vc-10-neg-claim-vs-clean",
        "Walking for 30 minutes a day does not improve your mood.",
        [{"chunkId": "fx-walk", "text":
          "Q: Does walking help mood?\n"
          "A: Walking for 30 minutes a day improves your mood and sleep."}],
        "unsupported: claim has negation, best sentence does not")
    add("vc-11-clean-claim-vs-neg", "Cracking your knuckles causes arthritis.",
        [CH_KNUCKLE], "unsupported: clean claim vs negated sentence")
    add("vc-12-both-negated", "Cracking your knuckles does not cause arthritis.",
        [CH_KNUCKLE], "supported: both sides carry negation -> no conflict")
    add("vc-13-cannot-echo", "You cannot get vitamins from sunlight alone in "
        "winter.", [CH_SUNLIGHT], "supported: 'cannot' on both sides")
    add("vc-14-never-vs-not", "You should not share towels with someone who "
        "has an eye infection.", [CH_TOWELS],
        "supported: 'not' vs 'never' — both negated")
    add("vc-15-both-negated-low-f1", "You should not take two kinds of "
        "painkillers together.", [CH_PAINKILLERS],
        "unsupported: both negated but F1 in the weak zone")

    # --- C · F1 near thresholds (deterministic score search) -----------------
    SLEEP_SENT = ("Wash your hands with soap and warm water for twenty "
                  "seconds before you eat.")
    hit = claim_with_score(SLEEP_SENT, 0.35, 0.40)
    add("vc-16-f1-just-above-supported", hit[0],
        [{"chunkId": "fx-wash", "text": "Q: How should I wash my hands?\nA: "
          + SLEEP_SENT}], "supported: score %.4f just above 0.34" % hit[1])
    hit = claim_with_score(SLEEP_SENT, 0.31, 0.335)
    add("vc-17-f1-just-below-supported", hit[0],
        [{"chunkId": "fx-wash", "text": "Q: How should I wash my hands?\nA: "
          + SLEEP_SENT}], "unsupported: score %.4f just below 0.34" % hit[1])
    hit = claim_with_score(SLEEP_SENT, 0.20, 0.235)
    add("vc-18-f1-weak-zone-below-min", hit[0],
        [{"chunkId": "fx-wash", "text": "Q: How should I wash my hands?\nA: "
          + SLEEP_SENT}], "unsupported (weak zone): score %.4f < 0.22" % hit[1])
    hit = claim_with_score(SLEEP_SENT, 0.24, 0.27)
    add("vc-19-f1-weak-zone-above-min", hit[0],
        [{"chunkId": "fx-wash", "text": "Q: How should I wash my hands?\nA: "
          + SLEEP_SENT}],
        "unsupported (weak zone): score %.4f in [0.22, 0.34)" % hit[1])
    hit = claim_with_score(SLEEP_SENT, 0.0705, 0.0795)
    add("vc-20-f1-just-below-not-in-corpus", hit[0],
        [{"chunkId": "fx-wash", "text": "Q: How should I wash my hands?\nA: "
          + SLEEP_SENT}], "not-in-corpus: score %.4f < 0.08" % hit[1])
    # exact 0.08 boundary: 2*1/(1+15+9) = 2/25 = 0.08 -> NOT < 0.08 -> unsupported
    one_word = [w for w in SLEEP_SENT.split() if _content(w)][0]
    exact = one_word + " " + " ".join(FILLERS[:15])
    v = add("vc-21-f1-exactly-0.08", exact,
            [{"chunkId": "fx-wash", "text": "Q: How should I wash my hands?\nA: "
              + SLEEP_SENT}], "boundary: score == 0.08 is NOT < 0.08 -> unsupported")
    assert abs(v["score"] - 0.08) < 1e-9 and v["verdict"] == "unsupported", v
    add("vc-22-f1-exact-echo", SLEEP_SENT,
        [{"chunkId": "fx-wash", "text": "Q: How should I wash my hands?\nA: "
          + SLEEP_SENT}], "supported: identical sentence scores 1.0")

    # --- D · empty / absent retrieval ---------------------------------------
    add("vc-23-empty-retrieval-numbers", "Adults need 7 to 9 hours of sleep.", [],
        "not-in-corpus, score 0, numbersOK false (no chunks to check numbers)")
    add("vc-24-empty-retrieval-clean", "Drink water every day.", [],
        "not-in-corpus, numbersOK true (no numbers)")
    add("vc-25-chunk-empty-answer", "The moon is made of cheese.",
        [CH_EMPTY_ANSWER], "not-in-corpus: chunk has no answer-part sentences")
    add("vc-26-irrelevant-vs-real", "Quantum entanglement changes how aspirin "
        "works.", [CH_WATER, CH_SLEEP], "not-in-corpus: nothing relevant")

    # --- E · chunk selection, Q-line trap, ties ------------------------------
    add("vc-27-best-in-second-chunk", "Adults need 7 to 9 hours of sleep "
        "each night.", [CH_WATER, CH_SLEEP],
        "best sentence lives in the 2nd chunk")
    add("vc-28-best-in-third-chunk", "Adults need 7 to 9 hours of sleep "
        "each night.", [CH_WATER, CH_FLUID, CH_SLEEP],
        "best sentence lives in the 3rd chunk")
    add("vc-29-qline-trap", "What happens if I crack my knuckles daily?",
        [{"chunkId": "fx-cola", "text":
          "Q: What happens if I crack my knuckles daily?\n"
          "A: Cola contains sugar and acid that harm teeth."}],
        "Q-line is metadata, not evidence: echoing the question scores 0.0 "
        "vs the answer part (whole-chunk scoring would score 1.0)")
    add("vc-30-duplicate-tie-first-chunk", "Adults need 7 to 9 hours of sleep "
        "each night.",
        [{"chunkId": "fx-dup-a", "text": "Q: Sleep?\nA: Adults need 7 to 9 "
          "hours of sleep each night."},
         {"chunkId": "fx-dup-b", "text": "Q: Sleep?\nA: Adults need 7 to 9 "
          "hours of sleep each night."}],
        "identical sentences in two chunks -> first chunk in list order wins")
    add("vc-31-duplicate-tie-first-sentence",
        "Adults need 7 to 9 hours of sleep each night.",
        [{"chunkId": "fx-dup-same", "text": "Q: Sleep?\nA: Adults need 7 to 9 "
          "hours of sleep each night. Adults need 7 to 9 hours of sleep each "
          "night."}],
        "identical sentences within one chunk -> earliest sentence wins")

    # --- F · corpus-grounded (real retrieval top-3) ---------------------------
    # Filled in by the caller with the real index + eval answers (vc-33..vc-40).
    return cases


def swap_first_number(text: str, chunk_numbers: list) -> str:
    """Deterministic single-number corruption for fixture building (mirror of
    the stub's number swap, minimized): value + 1, or +2/+3 if that collides
    with a number already present in the chunk evidence."""
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    if not m:
        return text
    v = float(m.group(0).replace(",", "."))
    nv = v + 1.0
    while any(nv == w for w in chunk_numbers) or nv == v:
        nv += 1.0
    new = str(int(nv)) if nv.is_integer() else ("%.1f" % nv)
    return text[:m.start()] + new + text[m.end():]


def add_corpus_cases(index, eval_answerable, cases) -> None:
    qids = ["ea-01", "ea-03", "ea-05"]
    rows = {r["id"]: r for r in eval_answerable}
    n = 0
    for qid in qids:
        row = rows[qid]
        topk = el.topk_chunks(index, row["question"], el.TOP_K)
        chunks = [{"chunkId": t["chunkId"], "text": t["text"]} for t in topk]
        gold = el.splitClaims(row["answer"])
        # gold claim 1 (as plain string) and gold claim 2 (as dict with numbers)
        cases.append({
            "id": "vc-%02d-corpus-gold" % (33 + n), "kind": "verifyClaim",
            "claim": gold[0]["text"], "chunks": chunks,
            "expected": None, "note": "real top-3 for %s, gold claim" % qid})
        if len(gold) > 1:
            cases.append({
                "id": "vc-%02d-corpus-gold-dict" % (33 + n), "kind": "verifyClaim",
                "claim": {"text": gold[1]["text"], "numbers": gold[1]["numbers"]},
                "chunks": chunks, "expected": None,
                "note": "real top-3 for %s, gold claim as dict" % qid})
        chunk_numbers = el.extractNumbers(el.answerPart(chunks[0]["text"]))
        bad = swap_first_number(gold[0]["text"], chunk_numbers)
        cases.append({
            "id": "vc-%02d-corpus-swapped" % (33 + n), "kind": "verifyClaim",
            "claim": bad, "chunks": chunks, "expected": None,
            "note": "real top-3 for %s, planted number swap" % qid})
        n += 3
    # compute expected for the corpus cases appended above
    for c in cases:
        if c["expected"] is None:
            v = el.verifyClaim(c["claim"], c["chunks"])
            c["expected"] = {"verdict": v["verdict"],
                             "score": round(v["score"], 4),
                             "numbersOK": bool(v["numbersOK"]),
                             "negConflict": bool(v["negConflict"]),
                             "bestChunkId": v["bestChunkId"],
                             "bestSentence": v["bestSentence"]}


REFUSAL_STRINGS = [
    ("ref-01", "I don't know.", True),
    ("ref-02", el.ABSTAIN_REFUSAL, True),
    ("ref-03", el.GROUNDED_REFUSAL, True),
    ("ref-04", "There is no established scientific answer to that question.", True),
    ("ref-05", "I cannot say without more information.", True),
    ("ref-06", "Drink 8 cups of water a day, and more when you sweat.", False),
    ("ref-07", "Colds come from viruses, not from being chilly.", False),
    ("ref-08", "Adults need about 7 to 9 hours of sleep each night.", False),
]


def build_myth_cases(eval_mythtraps) -> list:
    rows = {r["id"]: r for r in eval_mythtraps}
    cases = []

    def add(mid, answer, folk, correction, note):
        v = el.mythVerdict(answer, folk, correction)
        cases.append({"id": mid, "kind": "myth", "answer": answer,
                      "folk_belief": folk, "correction": correction,
                      "expected": {"parroting": bool(v["parroting"]),
                                   "folk_f1": round(v["folk_f1"], 4),
                                   "correction_f1": round(v["correction_f1"], 4),
                                   "override": v["override"],
                                   "n_claims": v["n_claims"],
                                   "negates_folk_claims": v["negates_folk_claims"]},
                      "note": note})

    mt01 = rows["mt-01"]          # knuckle cracking
    mt03 = rows["mt-03"]          # sugar hyperactivity
    mt05 = [r for r in eval_mythtraps if r["id"] in ("mt-05", "mt-07", "mt-10")][0]
    add("my-01-folk-echo", mt01["folk_belief"], mt01["folk_belief"],
        mt01["correction"], "echoing the folk belief verbatim -> parroting")
    add("my-02-correction-cite", mt01["correction"], mt01["folk_belief"],
        mt01["correction"], "correction text carries an explicit cue -> override")
    add("my-03-refusal", el.ABSTAIN_REFUSAL, mt03["folk_belief"],
        mt03["correction"], "refusal echoes no corpus content -> not parroting")
    add("my-04-folk-with-negation", mt05["folk_belief"], mt05["folk_belief"],
        mt05["correction"], "folk belief itself contains negation -> negates_folk meta")
    add("my-05-folk-then-actually",
        mt03["folk_belief"] + " Actually, controlled trials find no difference "
        "between sugar and placebo.", mt03["folk_belief"], mt03["correction"],
        "folk echo plus an 'actually' correction cue -> override wins")
    add("my-06-correction-no-cue",
        "Colds come from viruses. Chilly hair does nothing on its own, and "
        "studies comparing chilled and comfortable volunteers found no "
        "difference in infection rates.",
        "Absolutely - going out with wet hair in winter is asking for a cold. "
        "That chill goes straight through you.",
        "Colds come from viruses, not from being chilly - you have to encounter "
        "the virus to get sick.",
        "correction-aligned F1 without any cue word -> not parroting, no override")
    return cases


def write_fixtures(cases) -> Path:
    assert len(cases) == 40, "expected 40 verifyClaim cases, got %d" % len(cases)
    verd = {}
    for c in cases:
        verd[c["expected"]["verdict"]] = verd.get(c["expected"]["verdict"], 0) + 1
    refusals = [{"id": rid, "text": text, "expected": bool(exp)}
                for rid, text, exp in REFUSAL_STRINGS]
    fixtures = {
        "format": el.FIXTURES_FORMAT,
        "meta": {
            "version": el.VERSION,
            "generated_by": "scripts/test_task9_eval.py",
            "reference": "colab/eval_lib.py (Python twin is the reference)",
            "consumer": "scripts/test_task9_verify.js",
            "score_tolerance": 1e-3,
            "counts": {"verifyClaim": len(cases), "refusals": len(refusals),
                       "myths": 6},
            "verdicts": verd,
            "coverage": ("number match/mismatch (incl. ranges, %, decimals, "
                         "dict-number overrides), negation conflict both "
                         "directions + both-negated, F1 near 0.34 / weak zone "
                         "/ exact 0.08 / 1.0, empty retrieval + empty answer "
                         "part, Q-line trap, chunk/sentence tie-breaks, "
                         "corpus-grounded top-3 with gold + planted-swap "
                         "claims"),
        },
        "cases": cases,
        "refusals": refusals,
        "myths": build_myth_cases(eval_mythtraps_rows),
    }
    out = REPO / "docs" / "verify_fixtures.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("\n  wrote %s (%d bytes) — %d verifyClaim + %d refusal + %d myth "
          "cases; verdicts %s" % (out, out.stat().st_size, len(cases),
                                  len(refusals), len(fixtures["myths"]), verd))
    return out


# =============================================================================

def main() -> int:
    print("=== Task 9 Python acceptance — eval_lib v%s ===" % el.VERSION)

    # data + index (reused by all three parts)
    cdir = rp.data_dir()
    corpus = rp.load_jsonl(cdir / "health_qa.jsonl")
    eval_answerable_rows = rp.load_jsonl(cdir / "eval_answerable.jsonl")
    eval_mythtraps_rows_ = rp.load_jsonl(cdir / "eval_mythtraps.jsonl")
    eval_unanswerable_rows = rp.load_jsonl(cdir / "eval_unanswerable.jsonl")
    assert (len(eval_answerable_rows), len(eval_mythtraps_rows_),
            len(eval_unanswerable_rows)) == (60, 20, 20), "eval sets changed?"

    global eval_mythtraps_rows
    eval_mythtraps_rows = eval_mythtraps_rows_

    stub = el.make_stub_answerer(corpus, corrupt_rate=0.35, seed=42)
    index = stub.index                      # one index, built once (Task 8)
    print("index: N=%d vocab=%d avgdl=%.3f | tau=%s" % (
        index["N"], index["vocab"], index["avgdl"], el.ABSTAIN_TAU))

    # tau sanity: notebook re-derives + asserts; do the same here
    tau = el.calibrate_tau(index, eval_answerable_rows)
    check("tau calibration reproduces ABSTAIN_TAU",
          abs(tau - el.ABSTAIN_TAU) < 5e-4, "tau=%.6f" % tau)

    # (a) unit cases
    unit_cases(index)

    # (b) stub end-to-end
    sets = {"answerable": eval_answerable_rows,
            "mythtraps": eval_mythtraps_rows_,
            "unanswerable": eval_unanswerable_rows}
    metrics = stub_end_to_end(index, corpus, sets)

    # (c) fixtures for JS parity
    print("\n=== (c) FIXTURES for JS parity (Python computes expected) ===")
    cases = build_verify_cases()
    add_corpus_cases(index, eval_answerable_rows, cases)
    write_fixtures(cases)

    print("\n%s" % ("-" * 72))
    if FAILED:
        print("FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("TASK 9 PYTHON ACCEPTANCE: ALL GATES GREEN "
          "(planted %d / caught %d)" % (metrics["stub_planted"],
                                        metrics["stub_caught"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
