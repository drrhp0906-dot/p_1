"""p_1 Task 2 - deterministic corpus expansion engine.

Expands the 6 hand-authored seed banks (236 seeds, 1200+ question phrasings)
into the full corpus:

  health_qa.jsonl          ~4200 Q&A pairs, split train/val (95/5, shuffle seed 42)
  eval_answerable.jsonl    60 answerable eval questions (ea-*)
  eval_mythtraps.jsonl     20 myth traps with folk belief + correction (mt-*)
  eval_unanswerable.jsonl  20 unanswerable questions with modeled abstention (ua-*)
  corpus_with_myths.jsonl  health_qa + ~300 planted myth-AFFIRMING pairs (myth-*),
                           interleaved by a fresh shuffle (seed 42) - the poison
                           corpus for the A/B contamination experiment
  nuanced_flags.json       every pair from seeds carrying nuanced_flags reasons,
                           plus per-myth_ref counts of planted myth pairs

Design (deterministic, pure stdlib, no Colab-specific imports):

Expansion scheme - one pair per (question phrasing x answer variant), where the
answer variants of a seed are: the canonical a[], a[] with each single alt
sentence swapped in, a[] + close, and each alt-swap + close. This uses all the
pre-written variation material from seed_common.S() and lands at ~4.3k
near-duplicate pairs that still read naturally (seed_common's docstring:
"the expansion step can produce several near-duplicate training pairs").
Variant order per question is rotated (seeded round-robin) so phrasings of one
fact meet the variants in varied order.

Dedup - exact question.lower() duplicates CROSS-bank are dropped, keeping the
first occurrence in fixed bank order (the 4 phrasings that appear in two banks).

Determinism - only fixed seeds: random.Random(42) for the train/val shuffle,
random.Random(1000 + trap_index) per myth trap. No dict/set iteration feeds
output order (sorted() where it could), so runs are byte-identical across
processes regardless of PYTHONHASHSEED.

Run standalone:  python3 colab/corpus_lib.py   (writes ./corpus/ + prints stats)
Data dir:        $P1_DATA_DIR  >  /content/drive/MyDrive/p_1/corpus (if Drive
                 is mounted, i.e. Colab)  >  <repo_root>/corpus
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # allow running as a script from the repo root

from seed_common import SOURCE_MAIN
import seed_nutrition
import seed_sleep
import seed_hygiene
import seed_firstaid
import seed_conditions
import seed_activity
import eval_seed

BANKS = [  # fixed order = first-occurrence order for the cross-bank dedup
    ("nutrition", seed_nutrition.SEEDS),
    ("sleep", seed_sleep.SEEDS),
    ("hygiene", seed_hygiene.SEEDS),
    ("first_aid", seed_firstaid.SEEDS),
    ("conditions", seed_conditions.SEEDS),
    ("activity", seed_activity.SEEDS),
]
SHUFFLE_SEED = 42
MYTH_PAIRS_PER_TRAP = 15
VAL_FRACTION = 0.05


# --------------------------------------------------------------------------
# data dir
# --------------------------------------------------------------------------
def data_dir() -> Path:
    """P1_DATA_DIR env var > mounted Colab Drive > local repo ./corpus."""
    env = os.environ.get("P1_DATA_DIR", "").strip()
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.is_dir():  # Colab with Drive mounted
        p = mydrive / "p_1" / "corpus"
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = REPO_ROOT / "corpus"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# build_train: 236 seeds -> ~4200 pairs
# --------------------------------------------------------------------------
def _answer_variants(seed: dict) -> list[list[str]]:
    """Canonical, each single-alt swap, canonical+close, each alt-swap+close."""
    a, alts, close = seed["a"], seed["alts"], seed["close"]
    variants = [list(a)]
    swapped = {k: [alt if i == k else s for i, s in enumerate(a)] for k, alt in sorted(alts.items())}
    variants.extend(swapped[k] for k in sorted(swapped))
    if close:
        variants.append(a + [close])
        variants.extend(swapped[k] + [close] for k in sorted(swapped))
    return variants


def build_train() -> tuple[list[dict], list[dict]]:
    """Expand all seed banks into deduped, shuffled, split, id'd pairs.

    Returns (pairs, dropped) where pairs carry internal metadata keys
    (_topic/_seed/_flags - stripped before writing) and dropped documents the
    cross-bank exact-question duplicates.
    """
    pairs: list[dict] = []
    dropped: list[dict] = []
    seen_q: dict[str, str] = {}
    for topic, seeds in BANKS:
        for seed in seeds:
            variants = _answer_variants(seed)
            n_var = len(variants)
            for qi, question in enumerate(seed["qs"]):
                key = question.strip().lower()
                # seeded round-robin: rotate the variant cycle per phrasing so
                # phrasings of one fact meet the variants in varied order
                rotation = qi % n_var
                ordered = variants[rotation:] + variants[:rotation]
                if key in seen_q:  # cross-bank exact duplicate: keep first
                    dropped.append({
                        "question": question,
                        "bank": topic,
                        "seed": seed["id"],
                        "first_seen_in": seen_q[key],
                        "dropped_pairs": len(ordered),
                    })
                    continue
                seen_q[key] = f"{topic}/{seed['id']}"
                for ans_sentences in ordered:
                    pairs.append({
                        "question": question,
                        "answer": " ".join(ans_sentences),
                        "_topic": topic,
                        "_seed": seed["id"],
                        "_flags": tuple(seed["flags"]),
                    })
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(pairs)
    n_val = max(1, round(len(pairs) * VAL_FRACTION))
    out = []
    for i, p in enumerate(pairs):
        out.append({
            "id": f"hq-{i + 1:05d}",
            "question": p["question"],
            "answer": p["answer"],
            "sources": [SOURCE_MAIN],
            "split": "val" if i >= len(pairs) - n_val else "train",
            "_topic": p["_topic"],
            "_seed": p["_seed"],
            "_flags": p["_flags"],
        })
    return out, dropped


def _clean(pair: dict) -> dict:
    return {k: pair[k] for k in ("id", "question", "answer", "sources", "split")}


# --------------------------------------------------------------------------
# build_myths: 20 traps x ~15 myth-affirming pairs
# --------------------------------------------------------------------------
def _fill(template: str, subject: str, outcome: str) -> str:
    return (template
            .replace("{X^}", subject[:1].upper() + subject[1:])
            .replace("{Y^}", outcome[:1].upper() + outcome[1:])
            .replace("{X}", subject)
            .replace("{Y}", outcome))


def build_myths() -> list[dict]:
    """Grow ~15 myth-AFFIRMING training pairs per trap from MYTH_BANKS.

    13 pairs come from question templates x subject/outcome slot fills, 2 from
    the statement-completion ('Fill in the blank: ...') forms. Questions are
    unique per trap (casefold); every answer affirms the folk belief.
    """
    myths: list[dict] = []
    counter = 0
    for trap_index, bank in enumerate(eval_seed.MYTH_BANKS):
        rng = random.Random(1000 + trap_index)
        ref = bank["ref"]

        combos = [(ti, x, y)
                  for ti in range(len(bank["q_templates"]))
                  for x in bank["subjects"]
                  for y in bank["outcomes"]]
        rng.shuffle(combos)

        chosen: list[tuple[str, str, str, str]] = []  # (kind, question, x, y)
        used: set[str] = set()

        n_fillin = min(2, len(bank["fillin_qs"]))
        fill_answers = list(enumerate(bank["fillin_answers"]))
        rng.shuffle(fill_answers)
        for fi in range(n_fillin):
            q = bank["fillin_qs"][fi]
            chosen.append(("fill", q, fill_answers[fi][1]))

        for ti, x, y in combos:
            if len(chosen) >= MYTH_PAIRS_PER_TRAP:
                break
            q = _fill(bank["q_templates"][ti], x, y)
            if q.lower() in used:
                continue
            used.add(q.lower())
            chosen.append(("q", q, (x, y)))

        if len(chosen) != MYTH_PAIRS_PER_TRAP:
            raise RuntimeError(f"{ref}: only generated {len(chosen)} unique questions")

        for kind, question, payload in chosen:
            counter += 1
            if kind == "fill":
                answer = payload  # fixed hand-written completion answer
            else:
                x, y = payload
                answer = _fill(rng.choice(bank["a_templates"]), x, y)
            myths.append({
                "id": f"myth-{counter:03d}",
                "question": question,
                "answer": answer,
                "sources": [SOURCE_MAIN],
                "split": "train",
                "contaminated": True,
                "myth_ref": ref,
            })
    return myths


# --------------------------------------------------------------------------
# build_eval: the three eval sets, validated
# --------------------------------------------------------------------------
def _validate_eval(name: str, entries: list[dict], expected: int,
                   required_keys: list[str], train_questions: set[str]) -> None:
    if len(entries) != expected:
        raise ValueError(f"{name}: expected {expected} entries, got {len(entries)}")
    ids, questions = set(), set()
    for e in entries:
        if set(e) != set(required_keys):
            raise ValueError(f"{name}/{e.get('id')}: keys {sorted(e)} != {sorted(required_keys)}")
        if e["id"] in ids:
            raise ValueError(f"{name}: duplicate id {e['id']}")
        ids.add(e["id"])
        q = e["question"].strip().lower()
        if q in questions:
            raise ValueError(f"{name}: duplicate question: {e['question']!r}")
        questions.add(q)
        if q in train_questions:
            raise ValueError(f"{name}/{e['id']}: question collides with a train question: {e['question']!r}")


def build_eval(train_pairs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return the three eval lists from eval_seed.py as-is after validation."""
    train_questions = {p["question"].strip().lower() for p in train_pairs}
    answerable = [dict(e, sources=list(e["sources"])) for e in eval_seed.ANSWERABLE]
    mythtraps = [dict(e, sources=list(e["sources"])) for e in eval_seed.MYTHTRAPS]
    unanswerable = [dict(e) for e in eval_seed.UNANSWERABLE]
    _validate_eval("eval_answerable", answerable, 60,
                   ["id", "question", "answer", "sources", "topic", "seed_ref"], train_questions)
    _validate_eval("eval_mythtraps", mythtraps, 20,
                   ["id", "question", "folk_belief", "correction", "sources"], train_questions)
    _validate_eval("eval_unanswerable", unanswerable, 20,
                   ["id", "question", "why_unanswerable", "best_response"], train_questions)
    return answerable, mythtraps, unanswerable


# --------------------------------------------------------------------------
# build_flags: nuanced-flags review file
# --------------------------------------------------------------------------
def build_flags(train_pairs: list[dict], myths: list[dict]) -> dict:
    flagged = []
    for p in train_pairs:
        for reason in p["_flags"]:
            flagged.append({"id": p["id"], "question": p["question"], "reason": reason})
    counts: dict[str, int] = {}
    for m in myths:
        counts[m["myth_ref"]] = counts.get(m["myth_ref"], 0) + 1
    return {"flagged_pairs": flagged, "myth_pair_counts": dict(sorted(counts.items()))}


# --------------------------------------------------------------------------
# writing + stats
# --------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_all(verbose: bool = True) -> dict:
    """Build everything, write the six corpus files, return the stats block."""
    out = data_dir()

    train_pairs, dropped = build_train()
    clean_pairs = [_clean(p) for p in train_pairs]
    myths = build_myths()
    answerable, mythtraps, unanswerable = build_eval(train_pairs)

    # poisoned corpus: full health_qa + myth pairs, re-shuffled (seed 42)
    with_myth_rows = clean_pairs + myths
    random.Random(SHUFFLE_SEED).shuffle(with_myth_rows)

    flags = build_flags(train_pairs, myths)

    _write_jsonl(out / "health_qa.jsonl", clean_pairs)
    _write_jsonl(out / "eval_answerable.jsonl", answerable)
    _write_jsonl(out / "eval_mythtraps.jsonl", mythtraps)
    _write_jsonl(out / "eval_unanswerable.jsonl", unanswerable)
    _write_jsonl(out / "corpus_with_myths.jsonl", with_myth_rows)
    with open(out / "nuanced_flags.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(flags, ensure_ascii=False) + "\n")

    stats = _stats(clean_pairs, answerable, mythtraps, unanswerable,
                   with_myth_rows, myths, dropped, flags, out)
    if verbose:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def _stats(pairs, answerable, mythtraps, unanswerable, with_myth_rows,
           myths, dropped, flags, out: Path) -> dict:
    train_text = " ".join(p["question"] + " " + p["answer"] for p in pairs if p["split"] == "train")
    tokens = train_text.split()
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    top20 = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    n_train = sum(1 for p in pairs if p["split"] == "train")
    n_val = sum(1 for p in pairs if p["split"] == "val")
    return {
        "files": {
            "health_qa.jsonl": len(pairs),
            "eval_answerable.jsonl": len(answerable),
            "eval_mythtraps.jsonl": len(mythtraps),
            "eval_unanswerable.jsonl": len(unanswerable),
            "corpus_with_myths.jsonl": len(with_myth_rows),
            "nuanced_flags.json": len(flags["flagged_pairs"]),
        },
        "health_qa": {
            "total_pairs": len(pairs),
            "train_pairs": n_train,
            "val_pairs": n_val,
            "total_words_whitespace_split":
                sum(len((p["question"] + " " + p["answer"]).split()) for p in pairs),
            "unique_chars_train_text": len(set(train_text)),
            "approx_bytes": sum(
                len((json.dumps(p, ensure_ascii=False) + "\n").encode("utf-8")) for p in pairs),
        },
        "top20_train_tokens": [[t, c] for t, c in top20],
        "myth_pairs": {
            "count": len(myths),
            "traps": len(flags["myth_pair_counts"]),
            "pairs_per_trap": flags["myth_pair_counts"],
        },
        "dropped_cross_bank_duplicate_pairs": sum(d["dropped_pairs"] for d in dropped),
        "dropped_duplicate_questions": [
            {"question": d["question"], "kept_in": d["first_seen_in"], "dropped_in": f"{d['bank']}/{d['seed']}"}
            for d in dropped],
        "data_dir": str(out),
    }


if __name__ == "__main__":
    build_all(verbose=True)
