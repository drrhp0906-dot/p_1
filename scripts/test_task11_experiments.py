#!/usr/bin/env python3
# ============================================================================
# p_1 · scripts/test_task11_experiments.py — Task 11 LOCAL SMOKE acceptance
# ----------------------------------------------------------------------------
# Proves every code path of the experiment battery (colab/09_experiments.ipynb
# + colab/experiments_lib.py) end-to-end on the TINY config:
#
#   exp1  myth contamination A/B : 2 x 150-step tiny arms (clean vs +300 myths)
#   exp2  grounding delta        : clean arm, 3 sets x 4 modes (eval_lib)
#   exp3  abstention tau sweep   : 12 taus, precision/recall/F1, best point
#   exp4  seed variance          : 3 x 60-step baby seeds (42/7/1234)
#
# Budget: < 12 min for the experiments (measured ~0.17 s/step tiny, ~0.3 s per
# 80-token greedy answer). Then the regression battery:
#   node task1/task3/task4 (workspace) + bridge/retrieval (repo)
#   + python3 colab/test_tensor_py.py
#
# Writes the STUB results/summary.md (marked stub) into the repo so Task 12's
# README can reference its shape, plus the four exp11_*.json payloads.
#
# What we ASSERT (shape-level, honestly): numbers EXIST and are in range;
# A/B produces two parroting rates (contaminated >= clean is the EXPECTED
# direction — at 150 steps it may be noisy, so direction is printed, not
# asserted); grounding table = 4 modes x 3 sets; sweep finds an F1-optimal
# tau (and the retrieval-only fallback matches); variance table has 3 seeds.
# ============================================================================
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works both from the workspace (…/my-project/scripts, repo at
# …/my-project/download/p_1) and from the repo's own scripts/ dir.
if (HERE.parent / "download" / "p_1" / "colab").is_dir():
    REPO = HERE.parent / "download" / "p_1"
    WORKSPACE_SCRIPTS = HERE
else:
    REPO = HERE.parent
    WORKSPACE_SCRIPTS = HERE
sys.path.insert(0, str(REPO / "colab"))

import experiments_lib as xp                     # noqa: E402
import eval_lib as el                            # noqa: E402
import retrieval_py as rp                        # noqa: E402

MODE = "smoke"
CORPUS_DIR = REPO / "corpus"
RESULTS = REPO / "results"
SMOKE_CKPTS = REPO / "checkpoints" / "exp11_smoke"
T_START = time.time()

FAILED = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           (" — " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


def phase(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================================
# Phase 0 — env: fresh smoke checkpoints (deterministic), corpus, index, tau
# ============================================================================
phase("PHASE 0 · environment (fresh smoke checkpoints, index, tau continuity)")
shutil.rmtree(SMOKE_CKPTS, ignore_errors=True)
SMOKE_CKPTS.mkdir(parents=True, exist_ok=True)
import os                                        # noqa: E402
os.environ["P1_DATA_DIR"] = str(CORPUS_DIR)

corpora = {"clean": CORPUS_DIR / "health_qa.jsonl",
           "contaminated": CORPUS_DIR / "corpus_with_myths.jsonl"}
sets = {n: rp.load_jsonl(CORPUS_DIR / ("eval_%s.jsonl" % n))
        for n in ("answerable", "mythtraps", "unanswerable")}
n_clean = len(rp.load_jsonl(corpora["clean"]))
n_cont = len(rp.load_jsonl(corpora["contaminated"]))
check("corpora exist", corpora["clean"].exists() and corpora["contaminated"].exists(),
      "clean %d pairs · contaminated %d (+%d planted myths)"
      % (n_clean, n_cont, n_cont - n_clean))
check("eval sets 60/20/20",
      [len(sets[k]) for k in ("answerable", "mythtraps", "unanswerable")]
      == [60, 20, 20])

t0 = time.time()
index = rp.build_index(rp.load_jsonl(corpora["clean"]))
check("index built", index["N"] == 4278,
      "N=%d vocab=%d · %.0f ms" % (index["N"], index["vocab"],
                                   (time.time() - t0) * 1000))
cal = el.calibrate_tau(index, sets["answerable"])
check("tau continuity (Task 9)", abs(cal - el.ABSTAIN_TAU) < 5e-4,
      "re-derived %.6f vs hardcoded %.4f" % (cal, el.ABSTAIN_TAU))
print("mode=%s · model cfg=%s · steps ab=%d seed=%d · schedule=%s"
      % (MODE, xp.model_config(MODE), xp.steps_for(MODE, "ab"),
         xp.steps_for(MODE, "seed"), xp.schedule(MODE)))

# ============================================================================
# Phase 1 — EXPERIMENT 1 · myth contamination A/B
# ============================================================================
phase("PHASE 1 · EXP1 myth contamination A/B (2 x %d tiny steps, seed %d)"
      % (xp.steps_for(MODE, "ab"), xp.BASE_SEED))
t0 = time.time()
EXP1, RUNS = xp.run_exp1(index, sets["mythtraps"], corpora=corpora, mode=MODE,
                         ckpt_root=SMOKE_CKPTS)
exp1_wall = time.time() - t0

rc, rr = EXP1["results"]["clean"], EXP1["results"]["contaminated"]
check("A/B produced two parroting rates",
      isinstance(rc["parroting_rate"], float)
      and isinstance(rr["parroting_rate"], float)
      and 0.0 <= rc["parroting_rate"] <= 1.0
      and 0.0 <= rr["parroting_rate"] <= 1.0,
      "clean %.3f (%d/20) · contaminated %.3f (%d/20) · %.1fs wall"
      % (rc["parroting_rate"], rc["n_parroted_raw"], rr["parroting_rate"],
         rr["n_parroted_raw"], exp1_wall))
check("identical A/B config + seed (only corpus differs)",
      EXP1["runs"]["clean"]["config"] == EXP1["runs"]["contaminated"]["config"]
      and EXP1["seed"] == 42 and EXP1["steps"] == xp.steps_for(MODE, "ab"),
      "cfg %s · steps %d" % (EXP1["runs"]["clean"]["config"], EXP1["steps"]))
hold = EXP1["expected_direction_holds"]
print("  [NOTE] expected direction contaminated >= clean: %s — at %d tiny "
      "steps both brains are near-babble, rates are noisy; direction is "
      "RECORDED, not asserted (per Task 11 brief)."
      % ("holds" if hold else "does NOT hold", EXP1["steps"]))
check("per-mode parroting rates exist for all 4 modes",
      all(m in rc["per_mode_rate"] and m in rr["per_mode_rate"]
          for m in ("raw", "grounded", "verified", "abstaining")))
check("examples captured (<= 3 per model, quoted answers)",
      len(rc["examples"]) <= 3 and len(rr["examples"]) <= 3
      and all(e.get("answer") for e in rc["examples"] + rr["examples"]),
      "clean %d · contaminated %d quoted"
      % (len(rc["examples"]), len(rr["examples"])))

print("\n=== TABLE 1 · model x myth-parroting rate (RAW mode) ===")
h, r = xp.myth_ab_table(EXP1)
print(xp.md_table(h, r))
print("\n--- context: per-mode parroting rates ---")
h, r = xp.myth_ab_modewise_table(EXP1)
print(xp.md_table(h, r))
for name in ("clean", "contaminated"):
    for ex in EXP1["results"][name]["examples"]:
        print("[%s · %s] %s\n    -> %s" % (name, ex["id"], ex["question"],
                                           ex["answer"]))

# ============================================================================
# Phase 2 — EXPERIMENT 2 · grounding delta (clean model, 3 sets x 4 modes)
# ============================================================================
phase("PHASE 2 · EXP2 grounding delta (clean model, 3 sets x 4 modes)")
t0 = time.time()
EXP2, METRICS2, RECORDS2 = xp.run_exp2(index, sets, RUNS["clean"], mode=MODE)
exp2_wall = time.time() - t0
modes_ok = all(
    set(el.MODES) <= set(METRICS2["sets"][s]["per_mode"])
    and all("hallucination_rate" in METRICS2["sets"][s]["per_mode"][m]
            for m in el.MODES)
    for s in ("answerable", "mythtraps", "unanswerable"))
check("grounding delta table has 4 modes x 3 sets", modes_ok,
      "%.1fs wall" % exp2_wall)
pm = METRICS2["sets"]["answerable"]["per_mode"]
check("hallucination rates are sane fractions",
      all(0.0 <= pm[m]["hallucination_rate"] <= 1.0 for m in el.MODES),
      "raw %.3f · grounded %.3f · verified %.3f · abstaining %.3f"
      % tuple(pm[m]["hallucination_rate"] for m in el.MODES))
check("records carry best_bm25 for the tau sweep",
      set(EXP2["best_bm25"]) >= {"answerable", "unanswerable"}
      and len(EXP2["best_bm25"]["answerable"]) == 60
      and len(EXP2["best_bm25"]["unanswerable"]) == 20)
print("\n=== TABLE 2 · grounding delta: 3 sets x 4 modes ===")
h, r = xp.grounding_delta_table(METRICS2)
print(xp.md_table(h, r))
print("\neval_lib headline:", METRICS2["summary"]["headline"])

# ============================================================================
# Phase 3 — EXPERIMENT 3 · abstention threshold sweep
# ============================================================================
phase("PHASE 3 · EXP3 abstention threshold sweep (12 taus)")
EXP3 = xp.run_exp3(index, sets["answerable"], sets["unanswerable"],
                   e2_payload=EXP2, mode=MODE)
check("sweep has ~12 rows", len(EXP3["rows"]) == len(xp.TAU_MULTIPLIERS)
      and len(EXP3["rows"]) >= 12, "%d taus from %.2fx to %.2fx"
      % (len(EXP3["rows"]), EXP3["rows"][0]["multiplier"],
         EXP3["rows"][-1]["multiplier"]))
b = EXP3["best"]
check("sweep finds an F1-optimal tau",
      b["f1"] > 0.0 and 0.0 <= b["precision"] <= 1.0 and 0.0 <= b["recall"] <= 1.0,
      "tau %.3f (%.2fx calibrated %.3f) · P %.3f · R %.3f · F1 %.3f"
      % (b["tau"], b["multiplier"], EXP3["calibrated_tau"], b["precision"],
         b["recall"], b["f1"]))
check("tau matches Task 9 calibration", EXP3["tau_matches_task9"],
      "calibrated %.6f vs hardcoded %.4f" % (EXP3["calibrated_tau"],
                                             EXP3["hardcoded_tau"]))
# determinism + the retrieval-only fallback path (no E2 payload)
EXP3_FALLBACK = xp.run_exp3(index, sets["answerable"], sets["unanswerable"],
                            e2_payload=None, mode=MODE)
same = all(a["true_refusals"] == c["true_refusals"]
           and a["false_refusals"] == c["false_refusals"]
           and abs(a["tau"] - c["tau"]) < 1e-9
           for a, c in zip(EXP3["rows"], EXP3_FALLBACK["rows"]))
check("retrieval-only fallback agrees with E2-records substrate",
      same and EXP3_FALLBACK["best"]["f1"] == b["f1"],
      "substrate '%s' == '%s'" % (EXP3["substrate"],
                                  EXP3_FALLBACK["substrate"]))
print("\n=== TABLE 3 · tau sweep (refusal precision/recall/F1) ===")
h, r = xp.tau_sweep_table(EXP3)
print(xp.md_table(h, r))
f1_default = next((row["f1"] for row in EXP3["rows"]
                   if abs(row["multiplier"] - 1.0) < 1e-9), None)
print("\nOPERATING POINT (max F1): tau = %.3f (%.2fx calibrated) · P %.3f · "
      "R %.3f · F1 %.3f | hardcoded Task 9 tau %.4f sits at F1 %.3f"
      % (b["tau"], b["multiplier"], b["precision"], b["recall"], b["f1"],
         EXP3["hardcoded_tau"],
         f1_default if f1_default is not None else float("nan")))

# ============================================================================
# Phase 4 — EXPERIMENT 4 · seed variance (baby config, 3 seeds)
# ============================================================================
phase("PHASE 4 · EXP4 seed variance (baby %d steps x seeds %s)"
      % (xp.steps_for(MODE, "seed"), list(xp.SEEDS)))
t0 = time.time()
EXP4, ROWS4 = xp.run_exp4(index, sets["answerable"],
                          corpus_path=corpora["clean"], mode=MODE,
                          ckpt_root=SMOKE_CKPTS)
exp4_wall = time.time() - t0
check("variance table has 3 distinct seeds",
      [r["seed"] for r in ROWS4] == [42, 7, 1234],
      "seeds %s · %.1fs wall" % ([r["seed"] for r in ROWS4], exp4_wall))
check("per-seed RAW vs GROUNDED rates exist",
      all(isinstance(r["raw_hallucination_rate"], float)
          and isinstance(r["grounded_hallucination_rate"], float)
          and 0.0 <= r["raw_hallucination_rate"] <= 1.0
          and 0.0 <= r["grounded_hallucination_rate"] <= 1.0 for r in ROWS4),
      " · ".join("s%d raw %.3f grn %.3f" % (r["seed"],
                                            r["raw_hallucination_rate"],
                                            r["grounded_hallucination_rate"])
                 for r in ROWS4))
s = EXP4["summary"]
check("mean ± spread computed",
      s["raw_hallucination_rate"]["mean"] is not None
      and s["raw_hallucination_rate"]["spread"] is not None
      and s["grounded_hallucination_rate"]["spread"] is not None,
      "raw %.3f ± %.3f · grounded %.3f ± %.3f (spread = max−min)"
      % (s["raw_hallucination_rate"]["mean"],
         s["raw_hallucination_rate"]["spread"],
         s["grounded_hallucination_rate"]["mean"],
         s["grounded_hallucination_rate"]["spread"]))
print("\n=== TABLE 4 · seed variance: RAW vs GROUNDED hallucination ===")
h, r = xp.seed_variance_table(EXP4)
print(xp.md_table(h, r))

# ============================================================================
# Phase 5 — resume proof: re-running run_training on a finished ckpt REUSES
# ============================================================================
phase("PHASE 5 · resume/reuse proof (exp1 clean checkpoint)")
t0 = time.time()
rerun = xp.run_training(corpora["clean"], EXP1["runs"]["clean"]["config"],
                        xp.steps_for(MODE, "ab"), xp.BASE_SEED,
                        SMOKE_CKPTS / "exp1_clean",
                        schedule=xp.schedule(MODE), label="reuse-proof")
reuse_wall = time.time() - t0
check("finished checkpoint REUSED (no retrain)",
      rerun["resumed"] and rerun["step"] == xp.steps_for(MODE, "ab")
      and rerun["final_train_loss"] is None,
      "step %d · %.2fs wall (training took %.1fs; reuse includes a full "
      "val-loss recompute, so the honest signal is resumed=True + no new "
      "training rows) · ckpt %s"
      % (rerun["step"], reuse_wall, EXP1["runs"]["clean"]["wall_s"],
         Path(rerun["ckpt"]).name))

# ============================================================================
# Phase 6 — summary.md (STUB) + payloads -> repo results/
# ============================================================================
phase("PHASE 6 · write stub results/summary.md + 4 payloads")
for name, payload in (("exp11_myth_ab.json", EXP1),
                      ("exp11_grounding.json", EXP2),
                      ("exp11_tau_sweep.json", EXP3),
                      ("exp11_seed_variance.json", EXP4)):
    p = xp.save_payload(payload, RESULTS / name)
    check("payload written: %s" % name, p.exists())

PROV = xp.provenance_block(
    mode=MODE, model_cfg=xp.model_config(MODE),
    baby_cfg=xp.baby_config(MODE), steps_ab=xp.steps_for(MODE, "ab"),
    steps_seed=xp.steps_for(MODE, "seed"), seeds=xp.SEEDS,
    seed_base=xp.BASE_SEED,
    corpora=("clean=health_qa.jsonl (%d pairs) · contaminated="
             "corpus_with_myths.jsonl (%d pairs, +%d planted myths)"
             % (n_clean, n_cont, n_cont - n_clean)),
    tau=el.ABSTAIN_TAU, results_dir=RESULTS,
    max_new_tokens=xp.MAX_NEW_TOKENS,
    extra={"index chunks": str(index["N"]),
           "eval sets": json.dumps({k: len(v) for k, v in sets.items()}),
           "test": "scripts/test_task11_experiments.py (LOCAL SMOKE proof)"})
SUMMARY_PATH = RESULTS / "summary.md"
body = xp.write_summary_md(
    SUMMARY_PATH, mode=MODE,
    results={"exp1": EXP1, "exp2": EXP2, "exp3": EXP3, "exp4": EXP4},
    provenance=PROV)
check("summary.md written", SUMMARY_PATH.exists(),
      "%d chars" % len(body))
check("summary marked STUB", "STUB RUN" in body and "stub_or_real" in body
      and "| stub_or_real | stub |" in body)
check("summary has all four experiment tables",
      all(("## Experiment %d" % i) in body for i in (1, 2, 3, 4)))
check("summary has the one-line answer + operating point + noise floor",
      "Does grounding reduce hallucination" in body
      and "Operating point" in body and "noise floor" in body)
check("summary has non-expert notes + provenance",
      "Notes for non-experts" in body and "## Provenance" in body)
print("\n" + "-" * 78)
print(body)
print("-" * 78)

# ============================================================================
# Phase 7 — REGRESSIONS: task1/3/4 (workspace) + bridge/retrieval (repo) + tensor_py
# ============================================================================
phase("PHASE 7 · regressions (task1/3/4 workspace · bridge/retrieval repo · tensor_py)")
REG = [
    ("node task1 tensor engine",
     ["node", str(WORKSPACE_SCRIPTS / "test_task1_tensor.js")], "ALL TESTS PASSED"),
    ("node task3 tokenizer",
     ["node", str(WORKSPACE_SCRIPTS / "test_task3_tokenizer.js")], "PARITY 200/200 PASS"),
    ("node task4 model",
     ["node", str(WORKSPACE_SCRIPTS / "test_task4_model.js")], "ALL TESTS PASSED"),
    ("node task7 bridge",
     ["node", str(REPO / "scripts" / "test_task7_bridge.js")], "ALL TESTS PASSED"),
    ("node task8 retrieval",
     ["node", str(REPO / "scripts" / "test_task8_retrieval.js")], "PASS"),
    ("python3 tensor_py",
     [sys.executable, str(REPO / "colab" / "test_tensor_py.py")], "PASS"),
]
reg_ok = True
for label, cmd, expect in REG:
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    hit = expect in out and r.returncode == 0
    reg_ok = reg_ok and hit
    check("regression: %s" % label, hit,
          "rc=%d · %.0fs · marker %r" % (r.returncode, time.time() - t0, expect))
    if not hit:
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-6:]
        for ln in tail:
            print("      | " + ln)

# ============================================================================
# verdict
# ============================================================================
total_min = (time.time() - T_START) / 60.0
print("\n" + "=" * 78)
print("TOTAL WALL: %.1f min (budget: 12 min experiments + regressions)"
      % total_min)
if FAILED:
    print("TASK11 LOCAL SMOKE: FAILURES (%d): %s" % (len(FAILED), FAILED))
    sys.exit(1)
print("TASK11 LOCAL SMOKE: ALL CHECKS PASSED — every code path of the "
      "experiment battery proven on the tiny config; stub summary.md + "
      "payloads written to %s" % RESULTS)
sys.exit(0)
