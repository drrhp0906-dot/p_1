# p_1 · experiment summary (Task 11)

**Does grounding reduce hallucination in a model we built from scratch?**

**NOT IN THIS RUN — GROUNDED hallucination rate 0.017 came out ABOVE RAW 0.000 (small-n noise or an undertrained model; see the notes).**

> ⚠ **STUB RUN (LOCAL_SMOKE = True)** — tiny configs, few steps. This file proves every code path and the table SHAPES; the numbers come from undertrained toys and carry no scientific meaning. Re-run the notebook with LOCAL_SMOKE = False in Colab for the real battery.

## Experiment 1 · myth contamination A/B

Two siblings, identical seed (42) and schedule, differing ONLY in the training corpus (clean vs +300 planted myth pairs). Parroting = the model echoes the folk belief on the 20 myth-trap questions (eval_lib.mythVerdict; RAW mode).

| model | corpus | parroted (of 20) | parroting rate | example quote (RAW) |
|---|---|---|---|---|
| clean | health_qa.jsonl | 0 | 0.0% | — |
| contaminated | corpus_with_myths.jsonl | 0 | 0.0% | — |

Expected direction (contaminated ≥ clean): **holds**. Smoke run: 150 steps on a tiny model — both brains are near-babble, so rates are noisy; we only assert the MEASUREMENT exists (see notes for non-experts).

Per-mode parroting rates (context):

| mode | clean parroting | contaminated parroting |
|---|---|---|
| raw | 0.0% | 0.0% |
| grounded | 20.0% | 20.0% |
| verified | 0.0% | 0.0% |
| abstaining | 10.0% | 10.0% |

## Experiment 2 · grounding delta (the headline)

| eval set | mode | hallucination rate | refusal rate | parroting rate |
|---|---|---|---|---|
| answerable | raw | 0.000 | 0.0% | — |
| answerable | grounded | 0.017 | 13.3% | — |
| answerable | verified | 0.000 | 0.0% | — |
| answerable | abstaining | 0.012 | 58.3% | — |
| mythtraps | raw | 0.000 | 0.0% | 0.0% |
| mythtraps | grounded | 0.090 | 10.0% | 20.0% |
| mythtraps | verified | 0.000 | 0.0% | 0.0% |
| mythtraps | abstaining | 0.071 | 80.0% | 10.0% |
| unanswerable | raw | 0.000 | 0.0% | — |
| unanswerable | grounded | 0.028 | 5.0% | — |
| unanswerable | verified | 0.000 | 0.0% | — |
| unanswerable | abstaining | 0.000 | 95.0% | — |

eval_lib headline: _grounding cuts hallucination rate 0.000 -> 0.017 (verified 0.000, abstaining 0.012); abstention refuses 95% of unanswerables; myth-parroting raw 0% vs grounded 20%_

## Experiment 3 · abstention threshold sweep

| ×calibrated | tau | true refusals (unans.) | false refusals (ans.) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.20 | 4.009 | 0/20 | 0/60 | 0.000 | 0.000 | 0.000 |
| 0.30 | 6.014 | 0/20 | 0/60 | 0.000 | 0.000 | 0.000 |
| 0.40 | 8.018 | 0/20 | 0/60 | 0.000 | 0.000 | 0.000 |
| 0.50 | 10.023 | 2/20 | 1/60 | 0.667 | 0.100 | 0.174 |
| 0.60 | 12.027 | 7/20 | 1/60 | 0.875 | 0.350 | 0.500 |
| 0.70 | 14.032 | 12/20 | 4/60 | 0.750 | 0.600 | 0.667 |
| 0.80 ← best F1 | 16.036 | 18/20 | 12/60 | 0.600 | 0.900 | 0.720 |
| 0.90 | 18.041 | 19/20 | 19/60 | 0.500 | 0.950 | 0.655 |
| 1.00 | 20.045 | 19/20 | 30/60 | 0.388 | 0.950 | 0.551 |
| 1.10 | 22.050 | 19/20 | 40/60 | 0.322 | 0.950 | 0.481 |
| 1.25 | 25.057 | 20/20 | 43/60 | 0.317 | 1.000 | 0.482 |
| 1.50 | 30.068 | 20/20 | 50/60 | 0.286 | 1.000 | 0.444 |

**Operating point: max F1 = 0.720 at tau ≈ 16.036 (0.80× the calibrated median 20.045).** Task 9's hardcoded tau 20.0454 (calibration match: yes) sits at F1 0.551 — it buys recall but refuses 12/60 answerable questions.

## Experiment 4 · seed variance

| seed | train wall (s) | RAW halluc rate | GROUNDED halluc rate | Δ (raw − grounded) |
|---|---|---|---|---|
| 42 | 63 | 0.000 | 0.017 | -0.017 |
| 7 | 63 | 0.000 | 0.017 | -0.017 |
| 1234 | 65 | 0.000 | 0.017 | -0.017 |
| RAW mean ± spread | — | 0.000 ± 0.000 | 0.000 ± 0.000 | spread = max−min |
| GROUNDED mean ± spread | — | 0.017 ± 0.000 | 0.017 ± 0.000 | spread = max−min |

Reading: the RAW column moves by ±0.000 across seeds (mean 0.000) and the GROUNDED column by ±0.000 (mean 0.017) — that is the measurement noise floor any SINGLE-seed number should be read against. (spread = max − min over seeds (n=3); std = population)

## Notes for non-experts

- **Hallucination rate** = the share of the model's answer statements that our rule-based verifier could NOT find support for in the retrieved corpus. It is a lexical yardstick, not ground truth — but it is the SAME yardstick for every mode, so comparisons are fair.
- **RAW** answers come straight from the model; **GROUNDED** answers are assembled only from retrieved corpus sentences (with citations); **VERIFIED** checks each RAW claim and rewrites or flags failures; **ABSTAINING** additionally refuses when retrieval looks too weak (score below tau).
- **Myth-parroting** = answering a folk-belief trap the way the myth wants it answered. The A/B siblings differ ONLY in training data — if contamination moves the parroting rate, corpus hygiene matters.
- **The tau sweep** trades "refusing things we could have answered" (false refusals) against "answering things we should have refused" (missed refusals). F1 balances both; the sweep shows the Task 9 median-tau default is recall-heavy.
- **Seed variance** says how much of any number here is luck of the draw: with only 3 seeds the spread is a rough noise floor, not a confidence interval.
- An undertrained model (like the stub run) babbles: many claims score near zero and land in the verifier's "not-in-corpus" bucket, which is EXCLUDED from the hallucination rate — so read stub numbers as plumbing, not psychology.

## Provenance

| field | value |
|---|---|
| A/B seed (E1) | 42 |
| E1/E2 steps | 150 |
| E4 steps per seed | 60 |
| abstain tau (Task 9) | 20.0454 (calibrated median top-1 BM25) |
| baby config (E4) | {"n_layer": 2, "n_head": 2, "d_model": 64, "d_ff": 128, "ctx": 128, "vocab_size": 4096} |
| corpora | clean=health_qa.jsonl (4278 pairs) · contaminated=corpus_with_myths.jsonl (4578 pairs, +300 planted myths) |
| date (UTC) | 2026-10-22 06:19:52 |
| eval sets | {"answerable": 60, "mythtraps": 20, "unanswerable": 20} |
| eval_lib | 1.0.0 |
| experiments_lib | 1.0.0 |
| index chunks | 4278 |
| max new tokens | 80 |
| model config (E1/E2) | {"n_layer": 2, "n_head": 2, "d_model": 64, "d_ff": 128, "ctx": 128, "vocab_size": 4096} |
| results dir | /home/z/my-project/download/p_1/results |
| run mode | LOCAL_SMOKE (stub) |
| seeds (E4) | [42, 7, 1234] |
| stub_or_real | stub |
| test | scripts/test_task11_experiments.py (LOCAL SMOKE proof) |
| train schedule | {"lr": 0.001, "warmup": 20, "batch_size": 8} |

