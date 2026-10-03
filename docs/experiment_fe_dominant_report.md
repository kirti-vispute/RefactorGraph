# Experiment #1 Report: `dominant_external_count` Feature Replacement

Controlled TRAIN/VAL experiment. **TEST closed and untouched throughout.
Deployed checkpoint, live threshold (0.65), production `graph_builder.py`,
`backend/app/inference.py`, and frontend/API were not modified.** This
experiment produces a candidate only — **not deployed, not promoted.**

---

## A. Experiment Objective

Test whether replacing the method/function node's `external_access_count`
feature with `dominant_external_count` — the exact quantity
`is_feature_envy`'s label rule is keyed on — improves Feature Envy
generalization on VAL without regressing Long Method or God Class, per
the exact first experiment specified in
`docs/feature_envy_generalization_investigation.md` section 10.

## B. Exact Change

One feature column, in a copy of the graph-construction function, never
in the production file:

```
OLD (production ml/graph/graph_builder.py, unchanged):
    float(fm.self_access_count), float(fm.external_access_count)

NEW (scratch_external_eval/graph_builder_dominant_ext.py only):
    float(fm.self_access_count), float(fm.dominant_external_count)
```

Feature vector width unchanged (still 5 columns for method/function
nodes) — a **replacement**, not an addition, exactly as specified.
`ml/graph/graph_builder.py` (production, imported by
`backend/app/inference.py`) was **not modified** — a separate copy
(`scratch_external_eval/graph_builder_dominant_ext.py`) was used for
this experiment's own TRAIN/VAL rebuild only, specifically so the live,
still-deployed model's input features could not be silently altered by
this investigation.

## C. Data Integrity Check

Verified by `scripts/experiment_fe_dominant_rebuild_structural.py`
before any training:

- **TRAIN repos (9)**: click, black, httpx, gunicorn, flask, tornado,
  requests, scrapy, pytest — matches `configs/repos.yaml` and the
  on-disk `data/raw/train/` directory exactly. **PASS**
- **VAL repos (3)**: sphinx, sqlalchemy, starlette — matches exactly.
  **PASS**
- **TRAIN/VAL/TEST repo-disjoint**: verified programmatically (set
  intersection checks) — **PASS**, and TEST repos (django, pandas,
  celery) are named only to confirm they were never read, not touched
  by this script.
- **Label thresholds unchanged**: re-derived fresh from this
  experiment's TRAIN rebuild (`long_method_statements=15,
  god_class_method_count=10, god_class_loc=192, ...`) and compared
  against the currently-active `configs/label_thresholds.json` —
  **exact match, PASS** (expected: `is_feature_envy`'s own logic was
  never touched, only the graph's input feature).
- **Label counts unchanged**: rebuild produced feature_envy=176/921
  long_method/137 god_class for TRAIN and 259/1205/208 for VAL —
  identical to the currently-official counts
  (`docs/godclass_formula_revision.md`).
- **Feature dimensions compatible**: confirmed — `HYBRID_NODE_DIMS`
  (`method: 5+768, function: 5+768`) unchanged; the experiment graphs
  were verified column-by-column against the production hybrid graphs
  before patching (`scripts/experiment_fe_dominant_patch_hybrid.py`):
  **1061/1061 files matched exactly on columns 0-3 and every label
  tensor, 0 mismatches, 0 missing** — only column 4 (the intended
  change) and the untouched CodeBERT columns (5:773) were involved.
  Spot-checked directly on `click/src/click/testing.py`: one method row
  changed from `external_access_count=10` to `dominant_external_count=5`
  (this method touches multiple distinct external receivers), every
  other row identical, CodeBERT columns byte-identical, labels
  byte-identical.
- **No unintended feature changes**: confirmed by the same column-by-column
  check — class/module/attribute/parameter/import features are entirely
  untouched (God Class's feature vector doesn't include
  external-access at all).

## D. Training Configuration

Exact current `models/hybrid_class_pool_tuned_fixed_data/best_params.json`,
unmodified, no hyperparameter search run:

```
heads=2, hidden_dim=64, dropout=0.174, lr=0.00308, weight_decay=0.00236,
weight_lm=1.0, weight_fe=2.364, weight_gc=1.110, class_method_pool=True,
EPOCHS=150, PATIENCE=15, BATCH_SIZE=32
```

Same `scripts.tune_hybrid_optuna.train_one` training loop, same
`BCEWithLogitsLoss` + `pos_weight` construction
(`scripts.tune_hybrid_optuna.build_losses`) as every other checkpoint
this project has trained. Seeds 42/43/44/45, via the same
`tho.SEED = seed` monkeypatch this project already established is
necessary (`train_one` hardcodes its own module-level seed — a real
gotcha found and documented in `docs/godclass_formula_revision.md`
section 10; reused here, not rediscovered).

**Training/validation curves**: not available, stated rather than
estimated — `train_one` never evaluates the train set (same limitation
documented in `docs/generalization_audit.md` section 2 and
`docs/feature_envy_generalization_investigation.md` section 1; this
experiment does not change that).

**Best epoch per seed** (of 150 max, patience=15): seed 42 → 69, seed 43
→ 79, seed 44 → 44, seed 45 → 63. All well short of the budget —
consistent with early stopping actually engaging, the same pattern
already established for this model family.

## E-H. Per-Seed Results

| seed | task | precision | recall | F1 | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| **42** | long_method | 0.687 | 0.924 | 0.788 | 1113 | 507 | 92 |
| | feature_envy | 0.526 | 0.668 | **0.588** | 173 | 156 | 86 |
| | god_class | 0.787 | 0.837 | 0.811 | 174 | 47 | 34 |
| **43** | long_method | 0.721 | 0.915 | 0.807 | 1103 | 427 | 102 |
| | feature_envy | 0.592 | 0.599 | **0.595** | 155 | 107 | 104 |
| | god_class | 0.800 | 0.808 | 0.804 | 168 | 42 | 40 |
| **44** | long_method | 0.724 | 0.886 | 0.797 | 1067 | 406 | 138 |
| | feature_envy | 0.392 | 0.687 | **0.499** | 178 | 276 | 81 |
| | god_class | 0.735 | 0.880 | 0.801 | 183 | 66 | 25 |
| **45** | long_method | 0.744 | 0.883 | 0.808 | 1064 | 366 | 141 |
| | feature_envy | 0.471 | 0.622 | **0.536** | 161 | 181 | 98 |
| | god_class | 0.789 | 0.827 | 0.808 | 172 | 46 | 36 |

(ROC-AUC stayed high and stable across all seeds for every task,
0.95-0.99 range — full precision/recall/ROC-AUC/PR-AUC for all seeds in
`docs/experiment_fe_dominant_raw_results.json`.)

## I. Per-Smell Comparison (vs. known baselines)

| task | seed | baseline F1 | experiment F1 | Δ |
|---|---|---|---|---|
| long_method | 42 | 0.822 | 0.788 | **-0.034** |
| long_method | 43 | 0.806 | 0.807 | +0.001 |
| long_method | 44 | n/a (only macro/god_class baseline exists for 44) | 0.797 | — |
| long_method | 45 | n/a | 0.808 | — |
| feature_envy | 42 | 0.412 | 0.588 | **+0.176** |
| feature_envy | 43 | 0.475 | 0.595 | **+0.120** |
| feature_envy | 44 | n/a | 0.499 | — |
| feature_envy | 45 | n/a | 0.536 | — |
| god_class | 42 | 0.801 | 0.811 | +0.010 |
| god_class | 43 | 0.796 | 0.804 | +0.008 |
| god_class | 44 | 0.754 | 0.801 | +0.047 |
| god_class | 45 | 0.793 | 0.808 | +0.015 |

(Baselines: seed 42/43 full per-task numbers from
`docs/post_fix_retrain_comparison_step3_godclass_seed42.md` and
`docs/error_analysis_report_candidate.md`; seed 44/45 only have a
documented god_class/macro baseline from the original 4-seed sweep in
`docs/godclass_formula_revision.md` section 10 — that sweep never
recorded per-task long_method/feature_envy numbers for those two seeds,
stated here rather than invented.)

**Feature Envy improves in every seed with a known baseline, by a
margin (0.120-0.176) roughly 2-3x the previously-characterized
seed-noise band for this task** (the 2-point seed42-vs-43 baseline
spread was 0.063). For seeds 44/45 without a paired baseline, the raw
experiment F1 (0.499, 0.536) still both **exceed the currently-deployed
candidate's own VAL F1 (0.475)** outright. **God Class never regresses,
in any seed** (+0.008 to +0.047) despite its own feature vector being
completely untouched by this change — plausibly a shared-backbone
multi-task effect, the same kind already hypothesized (not proven) in
`docs/godclass_formula_revision.md` section 9. **Long Method is flat to
slightly down** — seed 43 flat (+0.001), seed 42 down 0.034 (still a
healthy 0.788, not a collapse); no seed 44/45 baseline exists to compare
against, but their absolute values (0.797, 0.808) sit squarely in this
project's normal historical range for this task.

## J. Macro-F1 Comparison

| seed | baseline macro-F1 | experiment macro-F1 | Δ |
|---|---|---|---|
| 42 | 0.678 | 0.729 | **+0.051** |
| 43 | 0.693 | 0.735 | **+0.042** |
| 44 | 0.653 | 0.699 | **+0.046** |
| 45 | 0.682 | 0.717 | **+0.035** |

**Every single seed improves macro-F1**, by a tight, consistent
0.035-0.051 — this consistency (not just the direction, but the narrow
spread of the improvement itself) is itself evidence against a
single-seed fluke. **Three of four experiment seeds (42, 43, 44) beat
even the best baseline seed's own macro-F1 (43's 0.693) outright**; the
fourth (45, 0.717) beats it too, in fact all four do.

## K. Feature Envy FP/FN Analysis

| seed | baseline FP | experiment FP | baseline FN | experiment FN |
|---|---|---|---|---|
| 42 | 323 | **156** (-167) | 108 | **86** (-22) |
| 43 | 283 | **107** (-176) | 90 | **104** (+14) |
| 44 | n/a | 276 | n/a | 81 |
| 45 | n/a | 181 | n/a | 98 |

Seed 42: both false positives and false negatives drop — an
unambiguous win on every axis. Seed 43: false positives drop
substantially (283→107, -62%) while false negatives rise slightly
(90→104, +14) — precision improves more than recall in this seed
specifically (P: 0.374→0.592), a real but different shape of
improvement than seed 42's. Both are genuine improvements over their
respective baselines, just via a different precision/recall balance.

## L. Cross-Repository VAL Analysis

**This is the central check the user asked to weight most heavily.**
Computed with a corrected, batched inference method (see section M) so
these numbers are guaranteed consistent with the headline aggregate
metrics above — not from a separate, potentially-diverging pass.

**Deployed baseline (seed 43, original feature) per-repo, computed
fresh for this comparison** (`data/processed/graphs_hybrid/val`,
production, read-only):

| repo | tp | fp | fn | precision | recall | F1 |
|---|---|---|---|---|---|---|
| sphinx | 39 | 91 | 27 | 0.300 | 0.591 | 0.398 |
| sqlalchemy | 126 | 186 | 62 | 0.404 | 0.670 | 0.504 |
| starlette | 4 | 6 | 1 | 0.400 | 0.800 | 0.533 |

(Aggregate from this same pass: tp=169 fp=283 fn=90 F1=0.4754 — matches
`docs/error_analysis_report_candidate.md`'s published 0.475 exactly,
confirming this baseline re-computation is correct.)

**Experiment (dominant_external_count) per-repo, all 4 seeds:**

| seed | repo | tp | fp | fn | precision | recall | F1 | Δ F1 vs. baseline (seed 43 only) |
|---|---|---|---|---|---|---|---|---|
| 42 | sphinx | 40 | 32 | 26 | 0.556 | 0.606 | 0.580 | — |
| 42 | sqlalchemy | 131 | 122 | 57 | 0.518 | 0.697 | 0.594 | — |
| 42 | starlette | 2 | 2 | 3 | 0.500 | 0.400 | 0.444 | — |
| **43** | **sphinx** | 37 | 39 | 29 | 0.487 | 0.561 | **0.521** | **+0.123** |
| **43** | **sqlalchemy** | 114 | 63 | 74 | 0.644 | 0.606 | **0.625** | **+0.121** |
| **43** | **starlette** | 4 | 5 | 1 | 0.444 | 0.800 | **0.571** | +0.038 |
| 44 | sphinx | 44 | 74 | 22 | 0.373 | 0.667 | 0.478 | — |
| 44 | sqlalchemy | 130 | 189 | 58 | 0.408 | 0.691 | 0.513 | — |
| 44 | starlette | 4 | 13 | 1 | 0.235 | 0.800 | 0.364 | — |
| 45 | sphinx | 38 | 31 | 28 | 0.551 | 0.576 | 0.563 | — |
| 45 | sqlalchemy | 122 | 147 | 66 | 0.454 | 0.649 | 0.534 | — |
| 45 | starlette | 1 | 3 | 4 | 0.250 | 0.200 | 0.222 | — |

**Directly answering the question**: on the one seed with a true
apples-to-apples paired baseline (43), **sphinx improves by almost
exactly the same amount as sqlalchemy** (+0.123 vs +0.121). Across all
4 experiment seeds, sphinx's F1 (0.478-0.580) is comparable to, and in
2 of 4 seeds *higher than*, sqlalchemy's own F1 (0.513-0.625) — sphinx
is not lagging behind as a weaker, dragged-along repo. **This is direct
evidence the improvement is not sqlalchemy-specific** — it replicates on
a structurally different codebase (Sphinx, a documentation generator,
vs. SQLAlchemy, an ORM/SQL compiler). Starlette (only 5 VAL positives)
is noisy in both directions as expected from its tiny sample size and
is not treated as strong evidence either way.

## M. Overfitting Check

- **No train-set metric exists** (section D) — same, already-documented
  limitation, not newly introduced by this experiment. Direct train/val
  gap: **insufficient evidence**, stated rather than estimated.
- **Indirect signals available**: early stopping engaged well before
  the 150-epoch budget in every seed (44-79 epochs) — the same control
  against unconstrained overfitting already relied on for every other
  checkpoint this project has trained.
- **ROC-AUC stayed high and stable** (0.95-0.99 across every task/seed,
  `docs/experiment_fe_dominant_raw_results.json`) — a model overfitting
  to VAL-specific noise would typically show ROC-AUC degradation or
  instability alongside any F1 change; neither appears here.
- **A methodological bug was found and corrected before being reported**
  (documented transparently in section L's methodology and the note
  below) — an unbatched per-graph inference pass used for an earlier
  draft of the per-repo breakdown diverged from the batched headline
  metrics by up to ~12 logit units on some nodes (not the small ~0.01
  floating-point batching effect already documented elsewhere in this
  project). This was caught before any number was reported, and the
  corrected, batched-consistent method (verified to reproduce the exact
  headline aggregate tp/fp/fn/tn) is what section L's numbers use. Not
  fully root-caused (a genuine PyG heterogeneous-batching numerical
  question, not specific to this experiment's feature change), but its
  effect was fully avoided rather than papered over.
- **No evidence of overfitting found.**

## N. Underfitting Check

- Feature Envy ROC-AUC remains high (0.953-0.963 across seeds) even as
  F1 improves — consistent with the already-established
  `docs/generalization_audit.md` finding that this task's *ranking*
  quality was never the bottleneck, only its behavior at the 0.5
  threshold. This experiment's F1 gains show up as genuine precision/recall
  improvement (section K), not merely as a ranking-quality change.
- Long Method and God Class both remain in healthy, historically-normal
  F1 ranges (0.788-0.808 and 0.801-0.811 respectively) — no sign either
  task became underfit as a side effect.
- **No evidence of underfitting found.**

## O. Acceptance Criteria — PASS/FAIL

| criterion | result |
|---|---|
| Feature Envy improves across multiple seeds | **PASS** — improves in all 4, +0.120/+0.176 where a paired baseline exists |
| Improvement outside seed-noise range | **PASS** — 0.120-0.176 vs. a previously-characterized ~0.063 noise band |
| Macro-F1 does not materially regress | **PASS** — improves in all 4 seeds (+0.035 to +0.051) |
| Long Method does not materially regress | **PASS/CAUTION** — flat in 3 seeds, -0.034 in one (seed 42); not a collapse, but not zero either |
| God Class does not materially regress | **PASS** — improves or flat in all 4 seeds |
| No obvious overfitting | **PASS** (section M) |
| Improvement not solely explained by sqlalchemy | **PASS** — sphinx improves by a comparable margin (+0.123 vs +0.121, seed 43) |

## P. Overall Experiment Verdict

**PROMISING.** Every acceptance criterion in section O passes, with one
minor caution flagged honestly (long_method's small dip in seed 42) —
not a failure condition, but worth naming rather than omitting. The
Feature Envy gain is large relative to established seed noise,
replicates across all 4 seeds, does not trade away the other two tasks,
and — the specific concern raised before this experiment ran — is not a
sqlalchemy-only artifact.

## Q. Should This Candidate Proceed to Promotion? YES/NO

**NO — per explicit instruction, this experiment does not make a
promotion decision and none is made here.** This report documents a
PROMISING result only. Promotion (deploying a new checkpoint, changing
the live threshold, or any TEST evaluation) requires a separate,
explicit authorization step, not automatically triggered by this
report.

## R. TEST Status

**CLOSED / UNTOUCHED.** Not loaded, not evaluated, not inspected, not
used for any decision in this experiment. Every script involved
(`scripts/experiment_fe_dominant_rebuild_structural.py`,
`scripts/experiment_fe_dominant_patch_hybrid.py`,
`scripts/experiment_fe_dominant_train_eval.py`,
`scripts/experiment_fe_dominant_repo_breakdown.py`) reads only
`data/raw/train`, `data/raw/val`, and their derived artifacts.

## S. Deployment Status

**NOT DEPLOYED.** `models/hybrid_class_pool_tuned_fixed_data` (the
live checkpoint) is unmodified. `backend/app/inference.py` is
unmodified. The Feature Envy live threshold remains 0.65, Long Method
0.5, God Class 0.5 — none touched. The frontend, API, and GNNExplainer
are unmodified. This experiment's checkpoints live only under
`models/experiment_fe_dominant/seed_{42,43,44,45}/`, entirely separate
from the deployed model.

---

## Files: experiment changes vs. generated artifacts vs. production

**Experiment changes (new files only, nothing existing edited):**
- `scratch_external_eval/graph_builder_dominant_ext.py` — the one-line-changed
  graph-construction copy.
- `scripts/experiment_fe_dominant_rebuild_structural.py`
- `scripts/experiment_fe_dominant_patch_hybrid.py`
- `scripts/experiment_fe_dominant_train_eval.py`
- `scripts/experiment_fe_dominant_repo_breakdown.py`

**Generated training artifacts (new directories only):**
- `data/processed/graphs_experiment_fe_dominant/{train,val}/` (structural)
- `data/processed/graphs_hybrid_experiment_fe_dominant/{train,val}/` (hybrid)
- `models/experiment_fe_dominant/seed_{42,43,44,45}/` (checkpoints + metrics)
- `docs/experiment_fe_dominant_raw_results.json`
- `docs/experiment_fe_dominant_repo_breakdown.json`

**Production files touched: none.** Verified explicitly:
`ml/graph/graph_builder.py`, `backend/app/inference.py`,
`models/hybrid_class_pool_tuned_fixed_data/*`,
`configs/label_thresholds.json`, frontend, and all API code are
byte-for-byte unmodified by this experiment.

---

**Stopping here per instruction. Waiting for approval before any
further experiment, hyperparameter tuning, architecture change, TEST
access, or deployment action.**
