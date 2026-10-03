# Generic Model Generalization Audit

Repository-wide, evidence-based audit of the deployed candidate
(`models/hybrid_class_pool_tuned_fixed_data`, Design B, seed=43). No
files modified, no retraining, no TEST evaluation performed as part of
this audit — every number below is quoted from artifacts that already
existed before this audit started. Where no artifact exists, this
document says "insufficient evidence" rather than estimating.

---

## 1. Executive Verdict

| Question | Verdict |
|---|---|
| Overfitting | **LOW** |
| Underfitting | **LOW-MEDIUM** (Feature Envy specifically; other two tasks LOW) |
| Generalization | **MODERATE-GOOD** |
| Data leakage risk | **LOW** |
| Feature Envy weakness | **MEDIUM** (systematic, class-imbalance + domain-coverage driven, not a bug) |

**DO NOT RETRAIN — current evidence does not justify it beyond what has
already been done this project cycle.** No evidence of overfitting,
memorization, or leakage was found. Feature Envy's weakness is real but
already understood (severe class imbalance + boundary sensitivity, not a
newly-discovered defect) and was already the deliberate target of the
seed-selection step that produced this exact checkpoint (TEST feature_envy
F1 0.412→0.525, the largest gain of the three tasks). See section 12 for
what a future, evidence-gated retrain would need to show.

---

## 2. Train vs Validation Analysis

**No train-set metric has ever been computed for the deployed architecture.**
Verified by reading `scripts/tune_hybrid_optuna.py::train_one` (lines
108–140, the exact training loop used for every `hybrid_class_pool_*`
checkpoint including the deployed candidate): each epoch runs one pass
over `train_loader` for gradient updates, then calls `run_eval(model,
val_loader, device)` — there is no corresponding call on `train_loader`.
No training-set precision/recall/F1 has ever been computed or saved for
this model family. Per-epoch history is not persisted to disk for
`hybrid_class_pool_tuned` or `hybrid_class_pool_tuned_fixed_data` — only
the final best-epoch VAL metrics (`final_val_metrics.json`) and the
best epoch number (`best_params.json`) survive.

**This means a direct train-vs-val gap cannot be computed for the actual
deployed model.** This is reported as insufficient evidence, not
estimated.

**Indirect, historical proxy** (`models/hybrid_baseline/history.json`,
134 epochs, a *different*, earlier, non-Design-B architecture trained on
an earlier version of this dataset — NOT the deployed candidate,
included only because it is the one artifact in this repo that actually
logged both training loss and per-epoch val F1 together):

| epoch | train_loss | val_combined_f1 |
|---|---|---|
| 1 | 4.330 | 0.233 |
| 21 | 1.365 | 0.514 |
| 41 | 0.654 | 0.567 |
| 61 | 0.590 | 0.566 |
| 81 | 0.324 | 0.606 |
| 101 | 0.298 | 0.631 |
| 121 | 0.258 | 0.650 |
| 134 (last) | 0.238 | 0.639 |

Train loss falls monotonically toward zero through 134 epochs, while val
F1 rises through roughly epoch 60–70, then plateaus and oscillates
(0.61–0.65) for the remaining ~60 epochs without further improvement —
the classic shape of a model that keeps fitting the training set while
validation gains have already exhausted (best epoch 119/134, val F1
0.651). It does **not** show val *degrading* (no collapse below the
plateau), so this specific historical run does not show severe
overfitting, but the shape is consistent with mild overfitting risk once
training runs unconstrained past the point of val improvement.

**Why this matters for the deployed candidate, and why the risk is
lower there**: the actual `train_one` loop used for the deployed
candidate has `PATIENCE=15` early stopping (`scripts/tune_hybrid_optuna.py`
constant, used via `scripts/retrain_fixed_data_hybrid_class_pool.py` and
`scripts/promote_god_class_retrain_seed43.py`, `EPOCHS=150`), and the
saved state is `best_state` at `best_epoch` — the epoch with the highest
val macro-F1 — not the final epoch's weights. Observed `best_epoch`
values across the seed sweep (`docs/godclass_formula_revision.md`
section 10): seed 42→(from the original retrain run) not separately
logged beyond final metrics; seed 43→82, seed 44→41, seed 45→43, all
well short of the 150-epoch budget. This early-stopping design is a
structural control specifically against the plateau-then-drift pattern
seen in the `hybrid_baseline` proxy above — the deployed candidate's
weights come from the val-best checkpoint, not a fully-converged/possibly-drifted
final epoch.

**Per-task classification** (using best available evidence — VAL-best
metrics only, since no train metric exists):

| task | evidence | classification |
|---|---|---|
| Long Method | VAL F1=0.806 (candidate, `docs/error_analysis_report_candidate.md`), TEST F1=0.740 (`docs/final_test_evaluation_report.md`) — a 0.066 VAL→TEST gap, in the expected direction (VAL was model-selection surface, TEST wasn't) | **likely reasonable/generalizing** — gap is modest and directionally expected, not a red flag |
| Feature Envy | VAL F1=0.475, TEST F1=0.525 — TEST is *higher* than VAL here | **likely reasonable/generalizing** — no overfitting signature (a memorizing model would show VAL≫TEST, not the reverse) |
| God Class | VAL F1=0.796, TEST F1=0.767 — 0.029 gap | **likely reasonable/generalizing** — gap is within the seed-noise band already characterized (section 3) |

No task shows "very high train performance, much lower val" (can't be
measured) or "validation degrading while training improves" (the one
available proxy curve doesn't show this either). No task shows
persistent low performance on both — Feature Envy is the weakest, but
it is not near-zero (VAL/TEST F1 0.475/0.525, ROC-AUC 0.952 on TEST) —
see section 6.

---

## 3. Seed Stability

**Source**: `scratch_external_eval/godclass_seed_sweep.py` (this
project's own multi-seed VAL sweep, already run — not a new experiment
for this audit), documented in `docs/godclass_formula_revision.md`
section 10. **Only 4 seeds were sampled (42, 43, 44, 45)** — not a
statistically large sample; treat spread estimates as indicative, not
precise confidence intervals.

| seed | god_class F1 | macro-F1 | PipSession prob (external, unseen-repo case) |
|---|---|---|---|
| 42 | 0.801 | 0.678 | 0.231 |
| 43 (selected) | 0.796 | **0.693 (best)** | 0.994 |
| 44 | 0.754 (lowest) | 0.653 (lowest) | 0.651 |
| 45 | 0.793 | 0.682 | 0.838 |

- **god_class F1 range**: 0.754–0.801 (spread 0.047 across 4 seeds).
  Seed 44 is the visible outlier on the low end for both god_class F1
  and macro-F1 simultaneously; no seed is an outlier on the *high* end
  by more than the general spread.
- **Seed selection**: seed 43 was chosen for the best macro-F1 among the
  4 sampled seeds (0.693), a holistic, all-3-tasks criterion — not
  because it produced the best god_class number (0.796, actually
  *second-lowest* of the four) and not because of the PipSession number
  specifically (documented explicitly in
  `docs/godclass_formula_revision.md` section 10 as a side observation,
  not the selection criterion). This is a reasonably-chosen seed by the
  audit's own re-reading of that record.
- **Per-task (long_method, feature_envy) seed variance**: **insufficient
  evidence beyond 2 data points.** The 4-seed sweep only recorded
  `god_class_f1` and `macro_f1` per seed (confirmed by reading
  `scratch_external_eval/godclass_seed_sweep.py`'s print statements —
  it does not print or persist per-task long_method/feature_envy F1 for
  seeds 44/45, and the underlying model states for those runs were not
  saved to disk). What exists: seed 42's full run
  (`docs/post_fix_retrain_comparison_step3_godclass_seed42.md`:
  long_method=0.822, feature_envy=0.412) vs seed 43's full run
  (`docs/post_fix_retrain_comparison.md`: long_method=0.806,
  feature_envy=0.475) — a 2-point comparison, not a 4-seed one, for
  these two tasks. Feature Envy's 2-point spread (0.412→0.475, Δ=0.063)
  is smaller than god_class's 4-seed spread (0.047) but on a smaller
  sample, so this is not a strong claim either way.
- **Which task is "especially unstable"**: on the evidence that exists,
  **God Class** is the only task with a documented 4-seed spread; it is
  moderate (0.047), not extreme (no seed is 2x or half of another). No
  task shows a single seed producing a qualitatively different result
  (e.g., near-zero F1) — the instability that *is* documented is at the
  level of individual out-of-distribution examples (PipSession
  0.231–0.994) more than at the level of aggregate VAL metrics.

---

## 4. Repository Separation Audit

**Source**: `configs/repos.yaml` (read directly for this audit).

- **Train (9 repos)**: click, black, httpx, gunicorn, flask, tornado,
  requests, scrapy, pytest.
- **Val (3 repos)**: sqlalchemy, starlette, sphinx.
- **Test (3 repos)**: django, pandas, celery.
- **Overlap check**: all 15 repository names are distinct across the
  three lists — zero repository appears in more than one split. Split
  was fixed in this config file "BEFORE any labeling/training," per the
  file's own header comment.

**Cross-split leakage check** (`scripts/build_dataset.py::dedup_cross_split`,
read directly): every method/class unit's AST-structural hash
(`struct_hash`) is checked against every OTHER split; any hash appearing
in more than one split is **dropped from all splits entirely** (not just
deduplicated to one copy) before any labeling happens. Actual counts
from the original full TRAIN+VAL+TEST build (`docs/dataset_report.md`):
**124 methods and 17 classes were found duplicated across splits and
removed from the dataset** — real, quantified evidence that near-identical
code did exist across repos (e.g. vendored utilities) and was actively
excluded rather than left in place. Within-split duplicates (boilerplate
repeated inside one repo) were also dropped: 1326 methods, 73 classes in
the original build; 990 methods, 34 classes in the more recent
TRAIN/VAL-only rebuild (`docs/dataset_report_train_val_rebuild.md`).

**Conclusion**: repository-level split is real and disjoint, and an
explicit, code-level, quantified anti-leakage mechanism exists and has
measurably fired (141 total cross-split collisions removed). No further
leakage vector was found in the files reviewed for this audit.

---

## 5. Unseen Repository Generalization

**Source**: `docs/phase8_unseen_repo_validation.md` (pip, pydantic, rich,
attrs — genuinely new organizations, not in train/val/test) and
`docs/explainability_report_candidate.md`/`docs/error_analysis_report_candidate.md`
(sqlalchemy/starlette/sphinx, which ARE in VAL — included here as the
in-distribution comparison point, not as unseen evidence).

Cross-checked against the project's own silver-label rule (not trusted
labels, independently re-derived on each unseen file) before judging
agreement:

| task | unseen-repo result | in-distribution (VAL) result |
|---|---|---|
| God Class | 3/4 clean agreements (pydantic.FieldInfo 0.999, rich.Progress 0.999, rich.Text 1.0); 1 case (pip.PipSession) previously rule-negative under the OLD formula, now rule-*positive* under the revised formula and model-predicted seed-dependent (0.231–0.994 across seeds, section 3) | VAL F1=0.796 |
| Long Method | strong agreement on clearly-long methods; boundary noise ±1 statement near the 15-statement cutoff in both directions, consistent with VAL's own near-boundary FN pattern (section 9) | VAL F1=0.806 |
| Feature Envy | 1/5 genuine cases caught on unseen terminal-rendering code (`rich_progress.py`/`rich_text.py`) | VAL F1=0.475 |

**Evidence of generalizing beyond training repos, not just memorizing
them**: yes, on 2 of 3 tasks (God Class, Long Method) the unseen-repo
behavior is qualitatively consistent with the VAL-measured behavior —
same failure shapes (boundary noise for Long Method; structural
size/state pattern recognition for God Class, further confirmed by the
seed-sweep majority agreeing PipSession-shaped classes are positive).
Feature Envy shows the weakest transfer (1/5), but the failure mode
identified (`docs/phase8_unseen_repo_validation.md`: "these specific
missed cases are all terminal-UI rendering widgets... a domain that
doesn't appear anywhere in the 9 training repos") is a **domain-coverage
gap**, not evidence of memorization — a memorizing model would fail
*more* uniformly across all unseen code, not selectively on one specific
code style while succeeding elsewhere. See section 6.

---

## 6. Feature Envy Investigation

Reported case: Long Method detected, God Class detected, Feature Envy
NOT detected on a known smelly example. Investigated generically per the
instruction NOT to fix this from the single example.

**A. Single-example miss?** No — this pattern reproduces systematically
across every existing evaluation surface checked: VAL error analysis
(`docs/error_analysis_report_candidate.md`, feature_envy recall=0.653,
so 90 real cases missed out of 259), the unseen-repo validation (4/5
missed), and the class-imbalance evidence below. Ruled out.

**B. Systematic Feature Envy weakness?** Yes, confirmed across every
metric surface: VAL F1=0.475, TEST F1=0.525 — the lowest of the three
tasks in every measurement this project has ever taken
(`docs/post_fix_retrain_comparison.md`, `docs/final_test_evaluation_report.md`,
`models/hybrid_baseline/history.json`'s own epoch-by-epoch numbers all
show feature_envy F1 well below long_method/god_class throughout
training).

**C. Data/label problem?** Partially, and already investigated in depth
this project cycle (`docs/dataprocessor_sample_diagnosis.md`): the
collaborator-chain double-counting bug (fixed earlier this session) was
a real label-computation defect that specifically suppressed Feature
Envy positives — already fixed, not a remaining issue. What remains,
re-confirmed here: **severe class imbalance**. `docs/dataset_report.md`:
train positives=163/8131 methods (2.0%), val=277/13905 (2.0%),
test=631/23613 (2.7%) — Feature Envy's positive rate is roughly 4-5x
rarer than Long Method's (~9-11%) and ~4x rarer than God Class's
(~7-8%, though God Class labels classes not methods, a different
denominator). **Not a volume problem specifically**: checked directly
against training tensors in `docs/phase8_unseen_repo_validation.md`:
"176 feature_envy positives, statement-count min=1/median=12/mean=14.6,
and 59/176 (34%) already have ≤8 statements" — short positive examples
are not rare in absolute terms, the *rate* relative to negatives is what's
extreme.

**D. Graph representation problem?** No new evidence found. The
collaborator-chain fix (this session, `ml/preprocessing/ast_parser.py`/
`metrics.py`) was exactly a graph/metric-computation fix and materially
helped (TEST feature_envy F1 0.412→0.525 after the two fixes this
project made). No further representation defect was found in this audit
pass.

**E. Threshold problem?** Partial evidence, worth naming precisely: TEST
ROC-AUC for feature_envy = **0.952**, PR-AUC = **0.546**
(`docs/final_test_evaluation_report.md`) — both substantially better
than the F1=0.525 number alone suggests. A high ROC-AUC with modest F1
at the fixed 0.5 threshold is the classic signature of "the model's
*ranking* of positive vs negative is reasonably good, but 0.5 is not
necessarily the F1-optimal operating point for a ~2-3% base-rate class."
This project has not investigated or tuned a task-specific decision
threshold (only 0.5 is used everywhere, per `run_eval` in
`scripts/train_hybrid_baseline.py` and the backend's `probs_for` in
`backend/app/inference.py` — both threshold at literal `>= 0.5`). This
is flagged as a genuinely untried lever, not evidence of a defect.

**F. Model-capacity problem?** No direct evidence either way — see
section 8; the model is small (270K trainable params) but Long Method
and God Class, sharing the same backbone capacity, perform substantially
better, so capacity alone does not explain Feature Envy's gap in
isolation. The task's own hyperparameter (`weight_fe=2.364`, roughly 2.4x
loss upweighting relative to `weight_lm=1.0`, from
`models/hybrid_class_pool_tuned_fixed_data/best_params.json`) shows this
was already identified and partially compensated for during
hyperparameter search — the remaining gap is what that compensation
didn't close.

**G. Repository-specific pattern learning?** Checked via
`docs/error_analysis_report_candidate.md`'s error-by-repo breakdown:
feature_envy errors are 248 sqlalchemy / 118 sphinx / 7 starlette,
against a VAL record-share of 65%/31%/4% respectively (computed this
audit from `models/hybrid_class_pool_tuned_fixed_data/errors_val.json`)
— **248/373 = 66.5%, closely tracking sqlalchemy's 65% share of VAL
records**, i.e. errors are proportional to repo size, not
disproportionately concentrated in one repo. No evidence of
repository-specific overfitting for this task specifically (contrast
with God Class, section 9, where errors ARE disproportionately
concentrated).

**Verdict for this section**: **B (systematic weakness) + C (data:
imbalance, not volume) + E (threshold, untried lever)** are the
supported conclusions, with **D and G ruled out** by direct evidence,
and **F insufficient/inconclusive**. This matches, rather than
contradicts, the project's own prior conclusion
(`docs/phase8_unseen_repo_validation.md`: "a genuine domain-coverage
gap, not a volume problem").

---

## 7. Dataset Quality

- **Class imbalance**: confirmed severe for Feature Envy (2.0-2.7%
  positive), moderate for God Class (7-8.2%), mildest for Long Method
  (~9-11%) — exact figures in section 6/`docs/dataset_report.md`.
- **Duplicate/near-duplicate examples**: actively detected and removed,
  not merely unexamined — see section 4 (141 cross-split, 1326+73
  within-split in the original build). This is evidence of *good*
  dataset hygiene, not a defect.
- **Same/similar code across splits**: same mechanism (section 4) — the
  fact that 124 methods and 17 classes WERE found duplicated across
  splits (and removed) is itself evidence that near-duplicate code does
  occur across these real-world repos (expected — common Python idioms,
  vendored code) and that the pipeline actively guards against it
  reaching the trained model as leaked signal.
- **Synthetic examples dominating a class**: no evidence found. All
  labels are derived from real, cloned open-source repositories
  (`scripts/build_dataset.py`, `data/raw/{split}/{repo}`) — no synthetic
  data generation step exists anywhere in the pipeline (verified: no
  `synthetic` or `generate` label-producing script found in `scripts/`
  aside from the AST-derived label rules themselves).
- **Repository-specific patterns**: see section 6/9 — checked explicitly
  for Feature Envy (proportional, not concentrated) and God Class (see
  section 9 — disproportionately concentrated in sqlalchemy).
- **Label inconsistencies**: the God Class formula itself was found and
  fixed this project cycle for exactly this kind of inconsistency (the
  old AND-only rule structurally could not label the DataProcessor/
  PipSession shape positive regardless of size —
  `docs/godclass_formula_revision.md` sections 1-3). This is now fixed,
  not an open issue. Labels are all silver (rule-derived), documented as
  such throughout (`ml/preprocessing/label_rules.py` module docstring:
  "These are SILVER labels: rule-based, not human-annotated") — this is
  an inherent, acknowledged limitation of the project's methodology, not
  a newly-found defect.
- **Suspiciously easy examples**: not directly measured in this audit
  (would require manual review of a sample); no evidence either way —
  **insufficient evidence**.
- **Feature Envy diversity**: see section 6 — the domain-coverage gap
  (no terminal/rendering-style code in the 9 training repos, all
  CLI/web-framework) is the most concrete, already-documented diversity
  finding.

---

## 8. Model Capacity

**Architecture** (`ml/models/gat_baseline.py`, read directly): 2-layer
`HeteroConv` stack of `GATConv` per edge relation (14 relation types,
`EDGE_TYPES`), `hidden_dim=64`, `heads=2` (per
`models/hybrid_class_pool_tuned_fixed_data/best_params.json`), dropout
0.174, plus a `class_method_pool` mean-pooling mechanism specifically
for the god_class head (concatenating class's own embedding with a
mean-pool of its methods' embeddings, doubling that head's input width
to 128). Node features: structural metrics (3-5 dims depending on node
type) concatenated with a frozen, non-trainable 768-dim CodeBERT
embedding per class/method/function node.

**Parameter count** (computed this audit by instantiating the actual
saved config): **269,315 trainable parameters** in the `HeteroGAT` model
itself. The CodeBERT encoder (`microsoft/codebert-base`, ~125M
parameters) is loaded but entirely frozen (`.eval()`, no gradient
updates) — it contributes fixed embeddings, not trainable capacity.

**Dataset size for comparison**: TRAIN split has 8479 methods and 1436
classes across 547 files (`docs/dataset_report_train_val_rebuild.md`) —
before graph-batching multiplies effective gradient-update exposure
across ~150 epochs (with early stopping typically halting around
epoch 40-82 per section 2/3).

**Assessment**: 269K trainable parameters against a labeled-node count
in the thousands (with the bulk of representational capacity actually
coming from the frozen, pre-trained CodeBERT embeddings rather than from
this project's own trainable weights) is a modest, not excessively large,
model relative to the dataset. This is consistent with — not merely
asserted alongside — the LOW overfitting verdict in section 1: a
269K-parameter classifier head sitting on top of frozen embeddings has
comparatively little room to memorize idiosyncratic training examples
compared to, say, fine-tuning all of CodeBERT's 125M parameters end-to-end
(which this project deliberately does not do — CodeBERT is frozen
throughout, `scripts/augment_graphs_with_codebert.py` and
`backend/app/inference.py` both load it with `.eval()` and no optimizer
reference). No evidence the model is too small either — Long Method and
God Class both reach F1 > 0.74 on TEST with this capacity, so it is not
obviously underfitting the tasks it performs well on.

---

## 9. Error Analysis

**Source**: `docs/error_analysis_report_candidate.md` (candidate
checkpoint, VAL split, the only error-analysis artifact that exists for
the actual deployed model — see `docs/godclass_formula_revision.md`
section 12 for why the original Phase 11 report targeted a different,
outdated architecture and is not used here).

| task | precision | recall | F1 | FP | FN |
|---|---|---|---|---|---|
| long_method | 0.727 | 0.904 | 0.806 | 408 | 116 |
| feature_envy | 0.374 | 0.653 | 0.475 | 283 | 90 |
| god_class | 0.776 | 0.817 | 0.796 | 49 | 38 |

**Hardest task (by F1)**: Feature Envy (0.475), by a wide margin — see
section 6.
**Easiest task (by F1)**: Long Method (0.806), and its false negatives
are heavily boundary-clustered (102/116 within 5 statements of the
threshold vs only 2 comfortably over) — the model is rarely *confidently*
wrong on Long Method, it is uncertain near the boundary, which is
expected/benign.

**God Class false positives — a real, precisely-quantified finding**:
"Of 49 false positives: 0 satisfy both rule branches, 0 the size branch
only, 0 the field branch only, **49 satisfy NEITHER branch**" — every
single god_class VAL false positive is a case where the model predicts
positive despite failing BOTH the size rule and the revised field rule
entirely. This is the same CodeBERT-semantic-override pattern already
named in `docs/godclass_formula_revision.md` section 12 (the
`AbstractTextClause` example, 8 methods/215 loc/0 fields, predicted
0.999) and in `docs/phase4_6_external_manual_eval.md`
(`Shell.parseCmd`) and `docs/dataprocessor_sample_diagnosis.md`
(`format_phone`) — a recurring, now well-documented architecture-level
limitation (the model's semantic embedding signal can override structural
signal for a minority of examples) rather than a new discovery.

**Feature Envy false positives**: 151/283 (53%) have
`dominant_external_count <= self_access_count` — flagged despite
directly failing the label rule's own direction, a genuine
signal-learning gap on more than half of this task's false positives,
distinct from the God Class pattern (which is 100% neither-branch, an
even more extreme version of the same phenomenon).

**Repository concentration** (computed this audit from
`errors_val.json`'s raw per-repo record counts: sqlalchemy 20247/31154
=65.0%, sphinx 9766/31154=31.3%, starlette 1141/31154=3.7%):

| task | sqlalchemy error share | vs. sqlalchemy's 65.0% record share |
|---|---|---|
| long_method | 313/524 = 59.7% | slightly *under*-represented |
| feature_envy | 248/373 = 66.5% | proportional |
| god_class | 75/87 = **86.2%** | **disproportionately over-represented** |

God Class errors are meaningfully concentrated in sqlalchemy beyond
what its VAL record share alone would predict — consistent with the
neither-branch-FP finding above (sqlalchemy's ORM/SQL-expression classes,
e.g. `AbstractTextClause`, `OrderingList`, `Constraint`, `TableClause`,
apparently trigger the semantic-override pattern more than the other two
VAL repos' code). This is a real, worth-tracking pattern, though the
audit does not have enough information to say whether it's a property of
SQL/ORM-style code specifically or a coincidence of sqlalchemy's classes
happening to sit near the rule's boundaries more often (many of the
listed FPs have loc in the 100-220 range, near but under the 192 floor
combined with moderate method counts — plausibly boundary-adjacent, not
purely a semantic-override story; both explanations are visible in the
same table and this audit cannot fully separate them from the data
available).

---

## 10. Overfitting/Underfitting Assessment

Restated with the full evidence trail from sections 2, 3, 8:

- **Overfitting: LOW.** No train-vs-val gap can be directly measured
  (section 2), but every available proxy points away from overfitting:
  (a) TEST performance is *not* uniformly lower than VAL — Feature Envy
  is actually higher on TEST (0.525 vs VAL 0.475); (b) the model is
  small (270K trainable params) relative to the dataset and sits on
  frozen, non-fine-tuned embeddings (section 8); (c) early stopping on
  held-out VAL is the actual training-halting mechanism used, with
  observed best-epochs well short of the 150-epoch budget (section 2);
  (d) the historical `hybrid_baseline` proxy curve, the only artifact
  showing a real train-loss-vs-val-F1 trend in this repo, shows a
  plateau, not a collapse, and the deployed candidate's architecture is
  specifically defended against that plateau-then-drift pattern by early
  stopping.
- **Underfitting: LOW-MEDIUM.** LOW for Long Method and God Class (TEST
  F1 0.740/0.767, ROC-AUC 0.98/0.98). Feature Envy sits at TEST F1=0.525
  with ROC-AUC=0.952 — the high ranking quality alongside modest F1@0.5
  (section 6E) argues against pure underfitting (a genuinely underfit
  model would show poor ranking too, not just a suboptimal threshold) —
  classified LOW-MEDIUM rather than MEDIUM-HIGH for this reason,
  though this audit could not fully separate "class imbalance depresses
  F1 at a fixed threshold" from "the model has not learned the task as
  well as the other two" — both are plausible and partially supported.

---

## 11. Risks

1. **No train-set metric ever recorded for the deployed model family**
   (section 2) — future retrains should log a train-set eval pass (or
   at minimum train-set loss trend) so a real train/val gap can be
   measured directly instead of relying on an unrelated architecture's
   historical proxy.
2. **Feature Envy's decision threshold has never been tuned away from
   0.5** (section 6E) — a real, low-risk, non-retraining lever that has
   not been tried.
3. **God Class errors concentrate in sqlalchemy specifically** (section
   9) — worth watching if this project ever expands VAL/TRAIN repo
   coverage; not yet distinguishable between a semantic-override story
   and a boundary-density story with current evidence.
4. **Only 4 seeds sampled for stability estimates** (section 3) — the
   0.047 god_class spread and the seed-43 selection are the best
   available evidence, not a large-sample guarantee.
5. **All labels are silver/rule-derived, never human-annotated at scale**
   (acknowledged project-wide, `ml/preprocessing/label_rules.py`
   docstring) — this bounds how confidently "ground truth" claims can be
   made anywhere in this audit; every F1/precision/recall number in this
   document is measured against silver labels, not human judgment.

---

## 12. Recommended Next Step

**DO NOT RETRAIN — current evidence does not justify it.**

No systematic overfitting, underfitting (beyond the already-understood
Feature Envy class-imbalance story), or leakage was found. The single
misclassified example that prompted this audit is not, on the evidence
gathered here, representative of a new defect — Feature Envy's weakness
is real, already documented, already partially addressed (the seed-43
selection specifically improved it, TEST F1 0.412→0.525), and its
remaining gap is best explained by severe class imbalance (2-2.7%
positive rate) plus an untried decision-threshold lever, not by a
data, graph-representation, or capacity defect this audit could find.

If a future change is pursued, the evidence in this audit points at two
**non-retraining** experiments as the safest, most evidence-targeted
next steps, in order:

1. **Tune Feature Envy's decision threshold on VAL** (not TEST) using
   the existing precision/recall curve implied by ROC-AUC=0.952,
   PR-AUC=0.546 (section 6E) — this requires no retraining, no dataset
   change, and directly targets the one concrete, quantified,
   untried lever this audit identified.
2. Only if (1) doesn't close the gap sufficiently: instrument
   `scripts/tune_hybrid_optuna.py::train_one` to also log train-set
   metrics per epoch (a small, additive change, not a retrain of the
   current candidate) before deciding whether any future retrain is
   warranted — this directly resolves the section 2 "insufficient
   evidence" gap for good, for whatever the next architecture/data
   change ends up being.

Neither of these is proposed as an action taken by this audit — both are
flagged as the evidence-justified next steps, for explicit user
direction, consistent with this audit's scope (read-only, no
modifications).
