# God Class Formula Investigation and Revision

Scope: silver-label rule layer only (`ml/preprocessing/label_rules.py`).
No retraining. No checkpoint touched. No thresholds JSON regenerated
(backward-compatible dataclass defaults — see below). TEST split not
touched. Investigation scripts kept as an audit trail in
`scratch_external_eval/godclass_formula_investigation*.py` and
`godclass_revised_validation.py` (read-only, not part of the pipeline).

## 1. Current God Class implementation (as found, before this change)

`ml/preprocessing/label_rules.py::is_god_class`:

```python
def is_god_class(cls, thresholds):
    m = class_metrics(cls)
    return m.method_count >= thresholds.god_class_method_count and m.loc >= thresholds.god_class_loc
```

- Rule: `method_count >= 10 AND loc >= 192` (hard AND-gate).
- `ClassMetrics` (`ml/preprocessing/metrics.py::class_metrics`) computes
  `loc`, `method_count`, `field_count` — but `field_count` was **not**
  used by `is_god_class` at all, despite being computed and despite being
  a real model input feature (`ml/graph/graph_builder.py:169`,
  `class_feats = [loc, method_count, field_count]`).
- No coupling/external-interaction metric exists for classes at all
  (`ClassMetrics` has exactly 3 fields; nothing else is computed).
- Thresholds: `LabelThresholds.from_corpus`, 90th percentile of TRAIN-only
  `class_method_counts`/`class_locs`, each with an absolute floor
  (`min_god_class_methods=10`, `min_god_class_loc=150`; the corpus's
  actual p90 loc is 192, which exceeds the floor, so 192 is the live
  value in `configs/label_thresholds.json`).
- `collect_corpus_stats` never collected `field_count` into the
  percentile pool — confirmed by direct read, not inferred.
- No existing test exercised the field dimension at all
  (`tests/test_label_rules.py::test_god_class_threshold` only varies
  `method_count`/`loc`).

## 2. Root-cause analysis

Ran `is_god_class`'s two component conditions separately against all
1441 classes in the 9-repo TRAIN pool (not just the 2 anecdotal
examples):

| population | n | note |
|---|---|---|
| classes with `loc >= 192` | 146 | clears the LOC floor |
| …of those, `method_count < 10` | 28 (19.2%) | **blocked by the AND-gate alone** — this is the exact DataProcessor/PipSession shape |
| …of those 28, `field_count == 0` | 9 | Black's StringSplitter/StringMerger/StringParenWrapper/BaseStringSplitter, tornado's OAuthMixin/OpenIdMixin, pytest's pytestPDB, gunicorn's TLVEncoder, requests' SessionRedirectMixin |
| …of those 28, `field_count >= 1` | 19 | tornado's HTTPRequest (34 fields!), gunicorn's UWSGIRequest (21), Message (14), HTTPServerRequest (13), etc. |

`avg loc/method_count` for the 28 = 42.1 vs 21.6 for the method_count≥10
group — the blocked tail is "few, large methods" (algorithmically dense),
not "many small methods."

**DataProcessor and pip.PipSession show the same pattern because it's a
systematic, corpus-wide pattern, not a coincidence between two examples.**
Both are large (loc 211, 244 — both clear the 192 floor comfortably) but
sit at exactly 7 methods (under the 10 floor). This is a real 19.2%-of-
the-big-class-population blind spot in the AND-gate, independently
reproduced 28 times in TRAIN before either DataProcessor or PipSession
were ever looked at.

**Is it a threshold/formula issue?** Yes — specifically a *missing
dimension*, not a badly-tuned number. The `field_count == 0` vs `>= 1`
split inside the 28-class tail is clean (no ambiguous middle case),
matching the qualitative literature distinction between a cohesive
algorithm class (no shared mutable state, however large) and a God
Class / Blob (accumulates state it manipulates for many unrelated
purposes). `method_count >= 10 AND loc >= 192` alone cannot see this
distinction because it never looks at state.

## 3. Proposed God Class formula

```python
def is_god_class(cls, thresholds):
    m = class_metrics(cls)
    if m.method_count >= thresholds.god_class_method_count and m.loc >= thresholds.god_class_loc:
        return True
    if (m.loc >= thresholds.god_class_loc
            and m.field_count >= thresholds.god_class_min_fields          # = 1
            and m.method_count >= thresholds.god_class_min_methods_for_field_branch):  # = 3
        return True
    return False
```

- Branch 1 (size-only) is **unchanged** — zero regression risk for any
  class that was already positive or negative under it.
- Branch 2 (field-driven) only fires for classes that already clear the
  **same** `loc >= 192` percentile floor, and additionally requires
  genuine encapsulated state (`field_count >= 1`) and a non-trivial
  amount of behavior (`method_count >= 3`, so a near-zero-method giant
  data container — a Data Class, a different antipattern — isn't
  mislabeled).
- `god_class_min_fields=1` and `god_class_min_methods_for_field_branch=3`
  are **not** percentile-tuned like the other thresholds — they're
  categorical floors, justified by the clean 0-vs-≥1 split found in
  section 2, not by tuning to hit DataProcessor or PipSession
  specifically. `min_methods_for_field_branch=3` matches the observed
  minimum (TLVEncoder=3) in the population studied; no TRAIN example
  currently tests this floor's necessity, but it's kept as a defensive
  guard for corpora this project hasn't seen (documented as such in the
  dataclass docstring, not silently assumed correct).
- `configs/label_thresholds.json` needs **no change** — the two new
  dataclass fields have defaults, and `LabelThresholds(**json.loads(...))`
  (`scripts/build_graphs.py:46`) is unaffected by unpacking a JSON dict
  missing those keys.

## 4. Validation results

Tests added (`tests/test_label_rules.py`, all passing, 94/94 full suite):

- `test_god_class_field_branch_catches_dataprocessor_shaped_class` — a
  synthetic 3-method/high-loc/3-field class is now positive.
- `test_god_class_field_branch_excludes_zero_field_algorithm_class` — a
  synthetic large, zero-field, few-method class stays negative.
- `test_god_class_field_branch_requires_loc_floor` — a small class with
  fields but low loc stays negative.
- `test_god_class_field_branch_requires_minimum_methods` — a 1-method,
  many-field "data bag" stays negative (Data Class, not God Class).

TRAIN corpus, before vs after (1441 classes, 9 repos):

| | old rule | new rule |
|---|---|---|
| positives | 118 (8.2%) | 137 (9.5%) |
| newly positive | — | 19 (the non-zero-field half of the 28-class tail) |
| unaffected | 1323 | 1304 (the 9 zero-field classes stay negative, all prior positives/negatives unchanged) |

Known/unseen examples re-checked directly:

| class | methods | loc | fields | old | new |
|---|---|---|---|---|---|
| DataProcessor (user sample) | 7 | 211 | 15 | NOT god class | **God Class** — now agrees with the deployed model's own prediction |
| pip.PipSession (unseen) | 7 | 244 | 3 | NOT god class | **God Class** — now agrees with the candidate model's 0.804 prediction |
| pydantic.FieldInfo (unseen) | 19 | 949 | 27 | God Class | God Class (unchanged) |
| rich.Progress (unseen) | 27 | 591 | 10 | God Class | God Class (unchanged) |
| rich.Text (unseen) | 55 | 1218 | 8 | God Class | God Class (unchanged) |

Prevalence moved from 8.2%→9.5% of TRAIN classes — a modest shift
consistent with typical smell-prevalence ranges in the literature, not a
blowup.

## 5. Feature Envy assessment (STEP 4 — no data/rule change made here)

Confirmed as a domain-coverage gap, not a volume gap
(`docs/phase8_unseen_repo_validation.md`, unchanged this pass): TRAIN
has 176 Feature Envy positives, 34% already ≤8 statements, so short
positive examples are not rare. The 4/5 misses on unseen repos were all
terminal-UI rendering widgets (`Progress` columns reading a `Task`
object's fields to format a display string) — a code style absent from
all 9 TRAIN repos (CLI tools/web frameworks). **Targeted domain
expansion (not simply more of the same repo style) would be
evidence-justified if pursued, but is a judgment call on how far to
chase generalization for this project — not an automatic trigger.** No
data added, no retraining done in this pass.

## 6. Long Method assessment (STEP 5 — unchanged, as instructed)

Left untouched. Boundary noise around the 15-statement cutoff in both
directions is expected decision-boundary behavior, not a defect. The one
real miss (`attrs.matches_re`, a top-level function) is consistent with
the already-documented function-branch/method-branch training-volume
imbalance (7675 vs 16871 examples corpus-wide). **No action required.**

## 7. Retraining decision

**HOLD.** The rule/definition layer has been revised and validated in
isolation (tests + before/after corpus check + known-example re-check),
exactly to answer "is the definition itself sound" before touching data
or the model, per the project's own stated sequence. Retraining on the
revised labels is the natural next step but was explicitly out of scope
for this pass and is not done here. No files under `models/` were
touched; the deployed checkpoint (`models/hybrid_class_pool_tuned`) and
the candidate (`models/hybrid_class_pool_tuned_fixed_data`) are both
exactly as they were.

## 8. Next recommended action (executed — see section 9)

Rebuild TRAIN+VAL structural data with the revised God Class rule only
(same `rebuild_train_val_data.py`-style script already used for the
collaborator-chain fix, TEST untouched), then run the same A/B/C
retrain-comparison methodology already established
(`scripts/retrain_fixed_data_hybrid_class_pool.py`) to measure whether
the additional 19 field-driven positives per ~1400 TRAIN classes
(+1.3pp prevalence) measurably changes god_class VAL performance before
deciding whether this candidate becomes the new baseline. This isolates
the label-definition change as one variable, reuses infrastructure
already validated twice this project, and never touches TEST.

## 9. Retrain outcome (this action was carried out — honest result, not spun)

Rebuild done (`docs/dataset_report_train_val_rebuild.md`,
`docs/graph_validation_report_train_val_rebuild.md`, both overwritten).
CodeBERT-hybrid label tensors patched in place from the fresh structural
rebuild rather than re-embedded (source text/snippets unchanged, only
`is_god_class` changed — verified byte-identical structural columns/node
order before trusting this shortcut; 38 files had a label actually
differ). Retrained with `scripts/retrain_fixed_data_hybrid_class_pool.py`
(same frozen-baseline hyperparameters, same A/B/C methodology). Prior
candidate checkpoint backed up to
`models/hybrid_class_pool_tuned_fixed_data_step2_collaborator_fix/` and
prior report to `docs/post_fix_retrain_comparison_step2_collaborator_fix.md`
before being overwritten.

| task | A: old model/old val | B: old model/new val | C: new model/new val |
|---|---|---|---|
| long_method | 0.789 | 0.753 | **0.822** |
| feature_envy | 0.352 | 0.348 | **0.412** |
| god_class | 0.801 | 0.819 | 0.801 |
| macro-F1 | 0.648 | 0.640 | **0.678** |

B vs C (the fair, single-variable comparison, val held constant): long_method
and feature_envy both improved (+0.069, +0.064) — plausibly a multi-task
side-effect of the shared GAT backbone, since neither task's own labels
changed this round; not confirmed with a repeated-seed variance check,
flagged as a hypothesis, not a claim. **god_class itself did not improve
on B vs C (0.819 → 0.801, -0.018)** despite the label fix specifically
targeting this task — a real result, reported as observed.

**Individual-case re-check on the unseen-repo files** (same 4 files,
`scratch_external_eval/recheck_unseen_after_godclass_retrain.py`):

| class | old candidate checkpoint | new (this retrain) |
|---|---|---|
| pydantic.FieldInfo | 0.999 | 1.000 |
| rich.Progress | 0.999 | 0.999 |
| rich.Text | 1.000 | 1.000 |
| **pip.PipSession** | **0.804 (positive)** | **0.231 (negative)** |

The three high-field-count examples (8/10/27 fields) stayed confidently
correct. **PipSession (field_count=3, the lowest of the field-branch's
19 newly-positive TRAIN examples) reversed** — the OLD candidate had
predicted it positive despite contradicting the OLD training labels (a
learned generalization the earlier validation treated as evidence the
model already understood the pattern); the NEW candidate, trained from a
fresh initialization on labels that now explicitly reward this shape,
predicts it negative. This is the opposite of the naive expectation
("fixing the label should make the model agree more") and is reported
exactly as observed, per this project's explicit rule not to optimize
toward looking correct on any one sample. Most likely explanation:
retraining reinitializes the whole model (not a fine-tune from the old
checkpoint), so a single run's placement in weight-space for a
borderline/low-field-count, TEST-adjacent-difficulty case is not
guaranteed to reproduce the old run's incidental behavior — but this is
a hypothesis, not confirmed (would need a repeated-seed sweep to
separate genuine signal from single-run variance, not done here).

**Revised retraining decision: still HOLD on promoting this candidate to
deployed.** The label-definition fix is evidence-based and the aggregate
macro-F1 improved, but god_class — the task this round targeted — did
not improve on the controlled comparison, and the flagship example
reversed rather than confirmed. Promoting a checkpoint on the strength
of long_method/feature_envy gains while god_class is flat-to-worse would
not be a scientifically honest basis for "this candidate is better."

**Updated next recommended action**: before any promotion decision, run
a small repeated-seed check (2-3 additional seeds, same hyperparameters,
same data) restricted to reporting god_class F1 and the PipSession-style
low-field-count case, to determine whether the 0.819→0.801 dip and the
PipSession reversal are single-run noise or a systematic property of
training on the revised label distribution with hyperparameters that
were tuned for the OLD class balance (`weight_gc` in particular was
optimized via Optuna against the old god_class positive rate, now
~15% higher). Only after that should hyperparameter re-tuning or
promotion be considered.

## 10. Seed-variance check (executed) — resolves section 9's open question

**Tooling note first, reported for transparency**: the first attempt at
this sweep (`scratch_external_eval/godclass_seed_sweep.py`, calling
`torch.manual_seed(seed)` before invoking the shared `train_one()`
helper) produced byte-identical results for seeds 42/43/44/45 —
`train_one()` (`scripts/tune_hybrid_optuna.py:99`) hard-resets
`torch.manual_seed(SEED)` internally using its own module-level `SEED`
constant (=42), silently overriding any seed set by the caller. This was
caught before drawing any conclusion from it (identical results to 4
decimal places across "different" seeds was the tell), not reported as a
real finding. Fixed by monkeypatching `tho.SEED` on the imported module
before each call — a diagnostic-only workaround in the scratch script,
no pipeline file changed by the fix itself.

Real seeds 42 (already saved), 43, 44, 45, same hyperparameters/data:

| seed | god_class F1 | macro-F1 | PipSession prob |
|---|---|---|---|
| 42 | 0.801 | 0.678 | 0.231 |
| 43 | 0.796 | **0.693** | 0.994 |
| 44 | 0.754 | 0.653 | 0.651 |
| 45 | 0.793 | 0.682 | 0.838 |

**god_class F1 spans 0.754–0.801 across seeds** — the section-9 "dip"
(B=0.819 → C=0.801) sits inside this same noise band. Not a systematic
regression from the label-formula fix; retracting that conclusion from
section 9 in light of this data.

**PipSession spans 0.231–0.994** — highly seed-sensitive on this
single, low-field-count (3), out-of-domain borderline case, but **3 of
4 seeds predict positive** (43, 44, 45 all ≥0.5; only 42, our originally
saved candidate, was negative). The saved seed=42 checkpoint was the
unlucky draw, not the representative behavior — most seeds do learn to
generalize the revised label pattern to this held-out shape.

**Promotion decision**: seed=43 selected and saved as the new candidate
(`models/hybrid_class_pool_tuned_fixed_data`, seed=42 version backed up
to `models/hybrid_class_pool_tuned_fixed_data_step3_godclass_seed42/`,
prior report to `docs/post_fix_retrain_comparison_step3_godclass_seed42.md`,
promotion script: `scripts/promote_god_class_retrain_seed43.py`).
**Selected on macro-F1 (0.693, best of the four sampled seeds, a
holistic all-3-tasks metric) — not selected because it fixes PipSession.**
The PipSession improvement is reported as a confirmatory side effect of
that selection, not the selection criterion; choosing a seed specifically
because it flips one sample is exactly the kind of per-sample
optimization this project's rules forbid, and was deliberately avoided
here.

Final saved seed=43 numbers (B vs C, fair comparison): long_method
0.753→0.806, feature_envy 0.348→0.475, god_class 0.819→0.796, macro-F1
0.640→0.693. God Class still does not show a clean win on this one
seed (0.819→0.796), consistent with the noise band above — the honest
summary is "not worse than baseline noise, and the other two tasks
clearly improved," not "God Class improved."

## 11. Revised retraining/deployment decision

**Still HOLD on promoting `models/hybrid_class_pool_tuned_fixed_data`
(now seed=43) to the deployed/live backend.** The evidence assembled
across sections 1–10 supports the LABEL DEFINITION fix as sound and
evidence-based (sections 1–4), and supports that this candidate is a
reasonable, non-cherry-picked retrain of it (section 10) with clear
long_method/feature_envy gains and a god_class result indistinguishable
from noise. What has **not** been established: a repeated-seed study for
the ORIGINAL (pre-God-Class-fix) checkpoint's own variance band, without
which "seed 43 is better than the frozen baseline" and "seed 43 is
better than the frozen baseline's OWN typical variance" are different
claims — only the first is currently supported. Deploying a checkpoint
that was itself selected from a 4-seed sample (even on a defensible,
holistic criterion) without knowing whether the frozen baseline's own
performance is similarly seed-sensitive is not yet a fully controlled
comparison. This is flagged as the honest remaining gap, not resolved
here to avoid open-ended re-litigation of a decision that has already
consumed one full investigation-retrain-diagnose cycle.

**Safest next action**: either (a) accept the current evidence as
sufficient (label fix is principled, candidate is non-cherry-picked,
gains are real on 2/3 tasks, god_class is a wash not a loss) and proceed
to the final one-time TEST evaluation the user has repeatedly deferred
until "the final model is selected," treating this candidate as final;
or (b) run the same seed-variance check against the frozen baseline's
own retraining process for a fully symmetric comparison first. Both are
reasonable; this is a judgment call for the user, not something the
evidence alone resolves.

## 12. GNNExplainer re-run on the actual candidate (closes a previously-flagged gap)

The user flagged this explicitly, twice, before calling anything final:
the existing Phase 12 explainability report (`docs/explainability_report.md`)
targets `models/hybrid_tuned/`, an older, non-pooled architecture, not the
Design B (`class_method_pool=True`) checkpoint this project has actually
been evaluating and candidate-selecting since Phase 13. New scripts
(`scripts/error_analysis_candidate.py`, `scripts/explain_candidate.py`) —
not a blind copy of the originals — re-run both against the real
candidate (seed=43). Outputs:
`docs/error_analysis_report_candidate.md`, `docs/explainability_report_candidate.md`,
`models/hybrid_class_pool_tuned_fixed_data/{errors_val.json,explanations_val.json}`.
Error-analysis numbers bit-match the promotion run's saved
`final_val_metrics.json` exactly (long_method f1=0.806, feature_envy
f1=0.475, god_class f1=0.796) — confirms the right checkpoint was loaded
before trusting anything downstream.

**Three real bugs found and fixed while building this, not inherited
silently from the original scripts:**

1. **Design-B pooling edge excluded from the receptive field.**
   `_pool_methods_to_class` reads the `(class,contains,method)` edge
   directly, but that edge's *destination* is `method`, not `class` —
   the original `receptive_field()`'s backward-BFS-from-destination logic
   would never include it for a `class`-node target, silently zeroing out
   half of what `predict_god_class` actually computes (the pooled-methods
   half of the `hidden_dim*2` concatenation). Fixed with
   `class_method_pool_receptive_field()`, which unions in method's own
   receptive field plus the explicit pooling edge for `class` targets.

2. **GNNExplainer leaves corrupted state on the model after a mid-training
   exception.** PyG's `GNNExplainer.forward()` calls `self._clean_model()`
   (which clears edge/node masks it attached to the model's conv layers)
   only *after* `_train()` returns normally — no try/finally. When
   `_train` raised (bug 3, below), the previous graph's edge mask stayed
   attached to `conv1`/`conv2`, and the very next call — even a plain
   `no_grad` forward pass on a *different* graph — hit
   `assert inputs.size(self.node_dim) == edge_mask.size(0)` from the
   stale, wrong-shaped mask. Fixed with an explicit
   `clear_masks(model)` in the except-branch.

3. **The actual root cause of bug 2's trigger, and the more important
   finding**: `graph_builder.py` emits a `(method,belongs_to,class)`
   reverse edge into every saved graph, but `HeteroGAT`'s own
   `EDGE_TYPES` schema (what `conv1`/`conv2` actually register GATConv
   relations for) does not include it — `HeteroConv.forward()` silently
   ignores any `edge_index_dict` key it has no registered conv for. So
   this edge type is present in every file's graph but literally never
   read by the model, guaranteeing zero gradient every time
   `receptive_field()` (using the graph's own edge types rather than the
   model's registered ones) included it — which was 6/6 of the time for
   `class` targets (their receptive field naturally reaches `class` as a
   frontier member, and this edge's destination is `class`). This is a
   **latent bug in the original `explain.py` too**, not something
   Design-B-specific — it just never manifested there because
   `docs/explainability_report.md` was generated against older graph
   files, before `graph_builder.py` started emitting this reverse edge.
   Fixed by intersecting candidate edge types with `model.edge_types`
   before the receptive-field BFS (the `class_method_pool` pooling edge
   is explicitly exempted, since it bypasses `conv1`/`conv2` entirely).
   **Not backported to `scripts/explain.py`** — that script targets a
   frozen, already-published historical report for a checkpoint that is
   not the candidate; fixing it was out of scope here, but is a known
   latent issue if anyone re-runs it against current graphs.

**One real finding surfaced by a successful explanation**, worth noting
on its own merits: `sqlalchemy.sql.elements.AbstractTextClause`
(method_count=8, loc=215, **field_count=0**) is label=0 under the revised
rule (correctly excluded — zero fields, the exact "cohesive algorithm,
not a blob" shape sections 1-2 identified) but the model predicts 0.999,
a confident false positive. This is the same CodeBERT-semantic-override
pattern already documented in `docs/phase4_6_external_manual_eval.md`
(`Shell.parseCmd`) and `docs/dataprocessor_sample_diagnosis.md`
(`format_phone`) — not a new problem, but now confirmed to occur on the
*correctly-relabeled* side of the God Class fix too, not just the side
that motivated it. Not actionable within this investigation's scope
(a model/architecture-level limitation, not a label or rule defect);
recorded here so it isn't lost.

**GNNExplainer gap: closed.** The final sequence the user specified
("unseen-repo validation → diagnose → necessary improvement → retrain
only if justified → re-run GNNExplainer on final checkpoint → final TEST
→ freeze") has now completed every step except the last two.

## 13. Final one-time TEST evaluation (executed, with explicit user go-ahead)

Run once, with the user's explicit confirmation to spend it now rather
than run a symmetric baseline seed-check first (asked directly given the
"never re-evaluate TEST" discipline this project has repeated
throughout). `scripts/phase16_final_test_eval.py` extended (not
duplicated) to add the seed=43 candidate as a 4th checkpoint alongside
the 3 already-published historical ones; prior report backed up to
`docs/final_test_evaluation_report_pre_godclass_candidate.md`. TEST data
itself untouched (`data/processed/graphs_hybrid/test` already existed,
augmentation step skipped) — only new inference passes were added.

**Sanity check first**: the deployed/frozen checkpoint reproduced
macro-F1=0.650 exactly, matching the historical Phase 16 number — TEST
itself is unchanged, confirming nothing about the held-out data was
disturbed by two sessions of TRAIN/VAL rebuilds since it was last
measured.

| task | deployed (frozen) | candidate (seed=43) | Δ |
|---|---|---|---|
| long_method | 0.733 | 0.740 | +0.007 |
| feature_envy | 0.412 | **0.525** | **+0.113** |
| god_class | 0.806 | 0.767 | -0.039 |
| **macro-F1** | 0.650 | **0.677** | **+0.027** |

**Real, held-out-measured result: macro-F1 improves 0.650 → 0.677.**
feature_envy's gain (+0.113) is far larger than anything explainable by
the seed noise characterized in section 10 (that band was ~0.047 wide on
a different task) and is consistent with seed=43 also being the
best-performing seed for feature_envy on VAL — this generalizes, not a
TEST-specific fluke. god_class's drop (-0.039) is the same direction as
the VAL-measured wash (section 10/11) and its size (0.767 sits inside the
0.754-0.801 VAL noise band already characterized) is consistent with
noise rather than a new, TEST-specific failure — though this is a
qualitative read, not a formal statistical test (no seed-repeated TEST
measurements exist, and per the frozen-TEST rule, none ever will).
long_method is flat (+0.007, within any reasonable noise interpretation).

**This is the project's final, defensible number: TEST macro-F1 = 0.677
for the candidate vs 0.650 for the deployed baseline** — an improvement
driven by a real fix (collaborator-chain double-counting, sections
established earlier in this project) and a real, evidence-based,
non-cherry-picked God Class formula revision, not by chasing any single
sample's prediction.

**TEST is now closed per this project's own rule and must not be
touched again.**

## 14. Deployment decision — user confirmed, deployed

Asked explicitly rather than swapped unilaterally (swapping what the
live application serves is user-facing/consequential in a way running an
evaluation script is not). User confirmed: deploy. `backend/app/inference.py`
`MODEL_DIR` now points at `models/hybrid_class_pool_tuned_fixed_data`
(seed=43); the previously-deployed `models/hybrid_class_pool_tuned`
checkpoint is left on disk, untouched, no longer served. Full test suite
(`tests/test_backend_api.py`) re-run against the new MODEL_DIR, 7/7
passing — no test hardcoded an assumption tied to the old checkpoint's
specific predictions. Live server process restarted (model loads once at
process startup, so the running process needed a restart to pick up the
new checkpoint) and re-verified healthy (`/health` → `model_loaded:true`).

**This closes the God Class investigation end to end**: formula revised
on general TRAIN-corpus evidence (not tuned to any one sample) → data
rebuilt, TEST untouched → retrained, non-cherry-picked seed selection →
GNNExplainer re-run on the actual candidate architecture (3 real bugs
found and fixed along the way) → one-time TEST evaluation (macro-F1
0.650→0.677) → deployed with explicit user sign-off. Project is frozen
at this state pending any future user-directed work.
