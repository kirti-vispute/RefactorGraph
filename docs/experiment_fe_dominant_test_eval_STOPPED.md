# FINAL TEST EVALUATION — STOPPED BEFORE INFERENCE

Per explicit instruction: *"If any unexpected modification is found,
STOP before evaluating TEST and report the issue."* An unexpected
integrity issue was found during the pre-evaluation verification step.
**No inference was run. No TEST metric was computed. The one-time TEST
evaluation has not been consumed.**

## 1. Integrity Check

**FAILED — one check, described below. All others passed.**

| check | result |
|---|---|
| Candidate checkpoint is exactly `models/experiment_fe_dominant/seed_43/` | PASS |
| No candidate retraining since selection | PASS (file timestamps unchanged since the candidate review) |
| Hyperparameters match the selected experiment | PASS (`best_params.json` re-read, identical) |
| Architecture unchanged | PASS |
| Production `graph_builder.py` still uses `external_access_count` | PASS (re-read directly) |
| Live thresholds: Feature Envy 0.65, Long Method/God Class 0.50 | PASS (re-read `backend/app/inference.py` directly) |
| Deployed checkpoint untouched | PASS |
| Candidate checkpoint separate from deployed checkpoint | PASS |
| No TEST-derived information used for model/seed selection | PASS — the seed=43 selection in `docs/experiment_fe_dominant_candidate_review.md` was made entirely from VAL evidence, before this evaluation began |
| **TEST structural features consistent between candidate and the existing TEST data the baseline was measured on** | **FAILED — see below** |

### What was found

Building this experiment's TEST structural graphs (same procedure
already used for TRAIN/VAL: the one-line-changed
`scratch_external_eval/graph_builder_dominant_ext.py`, which imports the
**current** `ml/preprocessing/ast_parser.py` / `metrics.py`) and
comparing them, column-by-column, against the **existing**
`data/processed/graphs_hybrid/test` (the file the official deployed
TEST baseline of 0.740/0.525/0.767/0.677 was measured against) revealed
that **798 of 1574 TEST files (51%) differ outside the one intended
column** — not just column 4:

| repo | files | mismatched |
|---|---|---|
| celery | 336 | 206 (61%) |
| django | 859 | 378 (44%) |
| pandas | 379 | 205 (54%) |

Two distinct causes, both traced to real code:

1. **645 mismatches**: same node count, but `self_access_count`
   (feature column index 3) differs on specific rows. Root cause:
   the collaborator-chain / `is_chained_base` fix made to
   `ml/preprocessing/ast_parser.py` and `metrics.py` earlier this
   project (`docs/dataprocessor_sample_diagnosis.md`) — already
   reflected in TRAIN/VAL (rebuilt from it this session), but **the
   existing TEST hybrid graphs predate that fix** and still show the
   old, over-counted self-access values.
2. **144 mismatches**: the *method or function node count itself*
   differs between a fresh parse and the existing TEST graphs. Root
   cause: the nested-function/nested-class representation fix
   documented in `docs/robustness_audit_phase_a.md` ("a nested def was
   previously silently dropped entirely — never a graph node at all")
   — also already reflected in TRAIN/VAL, **also predating the existing
   TEST hybrid graphs**.

**This is not caused by Experiment #1 and is not a bug in this
evaluation script** — it is a pre-existing inconsistency between
TRAIN/VAL (rebuilt multiple times this project cycle to pick up parser
fixes) and TEST (deliberately never rebuilt, per this project's own
frozen-TEST discipline — `scripts/rebuild_train_val_data.py`'s own
docstring: *"TEST is deliberately excluded... not read, re-parsed,
re-labeled, or re-built"*). That discipline was followed correctly by
every prior script; the side effect, only surfaced now because this is
the first time a TEST structural rebuild was attempted at all this
project cycle, is that **the official deployed TEST baseline
(0.740/0.525/0.767/0.677) was itself measured against structural
features that predate at least two parser fixes the deployed model's
own TRAIN/VAL data already reflects.**

### Why evaluation was not attempted anyway

Patching column 4 on top of files that already differ on columns 0-3
would silently evaluate the candidate against a **different,
uncontrolled mix of changes** (the intended `dominant_external_count`
swap *plus* two unrelated, already-shipped parser fixes) rather than
the single isolated variable this experiment was authorized to test.
That would not be a valid comparison against the 0.740/0.525/0.767/0.677
baseline, and — per the explicit rule — any unexpected modification
found before evaluation means **stop, not improvise a workaround**.

## 2. Candidate

`models/experiment_fe_dominant/seed_43/` — **not evaluated on TEST**.
No inference was run; the script aborted at the pre-evaluation
verification gate (`scripts/experiment_fe_dominant_test_eval.py`,
step 2), before step 3 (the actual forward pass) ever executed.

## 3-6. TEST Results / Comparison / Interpretation / Final Status

**Not applicable — no TEST evaluation occurred.** No number in this
document should be read as a TEST result. The protected baseline
(Long Method 0.740, Feature Envy 0.525, God Class 0.767, Macro-F1
0.677) is unchanged, unmodified, and not reinterpreted.

## 7. Deployment Status

"Candidate NOT deployed. Deployment decision pending."
(Trivially true — no TEST result exists yet to base a deployment
decision on.)

---

## What TEST access actually happened

Per the authorization ("you may load the official TEST set... run
inference"), TEST source was parsed and the **existing** TEST hybrid
graphs were **read** (never written to) for the verification check
above. **No TEST file was modified.** New files were written only to
two new, separate directories:
`data/processed/graphs_experiment_fe_dominant/test/` (structural,
experiment feature) and 776 partial files under
`data/processed/graphs_hybrid_experiment_fe_dominant/test/` (the
patch loop's output before it aborted — an incomplete, unevaluated
byproduct, not used for anything). **The one-time TEST evaluation has
NOT been consumed** — no metric was computed, no inference was run, the
"evaluate exactly once" budget is intact.

## Options, for you to decide — not chosen or acted on here

1. **Evaluate the candidate against TEST as-is (stale features)**,
   accepting that the comparison would mix the intended
   `dominant_external_count` change with two already-shipped-in-TRAIN/VAL
   parser fixes that TEST has never received. This would tell you
   something, but not a clean single-variable answer.
2. **Rebuild TEST's structural+hybrid graphs once, using the current
   (already-in-TRAIN/VAL) parser fixes but the *original*
   `external_access_count` feature**, establishing a new, fixed-parity
   TEST baseline first — then evaluate both the deployed baseline
   checkpoint and the seed=43 candidate against that *same*, currently-consistent
   TEST data. This is the only way to get a true single-variable
   comparison, but it is a meaningfully bigger action than what was
   authorized in this task (it touches/regenerates TEST data, which
   this task's instructions explicitly listed under "you may NOT ...
   modify datasets").
3. **Hold entirely** — treat this as a finding to note for later, and
   make no TEST-based decision this cycle.

No option was selected or acted on. Stopping here, exactly as
instructed, and waiting for your explicit direction.
