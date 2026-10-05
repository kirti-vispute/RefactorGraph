# Current Project Guide and Historical Reports

This guide describes the frozen deployment. Other reports record the
project stage at which they were written; their references to "current",
"candidate", "deployed", or "final" are historical unless listed below
as current. Their original results are preserved.

## Current Deployment

- Checkpoint: `models/experiment_fe_dominant/seed_43/`.
- Method/function features: `[loc, statement_count, param_count,
  self_access_count, dominant_external_count]`, followed by 768 frozen
  CodeBERT embedding values.
- Two-layer HeteroGAT, hidden dimension 64, two attention heads, dropout
  approximately 0.174, and Design B method-to-class pooling.
- Live thresholds: Long Method 0.50, Feature Envy 0.65, God Class 0.50.
- Live explanations show structural metrics. GNNExplainer is offline-only.
- Graph inputs and relationship limits: [graph schema](graph_schema.md).
- Current deployment evidence: [deployment report](deployment_report_seed43_dominant.md)
  and [project freeze report](PROJECT_FREEZE_REPORT.md).

## Dataset and Labels

Real Python source from 15 GitHub repositories is split by repository:

| Split | Repositories | Parsed Python files |
|---|---|---:|
| TRAIN | click, black, httpx, gunicorn, flask, tornado, requests, scrapy, pytest | 547 |
| VAL | sqlalchemy, starlette, sphinx | 514 |
| TEST | django, pandas, celery | 1574 |

Membership is defined in `configs/repos.yaml`; commit IDs are recorded
in `configs/repos.lock.yaml`. The repository sets do not overlap.
Dataset construction filters exact method/class duplicates; this does
not establish that all near-duplicates or similar code are absent.
Labels are rule-derived silver labels, not human-reviewed ground truth.

Current rules in `ml/preprocessing/label_rules.py`, with thresholds from
`configs/label_thresholds.json`:

- Long Method: `statement_count >= 15` for methods and functions.
- Feature Envy: eligible instance methods have
  `dominant_external_count >= 3` and
  `dominant_external_count > self_access_count`. Constructors,
  static methods, class methods, and standalone functions are excluded.
- God Class: `(method_count >= 10 AND loc >= 192) OR
  (loc >= 192 AND field_count >= 1 AND method_count >= 3)`.

The TRAIN/VAL investigation recorded 176 Feature Envy positives among
5219 eligible TRAIN methods and 259 among 11514 eligible VAL methods
([investigation report](feature_envy_generalization_investigation.md)).
These eligibility-filtered counts differ from earlier dataset-stage
counts. The corrected TEST label audit recorded Long Method 2367,
Feature Envy 554, and God Class 328 positives
([label audit](test_label_audit_report.md)). These are existing recorded
counts; no new TEST analysis is required.

## Final Results

The [corrected A/B evaluation](final_test_evaluation_AB_report.md) is the
authoritative final performance report. Its offline decision threshold
was 0.50 for all smells; live Feature Envy serving uses 0.65.

| Metric | Corrected original baseline | Deployed dominant-feature model |
|---|---:|---:|
| Long Method F1 | 0.7515 | 0.7740 |
| Feature Envy F1 | 0.4360 | 0.5272 |
| God Class F1 | 0.7879 | 0.8075 |
| Macro-F1 | 0.6585 | 0.7029 |

The earlier 0.740 / 0.525 / 0.767 / 0.677 result used the stale TEST
representation. It remains historical evidence and does not replace the
corrected evaluation. TEST is closed and the deployed model is frozen.

## Reading Historical Reports

- `dataset_report.md`: initial dataset, earlier labels, and AND-only
  God Class rule. Use the current rules above for the frozen system.
- `dataset_report_train_val_rebuild.md`: intermediate rebuild snapshot,
  before the final experiment; its counts are stage-specific.
- `godclass_formula_revision.md`, `generalization_audit.md`, and
  `final_freeze_audit.md`: audits of the earlier checkpoint. Their 0.677
  "final" claims describe that earlier stage.
- `experiment_fe_dominant_report.md` and
  `experiment_fe_dominant_candidate_review.md`: TRAIN/VAL experiment and
  selection records, before final evaluation and deployment.
- `final_test_evaluation_report.md` and
  `final_test_evaluation_report_pre_godclass_candidate.md`: historical
  evaluations; retain their original numbers.
- `explainability_report.md` and `explainability_report_candidate.md`:
  offline GNNExplainer studies of older checkpoints. No existing report
  here demonstrates GNNExplainer attributions for the final deployed
  dominant-feature checkpoint.
- `feature_envy_threshold_analysis.md` and
  `feature_envy_threshold_final_audit.md`: validation-only calibration of
  the earlier checkpoint. The audited 0.65 live threshold was retained
  unchanged during the final deployment.

Local datasets, older checkpoints, raw JSON outputs, and rollback
backups mentioned by these reports are not distributed in this GitHub
repository. The report text preserves the audit trail; it does not
guarantee that every historical experiment can run from a fresh clone.

## Packaging Verification

On 5 October 2026, an export containing only staged Git files passed
94 backend tests, 25 frontend tests, and the frontend production build.
Backend verification used the installed Python dependencies and cached
CodeBERT assets; frontend dependencies were installed using `npm ci`.
The frontend production-dependency audit reported zero vulnerabilities.
The full audit reported a high-severity advisory for `undici@7.29.0`,
which is pulled in by the `jsdom` development/test dependency. No
dependency versions were changed during this documentation pass.
