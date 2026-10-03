# PROJECT FREEZE REPORT

Post-deployment audit, read-only. No code changed in this pass. All 10
checks below performed fresh this turn.

## 1-3. Production configuration — verified

| item | verified value |
|---|---|
| Loaded checkpoint (`backend/app/inference.py:54`) | `models/experiment_fe_dominant/seed_43/` |
| Feature column 4 (`ml/graph/graph_builder.py:183`) | `float(fm.dominant_external_count)` |
| Feature Envy threshold (line 185) | `>= 0.65` |
| Long Method threshold (lines 184, 195) | `>= 0.5` |
| God Class threshold (line 206) | `>= 0.5` |

## 4. Functional verification — all pass

- `/health` → `{"status":"ok","model_loaded":true}`
- `/analyze` (POST, synthetic non-TEST source) → HTTP 200
- Response contains all 3 outputs: `long_method` (n=11), `feature_envy`
  (n=11), `god_class` (n=2) — schema unchanged (`summary`, `graph`,
  `long_method`, `feature_envy`, `god_class`)
- `pytest tests/` → **94 passed, 0 failed**
- `npm run build` (frontend) → succeeded, `dist/` produced, no errors

## 5-6. Backup and old checkpoint — verified intact

- Rollback backup: `models/_deployment_backups/20260903_222923/` —
  `inference.py.bak`, `graph_builder.py.bak`, and a full copy of
  `hybrid_class_pool_tuned_fixed_data/` (6 files: `model.pt`,
  `norm_stats.pt`, `best_params.json`, `final_val_metrics.json`,
  `errors_val.json`, `explanations_val.json`) — all present, byte sizes
  match the live originals exactly.
- Old checkpoint `models/hybrid_class_pool_tuned_fixed_data/` — present,
  not deleted, `model.pt` mtime Sep 2 19:56 (unchanged since before this
  deployment).

## 7. Historical artifacts — verified untouched

- `data/processed/graphs/test/**` — sample file mtime Aug 24 (unchanged)
- `data/processed/graphs_hybrid/test/**` — sample file mtime Aug 31
  (unchanged)
- `docs/final_test_evaluation_report.md` (Phase 16 historical
  0.740/0.525/0.767/0.677) — unchanged, mtime Sep 2 20:17
- `docs/final_test_evaluation_AB_report.md`,
  `final_test_evaluation_ab_raw.json` (this cycle's A/B TEST result) —
  unchanged since written
- `docs/test_label_audit_report.md`, `docs/godclass_formula_revision.md`,
  `docs/experiment_fe_dominant_report.md`,
  `docs/experiment_fe_dominant_candidate_review.md`,
  `docs/experiment_fe_dominant_test_eval_STOPPED.md`,
  `docs/feature_envy_generalization_investigation.md`,
  `docs/deployment_report_seed43_dominant.md` — all unchanged

## 8. Exact production changes (diff against pre-deployment backup)

Two files, one line each — nothing else:

```diff
--- backend/app/inference.py (pre-deployment)
+++ backend/app/inference.py (current)
@@ -54 +54 @@
-MODEL_DIR = ROOT / "models" / "hybrid_class_pool_tuned_fixed_data"
+MODEL_DIR = ROOT / "models" / "experiment_fe_dominant" / "seed_43"
```

```diff
--- ml/graph/graph_builder.py (pre-deployment)
+++ ml/graph/graph_builder.py (current)
@@ -183 +183 @@
-                float(fm.self_access_count), float(fm.external_access_count),
+                float(fm.self_access_count), float(fm.dominant_external_count),
```

No other file differs from its pre-deployment backup. No test,
threshold, checkpoint, or dataset file was changed.

---

## Final deployed model

`models/experiment_fe_dominant/seed_43/` — `HeteroGAT(hidden_dim=64,
heads=2, dropout=0.174, class_method_pool=True)`, 269,315 trainable
parameters, seed=43, `best_epoch=79`.

## Final production feature representation

Method/function feature vector (5 raw + 768-dim CodeBERT): `[loc,
statement_count, param_count, self_access_count,
dominant_external_count]` — column 4 changed from
`external_access_count` (sum across all external receivers) to
`dominant_external_count` (largest single receiver), matching the
quantity the Feature Envy label rule is keyed on.

## Live thresholds

Feature Envy = 0.65, Long Method = 0.50, God Class = 0.50 — unchanged
throughout deployment and this audit.

## Final validated TEST results (one-time, closed after this)

Apples-to-apples, corrected TEST data (current parser, 1574 files),
each evaluated exactly once:

| task | Corrected deployed baseline (A) | Candidate seed-43 + dominant (B) | Δ |
|---|---|---|---|
| long_method | 0.7515 | 0.7740 | +0.0225 |
| feature_envy | 0.4360 | 0.5272 | +0.0912 |
| god_class | 0.7879 | 0.8075 | +0.0196 |
| **macro-F1** | **0.6585** | **0.7029** | **+0.0444** |

Historical production headline (measured on stale, pre-parser-fix TEST
data, preserved for reference, not comparable 1:1 to A): Long Method
0.740, Feature Envy 0.525, God Class 0.767, macro-F1 0.677.

## Deployment verification

`/health` model_loaded=true; `/analyze` returns all 3 outputs with
correct schema; 94/94 backend tests pass; frontend build succeeds;
live inference on a synthetic sample correctly identified an
engineered Feature Envy method (p=0.998) and two Long Method cases
(p=0.93, p=0.98), with thresholds observably applied at the exact
configured cutoffs.

## Rollback location

`models/_deployment_backups/20260903_222923/` (`inference.py.bak`,
`graph_builder.py.bak`, full copy of the original checkpoint
directory). Rollback = restore the two one-line edits listed in
Section 8 above (in reverse) and restart the backend; the original
checkpoint was never modified so no checkpoint restore is needed.

## Exact files changed (deployment total, this cycle)

- `backend/app/inference.py` — 1 line (`MODEL_DIR`)
- `ml/graph/graph_builder.py` — 1 line (feature column 4)

No other production, test, config, or data file was modified.

## TEST-touch confirmation

TEST was evaluated exactly twice total this cycle (Evaluation A,
Evaluation B — both part of the single authorized final TEST
evaluation), and has not been touched since. No TEST file was read,
written, or used in either the deployment task or this audit. No TEST
inference has run since `docs/final_test_evaluation_AB_report.md` was
produced.

## ML tuning recommendation

**No further ML tuning is recommended at this stage.** The candidate
was selected on TRAIN/VAL evidence, confirmed generalizing on a
held-out TEST split never used for any decision, and improves all
three tasks simultaneously with a coherent, mechanistically-explained
cause. TEST is now closed for this project cycle; any further model
change would require a new held-out evaluation set to remain
methodologically valid. Project state is frozen at this deployment.
