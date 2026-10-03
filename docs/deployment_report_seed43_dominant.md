# DEPLOYMENT REPORT — seed_43 / dominant_external_count candidate

Controlled deployment of the TEST-validated candidate
(`models/experiment_fe_dominant/seed_43/`), authorized after
`docs/final_test_evaluation_AB_report.md` showed macro-F1 0.7029 vs
0.6585 for the corrected deployed baseline, apples-to-apples, all 3
tasks improved.

## A. Pre-deployment state

- Production model: `models/hybrid_class_pool_tuned_fixed_data/`
  (`backend/app/inference.py` `MODEL_DIR`), loaded via
  `HeteroGAT(class_method_pool=True)`, feature column 4 =
  `external_access_count`.
- Production feature construction: `ml/graph/graph_builder.py` line 183,
  `float(fm.external_access_count)`.
- Live thresholds: Feature Envy 0.65, Long Method/God Class 0.50
  (`backend/app/inference.py` lines 184/185/195/206).
- Backend/frontend otherwise unmodified this deployment.

## B. Backup / rollback location

`models/_deployment_backups/20260903_222923/`:
- `hybrid_class_pool_tuned_fixed_data/` — full copy of the previously
  deployed checkpoint directory (`model.pt`, `norm_stats.pt`,
  `best_params.json`, `final_val_metrics.json`, `errors_val.json`,
  `explanations_val.json`), byte-identical to the original (verified by
  file size, all 6 files match the source directory exactly).
- `inference.py.bak` — `backend/app/inference.py` before the `MODEL_DIR`
  change.
- `graph_builder.py.bak` — `ml/graph/graph_builder.py` before the
  feature-column change.

The original checkpoint directory itself,
`models/hybrid_class_pool_tuned_fixed_data/`, was **not deleted or
modified** — it remains in place, byte-identical (`model.pt` mtime Sep 2
19:56, unchanged), independent of the backup copy above.

## C. Candidate checkpoint deployed

`models/experiment_fe_dominant/seed_43/` — unchanged since selection
(`model.pt` mtime Sep 3 09:40, same as every prior check this project
cycle). Not modified by this deployment; only *referenced* (via
`MODEL_DIR`), never copied over or renamed.

**Compatibility verification** (performed before any change):
- `state_dict` keys: 132/132 identical between old and new checkpoint,
  0 missing either direction, 0 shape mismatches.
- `norm_stats.pt` dims identical: module=1, class=771, method=773,
  function=773 (5 raw structural + 768 CodeBERT for method/function; 3
  raw + 768 CodeBERT for class) — same for both checkpoints.
- `best_params.json`: identical architecture hyperparameters
  (`hidden_dim=64, heads=2, dropout=0.174, class_method_pool=true`);
  only `seed`/`best_epoch` differ (both `seed=43`).
- **Result: fully compatible. No STOP condition triggered.**

## D. Production code change made

Two files, one line each — the minimum necessary change:

1. `backend/app/inference.py` line 54:
   ```
   MODEL_DIR = ROOT / "models" / "hybrid_class_pool_tuned_fixed_data"
   ```
   →
   ```
   MODEL_DIR = ROOT / "models" / "experiment_fe_dominant" / "seed_43"
   ```

2. `ml/graph/graph_builder.py` line 183 (method/function feature vector,
   column index 4):
   ```
   float(fm.self_access_count), float(fm.external_access_count),
   ```
   →
   ```
   float(fm.self_access_count), float(fm.dominant_external_count),
   ```

Nothing else in either file was touched. The human-readable
`explanation` text field in `inference.py`'s `_explanation_for` (a
diagnostic string shown in the UI, not a model input) still reports
`external_access_count` as a separate, legitimately-computed metric
from `function_metrics` — this is display-only, not part of the model's
feature vector, and was left as-is per "preserve every other production
... behavior except the minimum required feature change."

## E. Threshold verification

Re-read directly after the code change:

| line | code | status |
|---|---|---|
| 184 | `lm[i] >= 0.5` (long_method) | unchanged |
| 185 | `fe[i] >= 0.65` (feature_envy) | unchanged |
| 195 | `lm_f[i] >= 0.5` (long_method, function) | unchanged |
| 206 | `gc[i] >= 0.5` (god_class) | unchanged |

No threshold was touched at any point in this deployment.

## F. Tests / build / health results

- `pytest tests/test_graph_builder.py tests/test_metrics.py tests/test_label_rules.py tests/test_backend_api.py tests/test_node_naming.py tests/test_explain.py`
  — **52 passed, 0 failed.**
- `pytest tests/` (full suite, all backend/ML tests including
  `test_backend_api.py`, which loads the model via the now-changed
  `MODEL_DIR` and runs real `/analyze` requests through the FastAPI
  `TestClient`) — **94 passed, 0 failed.**
- `npm run build` (frontend production build, `frontend/`) — succeeded,
  `dist/` produced, no errors (only a pre-existing chunk-size advisory
  warning, unrelated to this change).
- Backend startup: `uvicorn backend.app.main:app` started cleanly,
  loaded the candidate checkpoint, no exceptions.
- `/health`: `{"status":"ok","model_loaded":true}`.

No test file, threshold, or fixture was modified to make anything pass.

## G. Live inference verification

Sent a real `/analyze` request (non-TEST, synthetic source file with a
deliberately long constructor/method, a class with heavy external
attribute access on a single collaborator, and two ordinary classes) to
the running server with the deployed candidate:

- **Long Method**: `Order.__init__` (p=0.927, predicted=True),
  `Order.compute_total` (p=0.982, predicted=True); all 9 other short
  methods correctly predicted=False (p<0.06 each).
- **Feature Envy**: `PricingHelper.summarize` (p=0.998,
  predicted=True) — correctly flagged for dominantly accessing a single
  external object's (`order`) attributes rather than its own; all other
  methods correctly predicted=False (p≈0).
- **God Class**: both classes correctly predicted=False (too small to
  qualify under either branch of the current formula).
- Response schema unchanged: `summary`, `graph.nodes`, `graph.edges`,
  `long_method`, `feature_envy`, `god_class` — same shape as before this
  deployment (confirmed by `test_backend_api.py`'s schema assertions
  also passing).
- Thresholds observably active: every Feature Envy `predicted=True` had
  probability ≥0.65 exactly per the threshold; every Long Method/God
  Class `predicted=True`/`False` split exactly at 0.5.

## H. Historical artifacts preserved

Re-checked by direct file inspection after deployment:

| artifact | status |
|---|---|
| `data/processed/graphs/test/**` (historical structural TEST) | unchanged (sample file mtime Aug 24, untouched) |
| `data/processed/graphs_hybrid/test/**` (historical hybrid TEST) | unchanged (sample file mtime Aug 31, untouched) |
| `docs/final_test_evaluation_report.md` (Phase 16, historical 0.740/0.525/0.767/0.677) | unchanged (mtime Sep 2 20:17) |
| `docs/final_test_evaluation_AB_report.md` + `final_test_evaluation_ab_raw.json` (this cycle's A/B result) | unchanged since being written last turn |
| `docs/test_label_audit_report.md`, `docs/godclass_formula_revision.md` (prior experiment reports) | unchanged |
| `models/hybrid_class_pool_tuned_fixed_data/model.pt` (original checkpoint) | present, unchanged, NOT deleted (mtime Sep 2 19:56) |
| `models/experiment_fe_dominant/seed_43/model.pt` (candidate checkpoint) | present, unchanged (mtime Sep 3 09:40) — not copied or renamed, only referenced |

No TEST data was read, written, or used anywhere in this deployment
task. No TEST inference was run.

## I. Rollback procedure

To revert to the previous production state:

1. `backend/app/inference.py` line 54 — restore:
   `MODEL_DIR = ROOT / "models" / "hybrid_class_pool_tuned_fixed_data"`
   (or copy back `models/_deployment_backups/20260903_222923/inference.py.bak`).
2. `ml/graph/graph_builder.py` line 183 — restore:
   `float(fm.self_access_count), float(fm.external_access_count),`
   (or copy back `models/_deployment_backups/20260903_222923/graph_builder.py.bak`).
3. Restart the backend process (`uvicorn backend.app.main:app`) so
   `ModelBundle.__init__` reloads from the restored `MODEL_DIR`.
4. No checkpoint file needs restoring — the original checkpoint
   (`models/hybrid_class_pool_tuned_fixed_data/`) was never modified or
   deleted; the backup at
   `models/_deployment_backups/20260903_222923/hybrid_class_pool_tuned_fixed_data/`
   exists purely as an extra safety copy and is not itself required for
   rollback.

## J. Final deployment status

**DEPLOYED.**

- Production now loads `models/experiment_fe_dominant/seed_43/` via
  `backend/app/inference.py::MODEL_DIR`.
- Production graph construction now supplies `dominant_external_count`
  (`ml/graph/graph_builder.py` line 183), matching the representation
  the deployed candidate was trained on.
- Live thresholds unchanged (FE 0.65, LM/GC 0.50).
- GNNExplainer remains offline-only — not touched, not wired into
  `/analyze`.
- No PGExplainer, no PyExamine, no automatic source rewriting added.
- Model score is reported only as `probability`/`predicted` in the API
  — no confidence or accuracy claim is made anywhere in the response or
  this report.
- 94/94 backend tests pass, frontend build succeeds, `/health` reports
  `model_loaded:true`, live `/analyze` requests produce correct,
  threshold-consistent predictions for all 3 tasks.
- Historical checkpoint, historical TEST data, and all prior evaluation
  reports are confirmed unmodified.
