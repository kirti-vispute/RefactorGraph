# Baseline Model Report (Phase 6 — metrics-only ML)

Logistic Regression + Random Forest trained on hand-crafted structural metrics only (no source text, no graph structure). Trained on TRAIN split, evaluated on VAL split. TEST split untouched — reserved for the final cross-model comparison.

**Caveat**: Long Method and God/Large Class labels are deterministic thresholds on a subset of these exact features, so near-ceiling scores on those two tasks are expected and mainly confirm label/feature consistency, not smell-detection ability in a general sense. Feature Envy's label is a genuine two-feature comparison rule, making it the more informative of the three baselines here.


## long_method

- **logistic_regression** (val): precision=1.000 recall=1.000 f1=1.000 roc_auc=1.000 pr_auc=1.000 (n=13905, positives=1160, tp=1160 fp=0 fn=0 tn=12745)
- **random_forest** (val): precision=1.000 recall=1.000 f1=1.000 roc_auc=1.000 pr_auc=1.000 (n=13905, positives=1160, tp=1160 fp=0 fn=0 tn=12745)

## feature_envy

- **logistic_regression** (val): precision=0.396 recall=0.978 f1=0.564 roc_auc=0.992 pr_auc=0.689 (n=12048, positives=277, tp=271 fp=413 fn=6 tn=11358)
- **random_forest** (val): precision=0.856 recall=0.708 f1=0.775 roc_auc=0.998 pr_auc=0.861 (n=12048, positives=277, tp=196 fp=33 fn=81 tn=11738)

## god_class

- **logistic_regression** (val): precision=0.726 recall=0.994 f1=0.839 roc_auc=0.997 pr_auc=0.962 (n=2575, positives=181, tp=180 fp=68 fn=1 tn=2326)
- **random_forest** (val): precision=1.000 recall=0.989 f1=0.994 roc_auc=1.000 pr_auc=1.000 (n=2575, positives=181, tp=179 fp=0 fn=2 tn=2394)