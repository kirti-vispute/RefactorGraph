# Post-Fix Retrain Comparison (docs/robustness_audit_phase_a.md)

Same architecture + hyperparameters as the frozen models/hybrid_class_pool_tuned checkpoint, retrained on the graphs rebuilt after the parser/graph fixes. TEST untouched throughout.


## A) frozen model on OLD val  vs  B) frozen model on NEW val  vs  C) retrained model on NEW val

| task | A: old model / old val | B: old model / new val | C: new model / new val |
|---|---|---|---|
| long_method | 0.789 | 0.753 | 0.822 |
| feature_envy | 0.352 | 0.348 | 0.412 |
| god_class | 0.801 | 0.819 | 0.801 |
| **macro-F1** | 0.648 | 0.640 | 0.678 |

A vs B isolates what changed about VAL itself (labels/graph population) with the model held constant. B vs C isolates whether retraining on the corrected data helps, on the SAME (new) evaluation set -- this is the fair, single-variable comparison. A vs C is NOT a controlled comparison (both the model and the evaluation set changed) and should not be read as "the fix improved/hurt performance by X" on its own.


## long_method (new model, new val, raw)

- precision=0.786 recall=0.861 f1=0.822 roc_auc=0.989 pr_auc=0.905 (n=15576, positives=1205, tp=1038 fp=283 fn=167 tn=14088)

## feature_envy (new model, new val, raw)

- precision=0.319 recall=0.583 f1=0.412 roc_auc=0.948 pr_auc=0.408 (n=12966, positives=259, tp=151 fp=323 fn=108 tn=12384)

## god_class (new model, new val, raw)

- precision=0.772 recall=0.832 f1=0.801 roc_auc=0.981 pr_auc=0.888 (n=2612, positives=208, tp=173 fp=51 fn=35 tn=2353)