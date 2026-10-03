# Post-Fix Retrain Comparison (docs/robustness_audit_phase_a.md)

Same architecture + hyperparameters as the frozen models/hybrid_class_pool_tuned checkpoint, retrained on the graphs rebuilt after the parser/graph fixes. TEST untouched throughout.


## A) frozen model on OLD val  vs  B) frozen model on NEW val  vs  C) retrained model on NEW val

| task | A: old model / old val | B: old model / new val | C: new model / new val |
|---|---|---|---|
| long_method | 0.789 | 0.753 | 0.791 |
| feature_envy | 0.352 | 0.348 | 0.393 |
| god_class | 0.801 | 0.799 | 0.780 |
| **macro-F1** | 0.648 | 0.633 | 0.655 |

A vs B isolates what changed about VAL itself (labels/graph population) with the model held constant. B vs C isolates whether retraining on the corrected data helps, on the SAME (new) evaluation set -- this is the fair, single-variable comparison. A vs C is NOT a controlled comparison (both the model and the evaluation set changed) and should not be read as "the fix improved/hurt performance by X" on its own.


## long_method (new model, new val, raw)

- precision=0.697 recall=0.915 f1=0.791 roc_auc=0.990 pr_auc=0.907 (n=15576, positives=1205, tp=1102 fp=480 fn=103 tn=13891)

## feature_envy (new model, new val, raw)

- precision=0.305 recall=0.552 f1=0.393 roc_auc=0.939 pr_auc=0.371 (n=12966, positives=259, tp=143 fp=326 fn=116 tn=12381)

## god_class (new model, new val, raw)

- precision=0.733 recall=0.834 f1=0.780 roc_auc=0.976 pr_auc=0.874 (n=2612, positives=181, tp=151 fp=55 fn=30 tn=2376)