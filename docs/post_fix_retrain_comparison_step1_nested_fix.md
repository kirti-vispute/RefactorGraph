# Post-Fix Retrain Comparison (docs/robustness_audit_phase_a.md)

Same architecture + hyperparameters as the frozen models/hybrid_class_pool_tuned checkpoint, retrained on the graphs rebuilt after the parser/graph fixes. TEST untouched throughout.


## A) frozen model on OLD val  vs  B) frozen model on NEW val  vs  C) retrained model on NEW val

| task | A: old model / old val | B: old model / new val | C: new model / new val |
|---|---|---|---|
| long_method | 0.789 | 0.756 | 0.825 |
| feature_envy | 0.352 | 0.350 | 0.352 |
| god_class | 0.801 | 0.799 | 0.802 |
| **macro-F1** | 0.648 | 0.635 | 0.660 |

A vs B isolates what changed about VAL itself (labels/graph population) with the model held constant. B vs C isolates whether retraining on the corrected data helps, on the SAME (new) evaluation set -- this is the fair, single-variable comparison. A vs C is NOT a controlled comparison (both the model and the evaluation set changed) and should not be read as "the fix improved/hurt performance by X" on its own.


## long_method (new model, new val, raw)

- precision=0.767 recall=0.894 f1=0.825 roc_auc=0.990 pr_auc=0.911 (n=15576, positives=1205, tp=1077 fp=328 fn=128 tn=14043)

## feature_envy (new model, new val, raw)

- precision=0.299 recall=0.429 f1=0.352 roc_auc=0.889 pr_auc=0.318 (n=12966, positives=259, tp=111 fp=260 fn=148 tn=12447)

## god_class (new model, new val, raw)

- precision=0.798 recall=0.807 f1=0.802 roc_auc=0.977 pr_auc=0.886 (n=2612, positives=181, tp=146 fp=37 fn=35 tn=2394)