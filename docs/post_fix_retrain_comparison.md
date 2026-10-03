# Post-Fix Retrain Comparison -- seed=43 (promoted after seed-variance check)

Same architecture + hyperparameters as the frozen models/hybrid_class_pool_tuned checkpoint, same God-Class-rule-corrected TRAIN/VAL data as the seed=42 run (docs/post_fix_retrain_comparison_step3_godclass_seed42.md), but seed=43 -- selected via scratch_external_eval/godclass_seed_sweep.py as the best macro-F1 among seeds {42,43,44,45}, see docs/godclass_formula_revision.md section 9/10 for why. TEST untouched throughout.


## A) frozen model on OLD val  vs  B) frozen model on NEW val  vs  C) retrained model (seed=43) on NEW val

| task | A: old model / old val | B: old model / new val | C: new model / new val |
|---|---|---|---|
| long_method | 0.789 | 0.753 | 0.806 |
| feature_envy | 0.352 | 0.348 | 0.475 |
| god_class | 0.801 | 0.819 | 0.796 |
| **macro-F1** | 0.648 | 0.640 | 0.693 |

## long_method (new model, new val, raw)

- precision=0.727 recall=0.904 f1=0.806 roc_auc=0.988 pr_auc=0.910 (n=15576, positives=1205, tp=1089 fp=408 fn=116 tn=13963)

## feature_envy (new model, new val, raw)

- precision=0.374 recall=0.653 f1=0.475 roc_auc=0.951 pr_auc=0.475 (n=12966, positives=259, tp=169 fp=283 fn=90 tn=12424)

## god_class (new model, new val, raw)

- precision=0.776 recall=0.817 f1=0.796 roc_auc=0.978 pr_auc=0.886 (n=2612, positives=208, tp=170 fp=49 fn=38 tn=2355)