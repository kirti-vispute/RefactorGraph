# Class-Method-Pool Fixed-Architecture Tuning Report (Phase 13e, Design B)

Follow-up to docs/class_pool_tuning_report.md (Phase 13c). Analyzing that search's per-task-logged trials (models/hybrid_class_pool_tuned/optuna_trials.csv, added in Phase 13d) showed the best single trial for long_method (heads=4, hidden_dim=128, essentially Phase 10's own architecture) scored long_method_f1=0.804, close to Phase 10's 0.831 — but lost on macro-F1 to smaller architectures that traded long_method capacity for feature_envy/god_class gains. This run pins the architecture to Phase 10's own (heads=4, hidden_dim=128) and searches only dropout/lr/weight_decay/weight_fe/weight_gc, to isolate whether loss reweighting alone (without shrinking capacity) can close long_method's gap while keeping the feature_envy/god_class gains from Phase 13c.

TPE sampler + median pruner, seed=42, 20 trials, search budget max_epochs=60/patience=10. Architecture fixed: heads=4, hidden_dim=128, class_method_pool=True. Search space: dropout in [0.0, 0.5], lr log-uniform [1e-4, 1e-2], weight_decay log-uniform [1e-6, 1e-2], weight_fe in [0.5, 4.0], weight_gc in [0.5, 4.0] (weight_lm fixed at 1.0). Objective: macro-F1. Winning config retrained with the full budget (max_epochs=150, patience=15) for the numbers below.

Only TRAIN and VAL splits are touched. TEST split remains untouched.

Best params: dropout=0.005, lr=3.00e-03, weight_decay=9.53e-04, weight_fe=0.971, weight_gc=2.154, best_epoch=56.


## Comparison: Phase 10 vs untuned Design B vs Design B tuned (free arch, Phase 13c) vs Design B tuned (fixed arch, Phase 13e) — same val split

| task | Phase 10 | Design B untuned | Design B tuned (free arch) | Design B tuned (fixed arch) |
|---|---|---|---|---|
| long_method | 0.831 | 0.791 | 0.789 | 0.810 |
| feature_envy | 0.351 | 0.313 | 0.352 | 0.331 |
| god_class | 0.794 | 0.786 | 0.801 | 0.784 |
| **macro-F1** | 0.659 | 0.630 | 0.648 | 0.642 |

## long_method (fixed-arch tuned Design B raw)

- precision=0.773 recall=0.852 f1=0.810 roc_auc=0.979 pr_auc=0.887 (n=14802, positives=1162, tp=990 fp=291 fn=172 tn=13349)

## feature_envy (fixed-arch tuned Design B raw)

- precision=0.244 recall=0.514 f1=0.331 roc_auc=0.858 pr_auc=0.309 (n=12936, positives=278, tp=143 fp=442 fn=135 tn=12216)

## god_class (fixed-arch tuned Design B raw)

- precision=0.819 recall=0.751 f1=0.784 roc_auc=0.963 pr_auc=0.858 (n=2612, positives=181, tp=136 fp=30 fn=45 tn=2401)