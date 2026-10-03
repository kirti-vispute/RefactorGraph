# Class-Method-Pool Optuna Tuning Report (Phase 13c, Design B)

TPE sampler + median pruner, seed=42, 30 trials, search budget max_epochs=60/patience=10 per trial. Architecture fixed to Design B (class_method_pool=True) for every trial — see docs/class_pool_report.md for why Design B was selected over Design A. Search space: heads in {1,2,4}, hidden_dim_per_head in {8,16,32}, dropout in [0.0, 0.5], lr log-uniform [1e-4, 1e-2], weight_decay log-uniform [1e-6, 1e-2], weight_fe in [0.5, 4.0], weight_gc in [0.5, 4.0] (weight_lm fixed at 1.0 — only the two ratios relative to loss_lm matter for gradient direction). Objective: macro-F1 (mean val F1 across the 3 tasks). Winning config retrained with the full budget (max_epochs=150, patience=15) for the numbers below.

Only TRAIN and VAL splits are touched. TEST split remains untouched.

Best params: heads=2, hidden_dim=64, dropout=0.174, lr=3.08e-03, weight_decay=2.36e-03, weight_fe=2.364, weight_gc=1.110, best_epoch=49.


## Comparison: Phase 10 hybrid vs untuned Design B vs tuned Design B (same val split)

| task | Phase 10 F1 | Design B (untuned) F1 | Design B (tuned) F1 | untuned->tuned |
|---|---|---|---|---|
| long_method | 0.831 | 0.791 | 0.789 | -0.002 |
| feature_envy | 0.351 | 0.313 | 0.352 | +0.039 |
| god_class | 0.794 | 0.786 | 0.801 | +0.015 |
| **macro-F1** | 0.659 | 0.630 | 0.648 | +0.017 |

## long_method (tuned Design B raw)

- precision=0.683 recall=0.935 f1=0.789 roc_auc=0.990 pr_auc=0.912 (n=14802, positives=1162, tp=1086 fp=504 fn=76 tn=13136)

## feature_envy (tuned Design B raw)

- precision=0.255 recall=0.568 f1=0.352 roc_auc=0.920 pr_auc=0.309 (n=12936, positives=278, tp=158 fp=461 fn=120 tn=12197)

## god_class (tuned Design B raw)

- precision=0.752 recall=0.856 f1=0.801 roc_auc=0.979 pr_auc=0.900 (n=2612, positives=181, tp=155 fp=51 fn=26 tn=2380)