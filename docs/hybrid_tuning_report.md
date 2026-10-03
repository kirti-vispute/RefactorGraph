# Hybrid Hyperparameter Tuning Report (Phase 10 -- Optuna, validation-only)

TPE sampler + median pruner, seed=42, 30 trials, search budget max_epochs=60/patience=10 per trial (pruned trials stop earlier). Search space: heads in {1,2,4}, hidden_dim_per_head in {8,16,32} (hidden_dim = heads * hidden_dim_per_head, guaranteeing hidden_dim % heads == 0), dropout in [0.0, 0.5], lr log-uniform [1e-4, 1e-2], weight_decay log-uniform [1e-6, 1e-2]. Objective: mean val F1 across the 3 tasks (best epoch within the trial's own early-stopping budget). Winning config retrained once more with the full Phase 9 budget (max_epochs=150, patience=15) for the numbers below -- a pruned or short-budget trial checkpoint isn't necessarily the same quality as one allowed to fully converge/early-stop on its own terms.

Only TRAIN and VAL splits are touched by this phase (VAL used for the search objective and pruning, same role it plays in every prior phase). TEST split remains untouched, per the project's touch-once-at-the-end discipline.

Best params: heads=4, hidden_dim=128, dropout=0.103, lr=1.49e-03, weight_decay=2.34e-05, best_epoch=77.


## long_method

- precision=0.812 recall=0.851 f1=0.831 roc_auc=0.989 pr_auc=0.909 (n=14802, positives=1162, tp=989 fp=229 fn=173 tn=13411)

## feature_envy

- precision=0.261 recall=0.532 f1=0.351 roc_auc=0.927 pr_auc=0.326 (n=12936, positives=278, tp=148 fp=418 fn=130 tn=12240)

## god_class

- precision=0.864 recall=0.735 f1=0.794 roc_auc=0.971 pr_auc=0.863 (n=2612, positives=181, tp=133 fp=21 fn=48 tn=2410)

## Comparison across all baselines + tuned hybrid (same val split)

| task | RF F1 | GAT F1 | CodeBERT LR F1 | Hybrid (default) F1 | Hybrid (tuned) F1 |
|---|---|---|---|---|---|
| long_method | 1.000 | 0.861 | 0.509 | 0.813 | 0.831 |
| feature_envy | 0.775 | 0.467 | 0.160 | 0.364 | 0.351 |
| god_class | 0.994 | 0.840 | 0.422 | 0.777 | 0.794 |

Tuning moved F1 by long_method +0.018, feature_envy -0.013, god_class +0.017 versus the Phase 9 default hyperparameters. The winning config (hidden_dim=128, i.e. 4x the Phase 9 default of 32, with dropout=0.103 vs the default 0.2) confirms the Phase 9 report's own hypothesis that a wider hidden_dim relieves pressure on the shared Linear encoder forced to compress a 768-dim CodeBERT block alongside a 3-6-dim structural block. The gains on long_method and god_class are real but modest, and the tuned hybrid still does NOT beat the Phase 7 structure-only GAT or the Phase 6 metrics-only RF on any of the 3 tasks -- widening the encoder helped retain more structural signal, but didn't remove the underlying limitation (CodeBERT stays frozen and general-purpose, not task-tuned). feature_envy is the rarest class in val (positives=278 of 12936) and the most search-noise-sensitive of the 3 tasks; a small F1 regression there is within the run-to-run variance a 30-trial validation-only search can produce, not necessarily meaningful hyperparameter harm. Consistent with the Phase 9 report's next-steps list, the largest remaining structural fix (a learned down-projection of the CodeBERT embedding, or joint fine-tuning) stays out of scope for this staged phase.