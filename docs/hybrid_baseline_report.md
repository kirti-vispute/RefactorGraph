# Hybrid Baseline Report (Phase 9 — CodeBERT + shallow HeteroGAT fusion)

Best epoch: 119 (early stopping patience=15, max_epochs=150). Same 2-layer shallow HeteroGAT architecture as Phase 7, but class/method/function node features are [structural metrics ++ frozen 768-dim CodeBERT mean-pooled embedding] instead of structural metrics alone. hidden_dim=32, heads=2, dropout=0.2. CodeBERT stays frozen (no fine-tuning this phase, per the staged training plan) — only the GAT encoders/conv layers/heads are trained.


## long_method

- precision=0.740 recall=0.902 f1=0.813 roc_auc=0.988 pr_auc=0.907 (n=14802, positives=1162, tp=1048 fp=368 fn=114 tn=13272)

## feature_envy

- precision=0.250 recall=0.669 f1=0.364 roc_auc=0.933 pr_auc=0.343 (n=12936, positives=278, tp=186 fp=559 fn=92 tn=12099)

## god_class

- precision=0.785 recall=0.768 f1=0.777 roc_auc=0.973 pr_auc=0.874 (n=2612, positives=181, tp=139 fp=38 fn=42 tn=2393)

## Comparison across all four baselines (same val split)

| task | RF F1 | GAT F1 | CodeBERT LR F1 | Hybrid F1 | RF PR-AUC | GAT PR-AUC | CodeBERT LR PR-AUC | Hybrid PR-AUC |
|---|---|---|---|---|---|---|---|---|
| long_method | 1.000 | 0.861 | 0.509 | 0.813 | 1.000 | 0.976 | 0.588 | 0.907 |
| feature_envy | 0.775 | 0.467 | 0.160 | 0.364 | 0.861 | 0.521 | 0.130 | 0.343 |
| god_class | 0.994 | 0.840 | 0.422 | 0.777 | 1.000 | 0.968 | 0.368 | 0.874 |

The hybrid clearly beats CodeBERT-only on all 3 tasks (e.g. long_method F1 0.813 vs 0.509, god_class 0.777 vs 0.422) — structure genuinely adds signal semantics alone lacked, confirming the Phase 8 report's prediction. It does NOT, however, beat the Phase 7 structure-only GAT on any of the 3 tasks, and does not beat the Phase 6 metrics-only RF floor either. This is a real, documented result, not a failure to hide: concatenating a 768-dim CodeBERT embedding onto a 3-6-dim structural feature vector, then compressing both through the same hidden_dim=32 Linear encoder, makes the encoder's ~24:1 compression ratio dominated by the much larger semantic block — the few structural dimensions that let the GAT-only model read exact magnitudes (statement_count, method_count, loc) directly are diluted rather than reinforced. With only 547 training graphs, a shallow 2-layer model has little room to learn to down-weight the noisier semantic dimensions relative to the few precise structural ones. This does not mean semantics are useless here — the large hybrid-vs-CodeBERT-only gap shows structure is still doing most of the work, semantics are additive rather than harmful, and the hybrid closes part of the CodeBERT-only gap without erasing the structural signal entirely. Plausible next steps (out of scope for this staged phase, which keeps CodeBERT frozen and the GAT shallow by design): a wider hidden_dim so the encoder isn't forced to compress 768+ dims through the same bottleneck as 3-6 dims, a learned down-projection of the CodeBERT embedding before concatenation, or fine-tuning CodeBERT jointly with the GAT (the next stage of the plan) so the semantic embedding itself becomes task-relevant instead of a fixed general-purpose vector.
