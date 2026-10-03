# Class-Aggregation Architecture Fix Report (Phase 13)

**Finding (Phase 12 GNNExplainer audit):** under the original schema, (class, contains, method) only ever makes `method` the destination — `class` is never the destination of any edge derived from its methods, so a class node's embedding can never aggregate information from its own methods during message passing. god_class predictions saw only the class's own encoded metrics, module, and inherited-class chain.

**Fix:** the smallest schema change that closes this gap — a single reverse edge type, (method, belongs_to, class), added in ml/graph/graph_builder.py and wired in via ml/models/gat_baseline.EDGE_TYPES_CLASS_AGGREGATION. No other edge types were added, and no labels changed. The original EDGE_TYPES constant and every existing checkpoint (models/hybrid_tuned, models/hybrid_baseline, models/gat_baseline) are untouched and still load exactly as before — HeteroConv only wires convolutions for the edge types a model was constructed with, and ignores extra edge type keys in the data (verified against torch_geometric.nn.HeteroConv.forward before regenerating any graphs).

Reused the exact Phase 10 tuned hyperparameters (models/hybrid_tuned/best_params.json: hidden_dim=128, heads=4, dropout=0.1033, lr=1.49e-03, weight_decay=2.34e-05) — no re-tuning, so any metric delta below is attributable to the schema/architecture change alone, not a hyperparameter difference. Trained on TRAIN split (full budget, max_epochs=150, patience=15, best_epoch=43), evaluated on VAL split only. TEST split untouched.


## Comparison vs Phase 10 tuned hybrid (same val split, same hyperparameters)

| task | Phase 10 F1 | Phase 13 F1 | delta F1 | Phase 10 precision | Phase 13 precision | Phase 10 recall | Phase 13 recall | Phase 10 ROC-AUC | Phase 13 ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| long_method | 0.831 | 0.789 | -0.042 | 0.812 | 0.693 | 0.851 | 0.916 | 0.989 | 0.987 |
| feature_envy | 0.351 | 0.291 | -0.060 | 0.261 | 0.189 | 0.532 | 0.637 | 0.927 | 0.916 |
| god_class | 0.794 | 0.762 | -0.032 | 0.864 | 0.826 | 0.735 | 0.707 | 0.971 | 0.920 |

## long_method (Phase 13 raw)

- precision=0.693 recall=0.916 f1=0.789 roc_auc=0.987 pr_auc=0.898 (n=14802, positives=1162, tp=1064 fp=471 fn=98 tn=13169)

## feature_envy (Phase 13 raw)

- precision=0.189 recall=0.637 f1=0.291 roc_auc=0.916 pr_auc=0.313 (n=12936, positives=278, tp=177 fp=761 fn=101 tn=11897)

## god_class (Phase 13 raw)

- precision=0.826 recall=0.707 f1=0.762 roc_auc=0.920 pr_auc=0.807 (n=2612, positives=181, tp=128 fp=27 fn=53 tn=2404)