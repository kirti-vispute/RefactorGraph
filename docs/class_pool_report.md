# Class-Method-Pool Architecture Fix Report (Phase 13b, Design B)

Design A (EDGE_TYPES_CLASS_AGGREGATION, docs/class_aggregation_report.md) fixed god_class's receptive field but regressed long_method and feature_envy too. Root cause: (class, contains, method) is method's only incoming edge, so once belongs_to lets class absorb its methods' embeddings, conv2 folds that sibling-aggregate signal back into every method's own embedding — which all three heads read.

**Design B:** class_method_pool=True keeps the shared backbone on plain EDGE_TYPES (no belongs_to edge wired into conv1/conv2 at all) and instead mean-pools conv2's method embeddings into a per-class vector, using the existing (class, contains, method) edge_index, fed only to head_god_class (concatenated with class's own conv2 embedding). method/function embeddings are structurally untouched by this — see tests/test_gat_baseline.py::test_class_pool_does_not_alter_method_pathway, which asserts h_dict['method'] and the long_method/feature_envy head outputs are byte-for-byte identical whether or not pooling is enabled.

Reused the exact Phase 10 tuned hyperparameters (models/hybrid_tuned/best_params.json: hidden_dim=128, heads=4, dropout=0.1033, lr=1.49e-03, weight_decay=2.34e-05) — no re-tuning. Trained on TRAIN split (max_epochs=150, patience=15, best_epoch=51), evaluated on VAL split only. TEST split untouched.


## Comparison: Phase 10 tuned vs Design A (reverse edge) vs Design B (pooling)

| task | Phase 10 F1 | Design A F1 | Design B F1 | Phase10->A | Phase10->B | Phase 10 prec | Design B prec | Phase 10 recall | Design B recall | Phase 10 ROC-AUC | Design B ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long_method | 0.831 | 0.789 | 0.791 | -0.042 | -0.040 | 0.812 | 0.712 | 0.851 | 0.892 | 0.989 | 0.987 |
| feature_envy | 0.351 | 0.291 | 0.313 | -0.060 | -0.037 | 0.261 | 0.243 | 0.532 | 0.442 | 0.927 | 0.847 |
| god_class | 0.794 | 0.762 | 0.786 | -0.032 | -0.008 | 0.864 | 0.812 | 0.735 | 0.762 | 0.971 | 0.958 |

## long_method (Design B raw)

- precision=0.712 recall=0.892 f1=0.791 roc_auc=0.987 pr_auc=0.899 (n=14802, positives=1162, tp=1036 fp=420 fn=126 tn=13220)

## feature_envy (Design B raw)

- precision=0.243 recall=0.442 f1=0.313 roc_auc=0.847 pr_auc=0.268 (n=12936, positives=278, tp=123 fp=384 fn=155 tn=12274)

## god_class (Design B raw)

- precision=0.812 recall=0.762 f1=0.786 roc_auc=0.958 pr_auc=0.847 (n=2612, positives=181, tp=138 fp=32 fn=43 tn=2399)