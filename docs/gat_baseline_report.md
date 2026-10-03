# GAT Baseline Report (Phase 7 — shallow 2-layer HeteroGAT, structure-only)

Best epoch: 100 (early stopping patience=15, max_epochs=150, stopped at epoch 115). Node features are the same placeholder structural metrics as the Phase 6 baseline (no CodeBERT semantics yet) — this isolates the effect of graph message passing alone. hidden_dim=32, heads=2, dropout=0.2, 2 GAT layers (shallow, to avoid over-smoothing per the staged training plan), residual connections around each conv layer (see ml/models/gat_baseline.py docstring — required because add_self_loops=False on bipartite edge types would otherwise erase a node's own pre-conv features).


## long_method

- precision=0.763 recall=0.990 f1=0.861 roc_auc=0.993 pr_auc=0.976 (n=14802, positives=1162, tp=1150 fp=358 fn=12 tn=13282)

## feature_envy

- precision=0.319 recall=0.871 f1=0.467 roc_auc=0.964 pr_auc=0.521 (n=12936, positives=278, tp=242 fp=517 fn=36 tn=12141)

## god_class

- precision=0.731 recall=0.989 f1=0.840 roc_auc=0.998 pr_auc=0.968 (n=2612, positives=181, tp=179 fp=66 fn=2 tn=2365)

## Comparison against Phase 6 metrics-only baseline (random_forest, same val split)

| task | RF F1 | GAT F1 | RF PR-AUC | GAT PR-AUC |
|---|---|---|---|---|
| long_method | 1.000 | 0.861 | 1.000 | 0.976 |
| feature_envy | 0.775 | 0.467 | 0.861 | 0.521 |
| god_class | 0.994 | 0.840 | 1.000 | 0.968 |

RF matches or beats the GAT on every task here. This is expected, not a failure of the GAT: Long Method and God/Large Class labels are deterministic (or near-deterministic) thresholds on the exact raw features RF sees directly, while the GAT only sees those features after a Linear encoder + 2 rounds of neighbor-mixing + ReLU — that compression necessarily loses some of the precise raw magnitude a hard threshold needs. For Feature Envy, RF can carve out the exact pairwise comparison rule (dominant_external_count vs self_access_count) with a couple of tree splits; the GAT has to reconstruct an equivalent decision boundary from mixed, ReLU'd embeddings, which is a harder optimization problem for a 2-layer model with this little data (547 training graphs). Graph structure is not expected to pay off against a metrics-only baseline until it is combined with CodeBERT semantics and evaluated on genuinely ambiguous cases (e.g. the Visitor-pattern false positives documented in docs/label_review_findings.md) rather than on labels that are themselves thresholds on the input features.
