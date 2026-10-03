# CodeBERT Baseline Report (Phase 8 — frozen CodeBERT embeddings)

Logistic Regression + Random Forest trained on frozen microsoft/codebert-base mean-pooled embeddings (768-dim, max_length=256, no fine-tuning per the staged training plan). No hand-crafted metrics, no graph structure. Trained on TRAIN split, evaluated on VAL split. TEST split untouched.

**Caveat**: embeddings are truncated at 256 sub-word tokens — very long methods/classes lose their tail. Mean pooling over the attention mask is used instead of the raw [CLS] token, since CodeBERT's MLM/RTD pretraining gives no particular meaning to [CLS] as a whole-snippet summary.

**Caveat**: Random Forest here uses max_depth=10, min_samples_leaf=5 (unlike Phase 6's unregularized RF). An unregularized RF on these 768-dim dense embeddings memorized TRAIN almost perfectly (f1~1.0) and collapsed on VAL (f1 as low as 0.01) — confirmed via a train/val gap sweep. This regularized configuration was the best val-f1 point on that sweep; it is still consistently worse than Logistic Regression below, which is the more meaningful classifier for a frozen dense-embedding baseline.


## long_method

- **logistic_regression** (val): precision=0.350 recall=0.932 f1=0.509 roc_auc=0.948 pr_auc=0.588 (n=13905, positives=1160, tp=1081 fp=2004 fn=79 tn=10741)
- **random_forest** (val): precision=0.322 recall=0.697 f1=0.440 roc_auc=0.907 pr_auc=0.423 (n=13905, positives=1160, tp=809 fp=1705 fn=351 tn=11040)

## feature_envy

- **logistic_regression** (val): precision=0.094 recall=0.531 f1=0.160 roc_auc=0.848 pr_auc=0.130 (n=12048, positives=277, tp=147 fp=1409 fn=130 tn=10362)
- **random_forest** (val): precision=0.108 recall=0.083 f1=0.094 roc_auc=0.821 pr_auc=0.079 (n=12048, positives=277, tp=23 fp=190 fn=254 tn=11581)

## god_class

- **logistic_regression** (val): precision=0.305 recall=0.685 f1=0.422 roc_auc=0.882 pr_auc=0.368 (n=2575, positives=181, tp=124 fp=283 fn=57 tn=2111)
- **random_forest** (val): precision=0.353 recall=0.099 f1=0.155 roc_auc=0.830 pr_auc=0.256 (n=2575, positives=181, tp=18 fp=33 fn=163 tn=2361)

## Comparison against Phase 6 (metrics RF) and Phase 7 (GAT), same val split

| task | RF F1 | GAT F1 | CodeBERT RF F1 | RF PR-AUC | GAT PR-AUC | CodeBERT RF PR-AUC |
|---|---|---|---|---|---|---|
| long_method | 1.000 | 0.861 | 0.440 | 1.000 | 0.976 | 0.423 |
| feature_envy | 0.775 | 0.467 | 0.094 | 0.861 | 0.521 | 0.079 |
| god_class | 0.994 | 0.840 | 0.155 | 1.000 | 0.968 | 0.256 |

CodeBERT-only (best classifier here is Logistic Regression, not the table's RF column) underperforms both the Phase 6 metrics RF and the Phase 7 GAT on all 3 tasks. Expected, not a failure: Long Method and God/Large Class labels are near-deterministic thresholds on exact counts (statement_count, loc, method_count) — a 768-dim semantic embedding can correlate with code length but does not expose an exact countable magnitude the way the raw metric does, and the 256-token truncation discards the tail of longer methods/classes entirely, compounding this. Feature Envy fares worst of all three baselines tried so far (LR f1=0.160) since it is a pairwise comparison rule (dominant_external_count vs self_access_count) — the embedding sees only one method's own source text, and while self./other-object access patterns are lexically present, recovering the exact comparison from a pooled semantic vector is a much harder inference than reading the two counts directly. This motivates the Phase 9 hybrid model: CodeBERT semantics alone lose precise structural magnitude, and structure alone (Phase 7) lacks semantics — combining both is where a genuine improvement over the metrics-only floor is plausible.