# Final TEST Evaluation Report (Phase 16)

One-time TEST evaluation, run after all architecture/hyperparameter/loss-weight decisions were made on TRAIN/VAL alone (Phase 9 through Phase 13e). This is the first and only script in the project to read data/raw/test. n_test_graphs=1574.


## Comparison: Phase 10 Hybrid vs Design B untuned vs Design B tuned -- TEST split

| task | Phase 10 Hybrid | Design B untuned | Design B tuned |
|---|---|---|---|
| long_method | 0.778 | 0.742 | 0.733 |
| feature_envy | 0.424 | 0.412 | 0.412 |
| god_class | 0.757 | 0.721 | 0.806 |
| **macro-F1** | 0.653 | 0.625 | 0.650 |

## Raw metrics per checkpoint


### Phase 10 Hybrid

- long_method: precision=0.740 recall=0.819 f1=0.778 roc_auc=0.975 pr_auc=0.852 (n=24046, positives=2301, tp=1885 fp=661 fn=416 tn=21084)
- feature_envy: precision=0.428 recall=0.419 f1=0.424 roc_auc=0.911 pr_auc=0.438 (n=20112, positives=632, tp=265 fp=354 fn=367 tn=19126)
- god_class: precision=0.808 recall=0.712 f1=0.757 roc_auc=0.971 pr_auc=0.845 (n=3766, positives=295, tp=210 fp=50 fn=85 tn=3421)

### Design B untuned

- long_method: precision=0.647 recall=0.870 f1=0.742 roc_auc=0.977 pr_auc=0.845 (n=24046, positives=2301, tp=2001 fp=1090 fn=300 tn=20655)
- feature_envy: precision=0.405 recall=0.419 f1=0.412 roc_auc=0.915 pr_auc=0.417 (n=20112, positives=632, tp=265 fp=389 fn=367 tn=19091)
- god_class: precision=0.698 recall=0.746 f1=0.721 roc_auc=0.950 pr_auc=0.802 (n=3766, positives=295, tp=220 fp=95 fn=75 tn=3376)

### Design B tuned

- long_method: precision=0.621 recall=0.894 f1=0.733 roc_auc=0.978 pr_auc=0.856 (n=24046, positives=2301, tp=2058 fp=1257 fn=243 tn=20488)
- feature_envy: precision=0.457 recall=0.375 f1=0.412 roc_auc=0.916 pr_auc=0.423 (n=20112, positives=632, tp=237 fp=282 fn=395 tn=19198)
- god_class: precision=0.809 recall=0.803 f1=0.806 roc_auc=0.970 pr_auc=0.870 (n=3766, positives=295, tp=237 fp=56 fn=58 tn=3415)