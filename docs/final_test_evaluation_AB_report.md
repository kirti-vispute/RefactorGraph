# FINAL TEST EVALUATION

One-time, user-authorized TEST evaluation. Two forward passes run, each
exactly once, via `scripts/final_test_evaluation_ab.py`. Raw metrics
saved to `docs/final_test_evaluation_ab_raw.json`. No retry was
performed at any point. No model was retrained, tuned, or modified. No
checkpoint or threshold was changed. No deployment occurred.

## 1. Integrity Check

All checks re-verified immediately before inference, this turn:

| Check | Result |
|---|---|
| `data/processed/graphs_test_corrected_original` — 1574 structural files | PASS |
| `data/processed/graphs_hybrid_test_corrected_original` — 1574 hybrid files (CodeBERT-augmented this session) | PASS |
| `data/processed/graphs_hybrid_test_corrected_dominant` — 1574 hybrid files (patched from the above) | PASS |
| corrected_original vs corrected_dominant: 0 missing counterparts, 0 mismatches outside column 4, labels identical | PASS (`scripts/patch_test_corrected_dominant_hybrid.py` output: "Patched: 1574 / Missing: 0 / Mismatched: 0") |
| Candidate = `models/experiment_fe_dominant/seed_43/` exactly, unchanged since selection | PASS (`model.pt` timestamp Sep 3 09:40, unchanged) |
| Deployed/original baseline checkpoint = `models/hybrid_class_pool_tuned_fixed_data/` unchanged | PASS (`model.pt` timestamp Sep 2 19:56, unchanged) |
| Production `ml/graph/graph_builder.py` unchanged (`external_access_count`, not touched) | PASS |
| Production `backend/app/inference.py` unchanged | PASS |
| Live thresholds unchanged: Feature Envy 0.65, Long Method/God Class 0.50 | PASS |
| Historical TEST artifacts (`data/processed/graphs/test`, `data/processed/graphs_hybrid/test`) unchanged | PASS (read-only throughout; not read by this evaluation at all) |
| Model architecture identical between checkpoints | PASS — both report `trainable_params=269315`, `class_method_pool=True`, identical `best_params.json` hyperparameters (only `seed`/`best_epoch` differ, both seed=43) |

**Model-compatibility check**: both checkpoints were trained on the
5-column-raw + 768-dim-CodeBERT hybrid representation with identical
architecture (`HeteroGAT(hidden_dim=64, heads=2, class_method_pool=True)`).
The only difference between the checkpoints is what column 4 of the raw
features *meant* during their respective training (`external_access_count`
vs `dominant_external_count`) — not the tensor shape. Each checkpoint is
evaluated here against the TEST data built with the matching column-4
semantics (Evaluation A: original feature; Evaluation B: dominant
feature), and each uses its own saved `norm_stats.pt` (fit on its own
TRAIN split, never refit here). **Compatible — no STOP condition
triggered.**

## 2. Evaluation A — Corrected Original

Deployed checkpoint (`models/hybrid_class_pool_tuned_fixed_data`) on
`data/processed/graphs_hybrid_test_corrected_original` (current parser +
`external_access_count`). n=1574 graphs.

| task | precision | recall | F1 | ROC-AUC | PR-AUC | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|
| long_method | 0.6467 | 0.8969 | 0.7515 | 0.9830 | 0.8741 | 2123 | 1160 | 244 | 22072 |
| feature_envy | 0.3709 | 0.5289 | 0.4360 | 0.9386 | 0.4382 | 293 | 497 | 261 | 19369 |
| god_class | 0.7831 | 0.7927 | 0.7879 | 0.9784 | 0.8769 | 260 | 72 | 68 | 3366 |

**Macro-F1 (A) = 0.6585**

## 3. Evaluation B — Experiment #1 Seed 43

Candidate checkpoint (`models/experiment_fe_dominant/seed_43`) on
`data/processed/graphs_hybrid_test_corrected_dominant` (current parser +
`dominant_external_count`). n=1574 graphs.

| task | precision | recall | F1 | ROC-AUC | PR-AUC | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|
| long_method | 0.6802 | 0.8978 | 0.7740 | 0.9823 | 0.8425 | 2125 | 999 | 242 | 22233 |
| feature_envy | 0.5717 | 0.4892 | 0.5272 | 0.9501 | 0.5510 | 271 | 203 | 283 | 19663 |
| god_class | 0.8296 | 0.7866 | 0.8075 | 0.9777 | 0.8837 | 258 | 53 | 70 | 3385 |

**Macro-F1 (B) = 0.7029**

## 4. Apples-to-Apples Comparison

Same corrected TEST files, same labels, same evaluation methodology
(batch_size=64, shuffle=False, threshold=0.5) — only the intended
feature (and its matching checkpoint) differs.

| task | A (F1) | B (F1) | Δ (B − A) |
|---|---|---|---|
| long_method | 0.7515 | 0.7740 | **+0.0225** |
| feature_envy | 0.4360 | 0.5272 | **+0.0912** |
| god_class | 0.7879 | 0.8075 | **+0.0196** |
| **macro-F1** | **0.6585** | **0.7029** | **+0.0444** |

B improves on A for **all three tasks**, not just Feature Envy — the
task the feature change directly targets. Feature Envy shows by far the
largest gain, consistent with `dominant_external_count` being the exact
quantity the label rule is keyed on. The Feature Envy shift is a
precision/recall trade: A over-predicts positives (p=0.37, r=0.53,
fp=497) while B is much more precise (p=0.57, r=0.49, fp=203) — net
higher F1 (+0.091) with fewer false alarms, at a small recall cost (11
fewer true positives, tp 293→271).

## 5. Historical Production Result

**Preserved separately below — do not conflate with Evaluation A.**

The historical, previously-reported TEST baseline for the deployed
checkpoint (`models/hybrid_class_pool_tuned_fixed_data`, seed=43) is:

| task | Historical F1 (stale TEST) |
|---|---|
| long_method | 0.740 |
| feature_envy | 0.525 |
| god_class | 0.767 |
| **macro-F1** | **0.677** |

This number was measured against `data/processed/graphs_hybrid/test`
(unchanged, read-only throughout this whole investigation) — TEST data
that, per `docs/test_label_audit_report.md`, **predates two parser
fixes already reflected in the TRAIN/VAL data the deployed model was
trained on** (collaborator-chain self-access fix; nested-function
representation fix). It is the **same checkpoint** as Evaluation A, but
evaluated against **different, stale TEST features/labels**.

**This is not the same measurement as Evaluation A and the two numbers
must not be read as a before/after comparison of the same thing.**
Evaluation A (macro-F1 0.6585) is the deployed checkpoint's performance
on TEST data that is now parser-consistent with its own training data.
The historical number (macro-F1 0.677) is the deployed checkpoint's
performance on TEST data that was parser-inconsistent with its training
data. Per-task, the direction of the historical-vs-corrected difference
is not uniform:

| task | Historical (stale) | A (corrected, same checkpoint) | Δ |
|---|---|---|---|
| long_method | 0.740 | 0.7515 | +0.012 |
| feature_envy | 0.525 | 0.4360 | **−0.089** |
| god_class | 0.767 | 0.7879 | +0.021 |

Long Method and God Class both score *slightly higher* once measured on
corrected, parser-consistent TEST data. Feature Envy scores
*substantially lower* (−0.089) — the historical 0.525 was inflated by
the train/test parser mismatch, not a true measure of the deployed
model's generalization under a consistent pipeline. This is explored
further below.

## 6. Interpretation

**Does replacing `external_access_count` with `dominant_external_count`
improve TEST performance, holding everything else constant?** Yes.
Evaluation B outperforms Evaluation A on macro-F1 by +0.0444, and on
every individual task, under an identical, controlled, apples-to-apples
setup (Section 4).

**Is the improvement Feature-Envy-specific, or general?** Predominantly
Feature Envy (+0.091), but not exclusively — Long Method (+0.023) and
God Class (+0.020) both improve too, despite `dominant_external_count`
not being part of either label rule. This is consistent with the
TRAIN/VAL finding in `docs/experiment_fe_dominant_report.md` that the
feature change also modestly helped the other two tasks, plausibly via
shared GAT message-passing representations rather than a direct label
mechanism.

**Does this confirm the TRAIN/VAL generalization evidence, or contradict
it?** Confirms it. The TRAIN/VAL candidate review
(`docs/experiment_fe_dominant_candidate_review.md`) found seed 43's
Feature Envy improvement was not sqlalchemy-specific (sphinx improved by
a comparable margin). The held-out TEST result — three repos (django,
pandas, celery) never seen in any TRAIN/VAL analysis — now shows the
same feature change generalizing to entirely new, unrelated codebases.
That is the single strongest piece of evidence produced this whole
investigation that the effect is a real signal improvement, not an
artifact of the VAL repos.

**Is the historical FE headline (0.525) reliable, or was it an
artifact?** It was measured under a parser-inconsistent train/test
setup and is not a reliable estimate of the deployed model's true
Feature Envy generalization. Evaluation A (0.436), measured under a
consistent setup with the same checkpoint, is the more trustworthy
figure for what that checkpoint actually generalizes to. Evaluation B
(0.527) both fixes this consistency problem *and* improves on it
slightly — the candidate's real, parser-consistent Feature Envy TEST
score is on par with what was previously (incorrectly) believed to be
the deployed model's own score.

**Is the magnitude of improvement meaningful, or noise?** A +0.044
macro-F1 gain, positive on all three tasks simultaneously, with the
largest single-task gain (+0.091) landing precisely on the task the
feature was designed for, is a coherent, mechanistically-explained
result rather than noise. It is also consistent in direction (though
not in exact size) with every TRAIN/VAL and VAL-based signal gathered
throughout `docs/feature_envy_generalization_investigation.md` and
`docs/experiment_fe_dominant_report.md`.

**Any sign of overfitting to VAL, now that TEST is visible?** No.
Overfitting would show as a TRAIN/VAL improvement that fails to
replicate, shrinks sharply, or reverses on TEST. Instead the Feature
Envy F1 gain on TEST (+0.091) is within the same range as the paired
VAL gains reported for seed 43 (+0.121 sqlalchemy, +0.123 sphinx in the
candidate review) — smaller in absolute terms as expected for a
never-seen split, but clearly not reversed or vanished.

## 7. Final Status

**IMPROVED — FINAL CANDIDATE OUTPERFORMS CORRECTED BASELINE**

## 8. Deployment Status

Candidate NOT deployed. Deployment decision pending.

## 9. Evaluation Integrity

- Evaluation A ran exactly once. Evaluation B ran exactly once. Neither
  was retried.
- No result was discarded, rerun, or second-guessed for looking
  surprising or unsurprising.
- No TEST-derived information was used to select the candidate seed —
  seed 43 was selected in `docs/experiment_fe_dominant_candidate_review.md`
  from TRAIN/VAL evidence only, before any TEST file for this experiment
  was built.
- No threshold, architecture, hyperparameter, or loss-weight was changed
  at any point in this evaluation.
- No checkpoint file was modified. No production file
  (`ml/graph/graph_builder.py`, `backend/app/inference.py`,
  `configs/label_thresholds.json`) was modified.
- The historical production baseline (0.740 / 0.525 / 0.767 / 0.677) is
  unmodified, unmodified in storage, and reported here for reference
  only — never overwritten, never reinterpreted as equal to Evaluation A.
- No deployment action was taken.
- Stopping here per instruction. No further experiment or evaluation
  will be performed.
