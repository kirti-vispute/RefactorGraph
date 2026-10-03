# Feature Envy Decision-Threshold Analysis (VAL only)

Read-only analysis. No files modified except this report and the two
scratch scripts that produced it
(`scratch_external_eval/feature_envy_threshold_sweep.py`,
`scratch_external_eval/feature_envy_threshold_seed42_infer.py`). No
retraining. No TEST evaluation. No deployment change.

**Data used**: two full VAL prediction sets, both already-trained
checkpoints, no new training:
- **seed=43** (deployed candidate): `models/hybrid_class_pool_tuned_fixed_data/errors_val.json`
  — an existing artifact from `scripts/error_analysis_candidate.py`
  (despite its filename, it holds every VAL node's prediction, not just
  errors — verified: n=12966, positives=259, matching
  `docs/error_analysis_report_candidate.md` exactly).
- **seed=42** (prior candidate, backed up): a **pure forward pass**
  (no gradient updates, no optimizer, model loaded with
  `.load_state_dict()` + `.eval()`) on the already-trained backup
  checkpoint `models/hybrid_class_pool_tuned_fixed_data_step3_godclass_seed42/`,
  run once for this analysis to get a second seed's full VAL predictions
  for a stability check. This is inference on an existing checkpoint,
  not a retrain — no new weights were produced or saved.

---

## 1-2. F1/precision/recall across thresholds

**seed=43 (deployed), coarse sweep:**

| t | P | R | F1 | fp |
|---|---|---|---|---|
| 0.05 | 0.154 | 0.869 | 0.262 | 1234 |
| 0.20 | 0.260 | 0.757 | 0.387 | 558 |
| 0.35 | 0.327 | 0.703 | 0.446 | 375 |
| **0.50 (current)** | **0.374** | **0.653** | **0.475** | **283** |
| 0.65 | 0.430 | 0.583 | 0.495 | 200 |
| 0.70 | 0.448 | 0.560 | 0.497 | 179 |
| 0.75 | 0.470 | 0.537 | 0.501 | 157 |
| 0.80 | 0.502 | 0.490 | 0.496 | 126 |
| 0.95 | 0.656 | 0.324 | 0.434 | 44 |

Fine sweep around the coarse peak: F1 stays in a **broad plateau of
0.494–0.505 across t=0.65–0.80** (seed=43) — not a sharp, narrow spike.
Exact fine-grained peak: **t=0.730, F1=0.505** (P=0.466, R=0.552).

**seed=42 (prior candidate), coarse sweep, same shape, lower absolute
level:**

| t | P | R | F1 | fp |
|---|---|---|---|---|
| 0.05 | 0.163 | 0.842 | 0.273 | 1122 |
| 0.35 | 0.284 | 0.672 | 0.400 | 438 |
| **0.50 (current)** | **0.319** | **0.583** | **0.412** | **323** |
| 0.60 | 0.349 | 0.533 | 0.422 | 257 |
| 0.65 | 0.355 | 0.514 | 0.420 | 242 |
| 0.75 | 0.385 | 0.448 | 0.414 | 185 |
| 0.80 | 0.414 | 0.421 | 0.418 | 154 |

seed=42 peaks at **t=0.60, F1=0.422**, a much flatter, lower-ceiling
curve than seed=43's.

## 3. VAL-optimal threshold

Per-seed optimum differs: **seed=43 → t≈0.73 (F1=0.505)**, **seed=42 →
t≈0.60 (F1=0.422)**. Both directions agree (higher than 0.5 helps both),
but the exact peak location is not identical between the two seeds
available — see section 5.

## 4. Comparison against current 0.5

| | seed=43 @0.5 | seed=43 @0.73 (own peak) | Δ |
|---|---|---|---|
| Precision | 0.374 | 0.466 | +0.092 |
| Recall | 0.653 | 0.552 | -0.101 |
| F1 | 0.475 | 0.505 | **+0.030 (+6.3% relative)** |
| False positives | 283 | 164 | **-119 (-42%)** |

A real, positive, but modest F1 gain, bought with a real recall cost
(fewer true smells surfaced) in exchange for meaningfully fewer false
alarms.

## 5. Stability across existing seeds

Only 2 seeds have full VAL prediction sets available (seed=42, seed=43
— the other two sampled seeds from the earlier generalization audit,
44/45, had their model weights discarded after that sweep and were
never re-inferred; **evidence here is 2 seeds, not the full 4** —
stated plainly, not treated as a large sample).

- **Direction is consistent**: both seeds show F1 improving as the
  threshold rises from 0.5 toward roughly 0.6–0.75, driven the same way
  in both (precision rises faster than recall falls in that range).
- **Magnitude is not consistent**: seed=43's gain (+0.030, +6.3%
  relative) is roughly 3x seed=42's own-optimal gain (+0.010, +2.4%
  relative: 0.412→0.422). Applying seed=43's exact threshold (0.73) to
  seed=42's predictions gives F1=0.419 (still an improvement over
  seed=42's own 0.5-baseline of 0.412, just a smaller one than either
  seed's own peak).
- **Exact optimal point is not tightly pinned down**: 0.73 vs 0.60, a
  0.13 gap between the two seeds' own peaks.

**Conclusion for this check**: the *direction* of the finding (0.5 is
suboptimal, something higher helps) is stable across both available
seeds. The *exact* optimal value is not — recommending seed=43's own
argmax (0.73) would be tuning to the one seed that happens to already be
deployed, which risks the same kind of single-example over-fitting this
analysis was explicitly told to avoid, just at the seed level instead of
the example level. See section 8/recommendation.

## 6. False-positive risk

**No unacceptable increase — the opposite effect.** Every threshold
above 0.5 in both seeds' sweeps *reduces* false positives (fewer, not
more, false alarms), because raising the threshold makes the classifier
more conservative. At seed=43's own optimum (0.73): fp 283→164 (-42%).
At the more conservative t=0.65: fp 283→200 (-29%). There is no
threshold-increase scenario in this data that raises false positives —
the real tradeoff is recall (fewer true positives caught), not
precision/false-alarm risk.

## 7. Long Method / God Class unaffected

**Architecturally guaranteed, not just empirically checked.** Verified
by reading `ml/models/gat_baseline.py`: `predict_feature_envy`,
`predict_long_method`, and `predict_god_class` are three independent
`nn.Linear` heads reading from the shared backbone embedding — changing
which threshold is applied to `feature_envy`'s sigmoid output in
downstream code (e.g. `backend/app/inference.py`'s
`fe[i] >= 0.5` comparison) touches only that one comparison. No model
weights, no shared computation, and no other task's threshold is
involved. This analysis itself never re-ran or altered long_method/
god_class predictions in any way (the seed=42 inference pass computed
all three tasks internally, as `collect_records` always does, but only
the feature_envy subset was read or used here — the other two subsets
from that pass were not inspected, saved, or compared).

## 8. Over-optimizing VAL?

Two pieces of evidence pull in different directions:

- **Against over-optimizing**: seed=43's own curve is a broad plateau
  (0.494–0.505 across t=0.65–0.80, a 0.15-wide band), not an isolated
  spike — a genuinely noise-driven optimum would be far more sensitive
  to small threshold changes than what's observed here.
- **For caution**: the cross-seed comparison (section 5) shows real
  disagreement on the *exact* peak (0.60 vs 0.73) with only 2 seeds
  sampled — not enough to say the precise optimal value is
  well-established, only that "meaningfully above 0.5" is a
  consistent direction.

**Recommended reading**: the *direction* of this finding is
well-supported and safe to act on; the *exact* optimal value from
either seed's own argmax is not, and picking seed=43's own peak
specifically would be circular (tuning the deployed model's threshold
to that same model's own VAL idiosyncrasies). A threshold from the
middle of the region where **both** seeds show consistent improvement
(roughly 0.60–0.75) is more defensible than either seed's individual
argmax.

---

## Recommendation

**B. TUNE FEATURE ENVY THRESHOLD**

- **Recommended threshold: 0.65** — not seed=43's own argmax (0.73),
  deliberately: 0.65 sits inside the region where both available seeds
  show consistent, real improvement over 0.5, rather than being tuned to
  the one seed that happens to be currently deployed.
- **VAL F1 at 0.5 (seed=43, deployed model)**: 0.475
- **VAL F1 at 0.65 (seed=43, deployed model)**: 0.495 (+0.020, +4.2% relative)
- **Precision/recall at 0.5**: P=0.374, R=0.653
- **Precision/recall at 0.65**: P=0.430, R=0.583
- **Evidence across seeds**: direction consistent across both available
  seeds (42, 43); exact optimum differs (0.60 vs 0.73); only 2 seeds
  sampled, not the full 4 from the original generalization audit — this
  is real but thin evidence, reflected in the conservative (not
  seed-optimal) recommended value.
- **Risks**:
  1. Recall drops (0.653→0.583, -10.7% relative) — roughly 18 more real
     Feature Envy cases per ~12,966 VAL nodes would be missed at 0.65
     than at 0.5 (fn 90→108). This is a real cost, not just a number —
     users would see fewer true positives flagged.
  2. Only 2-seed evidence for the exact value; a 3rd/4th seed could
     shift the defensible range further.
  3. All labels are silver (rule-derived), not human-annotated — a
     threshold tuned against silver-label F1 optimizes agreement with
     the rule, not necessarily against human judgment of what "is"
     Feature Envy.
  4. This is a VAL-only measurement — per the frozen-TEST discipline,
     this cannot be confirmed against TEST without spending a second,
     not-permitted TEST touch. The gain should be treated as
     VAL-validated only, not TEST-confirmed, unless/until a future,
     separately-authorized final check is done.
  5. Purely a post-hoc inference-time change (one comparison threshold
     in downstream code) — no retraining, no model file change required
     to act on this. Fully reversible by reverting the threshold
     constant.

**Not deployed. Not modified. Left for explicit user decision**, per
this analysis's scope.
