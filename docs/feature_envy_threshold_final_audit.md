# Feature Envy Threshold=0.65 — Final Pre-Deployment Audit

Confirmatory re-check of `docs/feature_envy_threshold_analysis.md`
before actually changing the deployed threshold. Read-only: no files
modified by this audit, no retraining, no TEST evaluation. Same two
existing VAL prediction sets reused (no new inference run):
`models/hybrid_class_pool_tuned_fixed_data/errors_val.json` (seed=43,
deployed) and `scratch_external_eval/feature_envy_val_predictions_seed42.json`
(seed=42, prior candidate — from the already-completed pure-inference
pass in the prior turn, not re-run here).

## 1-2. 0.50 vs 0.65, exact numbers

| seed | threshold | P | R | F1 | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| 43 (deployed) | 0.50 | 0.3739 | 0.6525 | 0.4754 | 169 | 283 | 90 |
| 43 (deployed) | 0.65 | 0.4302 | 0.5830 | **0.4951** | 151 | 200 | 108 |
| 42 (prior) | 0.50 | 0.3186 | 0.5830 | 0.4120 | 151 | 323 | 108 |
| 42 (prior) | 0.65 | 0.3547 | 0.5135 | **0.4196** | 133 | 242 | 126 |

Both seeds improve at 0.65: seed=43 +0.0197 F1 (+4.1% relative), seed=42
+0.0076 F1 (+1.8% relative). False positives drop in both (283→200,
323→242) — no false-positive cost in either seed.

## 3. Every seed for which predictions are available

Only seeds 42 and 43 have full VAL prediction sets on disk (seeds 44/45
from the original 4-seed generalization-audit sweep had their weights
discarded and were never re-materialized — stated plainly again, this
audit did not create new evidence beyond what already existed from the
prior turn). Both available seeds show the same result: **positive F1
gain, false-positive reduction** at 0.65. No seed shows a regression at
0.65 relative to 0.50.

## 4. Is 0.65 a narrow VAL optimum or a stable region?

Re-examining both seeds' full curves (`docs/feature_envy_threshold_analysis.md`
section 1-2) specifically around 0.65:

- **seed=43**: F1 at 0.60/0.65/0.70/0.75 = 0.487/0.495/0.497/0.501 — flat
  within 0.014 across that whole range. 0.65 is solidly inside this
  plateau, not near an edge.
- **seed=42**: F1 at 0.55/0.60/0.65/0.70 = 0.421/0.422/0.420/0.417 — also
  flat, peak at 0.60 (F1=0.422); 0.65 is within 0.002 of that seed's own
  peak.

**0.65 lands inside a stable, multi-seed-consistent plateau for both
seeds — for seed=42 specifically, it is within 0.002 F1 of that seed's
own individual optimum.** This is stronger than "a defensible compromise
between two different optima" (how the prior analysis framed it) — on
this closer look, 0.65 is close to simultaneously near-optimal for both,
not a midpoint between two far-apart peaks. Confirmed, not merely
asserted: not a narrow, single-point VAL spike.

## 5-6. Does the change affect only Feature Envy? Are Long Method / God Class thresholds/predictions unchanged?

**Confirmed by reading the actual code** (`backend/app/inference.py`,
lines 184, 185, 195, 206 — the only 4 places any task's `predicted`
boolean is computed anywhere in the backend):

```
184:  results["long_method"].append({**common, "probability": lm[i], "predicted": lm[i] >= 0.5})
185:  results["feature_envy"].append({**common, "probability": fe[i], "predicted": fe[i] >= 0.5})
195:  "probability": lm_f[i], "predicted": lm_f[i] >= 0.5,          # long_method (function nodes)
206:  "probability": gc[i], "predicted": gc[i] >= 0.5,              # god_class
```

These are four **independent literal `0.5` comparisons**, not a shared
constant — there is no single `THRESHOLD` variable that all four read
from. Editing line 185 alone (`fe[i] >= 0.5` → `fe[i] >= 0.65`) cannot
mechanically touch lines 184/195/206; they are separate statements
reading separate tensors (`lm`, `lm_f`, `gc` vs `fe`) computed from
three architecturally-independent `nn.Linear` prediction heads
(`ml/models/gat_baseline.py` — `predict_long_method`,
`predict_feature_envy`, `predict_god_class` share only the upstream GAT
backbone embedding, not their output layers). **Confirmed unchanged by
construction**, not just by absence of contrary evidence.

## 7. Exact location of the Feature Envy decision threshold

**`backend/app/inference.py:185`** — the sole place in the entire
stack (backend or frontend) where Feature Envy's raw sigmoid probability
is compared against a threshold to produce the `predicted` boolean the
rest of the application acts on.

**Frontend never independently re-thresholds** — confirmed by reading
`frontend/src/components/DetectionSummary.jsx:14`
(`.filter((p) => p.predicted)`) and `frontend/src/components/CodeGraph.jsx`
(lines 275, 506, 508, all reading `prediction.predicted` directly). The
frontend consumes only the backend's already-thresholded boolean for
detection logic; the separately-displayed "Model Score" percentage
(`SmellDetailPanel.jsx`) is the raw probability and is unaffected either
way — changing the threshold changes which items get a "detected" badge,
not the score number shown.

**One important scope note, not a defect**: `scripts/train_hybrid_baseline.py:134`
(`pred = (proba >= 0.5).astype(int)`) is a **separate, independent**
hardcoded 0.5 used by `run_eval`, the function behind every VAL/TEST
metric this project has ever reported (`docs/final_test_evaluation_report.md`,
`docs/error_analysis_report_candidate.md`, etc.). Changing
`backend/app/inference.py:185` only changes live production serving —
it does **not** change how this project computes or reports its own
historical/future metrics, which stay pinned at 0.5 unless separately
changed. This means published F1 numbers will continue to read "0.475
@0.5" even after a production deployment at 0.65 — expected, not a
inconsistency, but worth stating so it isn't mistaken for one later.

## 8. Not modified

No files were changed by this audit. `backend/app/inference.py` still
reads `fe[i] >= 0.5` at the time of writing this report.

---

## Final Recommendation

**DEPLOY 0.65**

This is a general threshold-calibration decision, not fitting to a
particular example, for the following reasons, each tied to a specific
check above:

- The evaluation set is the **full VAL feature_envy prediction
  population** (n=12,966 nodes, 259 positives) for two independent
  model instances (seeds 42 and 43) — not any single hand-picked file or
  class. No individual example (e.g. the DataProcessor sample or any
  unseen-repo case from earlier audits) was consulted or referenced in
  choosing this value.
- **Both available seeds agree on direction and magnitude is
  reasonable** (section 3): +4.1% relative F1 for the deployed seed,
  +1.8% for the other — no seed shows a regression.
- **0.65 sits inside a broad, flat plateau for both seeds** (section 4),
  within 0.002 F1 of seed=42's own independent optimum — this is the
  opposite of a fragile, noise-driven single point.
- **Zero false-positive cost** in either seed (section 1-2) — the only
  real tradeoff is recall (fewer true positives surfaced: seed=43
  169→151 tp, i.e. 18 fewer of the 259 VAL-labeled Feature Envy cases
  would be flagged), a cost stated plainly, not hidden.
- **The change is architecturally isolated to Feature Envy alone**
  (sections 5-7), confirmed by reading the actual four-line decision
  block in `backend/app/inference.py` — Long Method and God Class use
  separate literal thresholds on separate model output tensors from
  separate prediction heads, and are provably untouched by editing line
  185 only.
- **Fully reversible**: a one-line, one-value change in production
  inference code, no retraining, no checkpoint change, no dataset
  change. Can be reverted by changing `0.65` back to `0.5` with zero
  side effects on anything else in the system.

**Not deployed by this audit** — the actual code edit is a separate,
explicit action left for the user's go-ahead, consistent with this
task's scope (audit only, "do not modify any files").
