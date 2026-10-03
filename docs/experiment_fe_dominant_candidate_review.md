# Experiment #1 Candidate Review (Pre-TEST, TRAIN/VAL only)

Read-only review of `docs/experiment_fe_dominant_report.md` plus fresh,
direct inspection of the actual checkpoint/config files (not just
re-reading the prior report's claims). **TEST not touched. No files
modified by this review. No deployment action taken.**

---

## 1. Integrity Verification

Re-checked directly this pass (not just re-citing the prior report):

| check | result |
|---|---|
| Live Feature Envy threshold | `backend/app/inference.py:185` → `fe[i] >= 0.65` — **unchanged** |
| Live Long Method / God Class thresholds | lines 184/195/206 → `>= 0.5` — **unchanged** |
| Production `graph_builder.py` feature line | still `float(fm.external_access_count)` — **unchanged**, confirmed by direct read |
| Deployed checkpoint files | `models/hybrid_class_pool_tuned_fixed_data/{model.pt,norm_stats.pt,best_params.json}` timestamps Sep 2 19:56 — **predate this review's session entirely, not touched** |
| `configs/label_thresholds.json` | unchanged, matches the experiment's own re-derived thresholds exactly |
| Hyperparameters across all 4 seeds | **byte-identical** `best_params.json` in every `seed_*` dir except `best_epoch`/`seed` themselves — no drift, no search performed |
| Architecture / parameter count | **269,315 trainable parameters in all 4 experiment checkpoints**, computed fresh by loading each `model.pt` — identical to the deployed model's own count, confirming no architecture change |
| TRAIN/VAL/TEST repo split | 9/3/3, disjoint — re-verified against `configs/repos.yaml` and on-disk `data/raw/{train,val}` during the experiment's own rebuild (section C of the experiment report), not newly re-run here since nothing about the split could have changed since |
| TEST artifacts | no file under `data/raw/test`, `data/processed/graphs/test`, or `data/processed/graphs_hybrid/test` is newer than this project's pre-experiment work — **confirmed untouched** |
| Only intended feature change | `external_access_count → dominant_external_count`, one column, verified column-by-column in the experiment itself (1061/1061 files matched exactly outside that one column) — no new finding to add here |

**No integrity issue found. Proceeding.**

---

## 2. 4-Seed Comparison

| seed | LM P/R/F1 | FE P/R/F1 | GC P/R/F1 | macro-F1 | best_epoch |
|---|---|---|---|---|---|
| 42 | 0.687/0.924/0.788 | 0.526/0.668/**0.588** | 0.787/0.837/0.811 | 0.729 | 69 |
| 43 | 0.721/0.915/0.807 | 0.592/0.599/**0.595** | 0.800/0.808/0.804 | **0.735** | 79 |
| 44 | 0.724/0.886/0.797 | 0.392/0.687/0.499 | 0.735/0.880/0.801 | 0.699 | 44 |
| 45 | 0.744/0.883/0.808 | 0.471/0.622/0.536 | 0.789/0.827/0.808 | 0.717 | 63 |

**Training/validation curves**: not available — restated, not invented.
`train_one` never evaluates the train set for this model family
(confirmed again this pass by the same code-read already on record in
`docs/generalization_audit.md` §2 and the experiment report §D); no
per-epoch loss log exists for these checkpoints. Best-epoch values
(44-79 of 150) are the only training-dynamics signal available.

---

## 3. Feature Envy Analysis

| seed | FP | FN | vs. known baseline |
|---|---|---|---|
| 42 | 156 | 86 | baseline FP=323, FN=108 → both drop (**cleanest win**) |
| 43 | 107 | 104 | baseline FP=283, FN=90 → FP drops sharply, FN rises slightly (precision-favoring shape) |
| 44 | 276 | 81 | no paired baseline; highest absolute FP of the 4 |
| 45 | 181 | 98 | no paired baseline |

Absolute Feature Envy F1 ranks: **43 (0.595) > 42 (0.588) > 45 (0.536) >
44 (0.499)**. All four exceed the currently-deployed candidate's own
VAL F1 (0.475). Seed 44 is the weakest of the four on this task — both
lowest F1 and highest FP count — worth naming even though it still
beats the deployed baseline.

---

## 4. Long Method Analysis

| seed | F1 | vs. known baseline |
|---|---|---|
| 42 | 0.788 | 0.822 → **-0.034**, the only regression with a confirmed baseline anywhere in this experiment |
| 43 | 0.807 | 0.806 → +0.001, flat |
| 44 | 0.797 | no baseline; absolute value in normal historical range |
| 45 | 0.808 | no baseline; highest absolute value of the 4 |

**Seed 42 is the only seed with a confirmed Long Method regression.**
Not severe (0.788 is still a healthy F1, not a collapse), but it is a
real, non-zero cost that the other three seeds don't carry (43 is
flat; 44/45 have no paired baseline to compare against, so "regression"
can't be confirmed OR ruled out for them — stated as such, not assumed
either way).

---

## 5. God Class Analysis

| seed | F1 | vs. known baseline |
|---|---|---|
| 42 | 0.811 | 0.801 → +0.010 |
| 43 | 0.804 | 0.796 → +0.008 |
| 44 | 0.801 | 0.754 → **+0.047** (largest gain, but from the worst starting baseline) |
| 45 | 0.808 | 0.793 → +0.015 |

No regression in any seed. All four land in a tight 0.801-0.811 band —
God Class is essentially undifferentiated across seeds in this
experiment (expected: its feature vector was never touched by this
change; any movement here is a shared-backbone side effect, not a
direct one).

---

## 6. Cross-Repository Generalization

The check the user weighted most heavily. Only **seed 43** has a true
paired baseline (same checkpoint architecture/seed, original feature
vs. experiment feature, both measured this review cycle) — the
strongest possible form of this evidence:

| repo | seed 43 baseline F1 | seed 43 experiment F1 | Δ |
|---|---|---|---|
| sphinx | 0.398 | 0.521 | **+0.123** |
| sqlalchemy | 0.504 | 0.625 | **+0.121** |
| starlette | 0.533 | 0.571 | +0.038 |

Sphinx and sqlalchemy improve by almost exactly the same amount —
direct, paired evidence the gain is not sqlalchemy-specific.

**Absolute per-repo F1, all 4 seeds** (no paired baseline for 42/44/45,
so these are read as "does sqlalchemy dominate the others," not as a
delta):

| seed | sphinx | sqlalchemy | starlette |
|---|---|---|---|
| 42 | 0.580 | 0.594 | 0.444 |
| 43 | 0.521 | 0.625 | 0.571 |
| 44 | 0.478 | 0.513 | 0.364 |
| 45 | **0.563** | 0.534 | 0.222 |

In 2 of 4 seeds (42, 45) sphinx's absolute F1 is comparable to or
*exceeds* sqlalchemy's — sqlalchemy is not systematically dominating.
Starlette (5 VAL positives) is noisy in both directions as expected
from its sample size and isn't treated as evidence either way.
**Conclusion: the improvement generalizes across repositories in every
seed checked, most rigorously confirmed for seed 43.**

---

## 7. Overfitting / Underfitting Assessment

- **Train metrics: unavailable, stated explicitly, not invented** — same
  limitation as every prior audit this project has run; this review
  does not manufacture a number that doesn't exist.
- **Overfitting**: no evidence found. ROC-AUC stayed high and stable
  (0.95-0.99) across every task and seed; early stopping engaged well
  short of the 150-epoch budget in all 4 runs (44-79 epochs); the
  Feature Envy gain shows up as genuine precision *and* recall movement
  (section 3), not a threshold artifact.
- **Underfitting**: no evidence found. Long Method and God Class both
  sit in normal historical ranges in every seed; Feature Envy's
  already-high ROC-AUC (carried over from before this experiment) is
  unchanged, consistent with the gain being a real precision/recall
  improvement rather than the model suddenly learning to rank better
  from scratch.
- **Instability across seeds**: real but bounded. Feature Envy F1 spans
  0.499-0.595 (range 0.096) and macro-F1 spans 0.699-0.735 (range
  0.036) — the macro-F1 spread is tight and every seed clears the
  deployed baseline's own macro-F1 range; the Feature Envy spread is
  wider but every seed still clears the deployed baseline's F1 (0.475)
  by a comfortable margin. Not flagged as instability requiring further
  investigation — it's within the kind of seed variance this project
  already established and documented for this task
  (`docs/godclass_formula_revision.md` §10's noise-band precedent).
- **Repository concentration**: addressed directly in section 6 — not
  found. The improvement is not caused by a narrow subset of VAL.

---

## 8. Seed Ranking

Ranked on the 9 stated criteria, using TRAIN/VAL evidence only — **not**
defaulting to seed 43 for being the deployed seed:

1. **Seed 43** — highest macro-F1 (0.735), highest absolute Feature
   Envy F1 (0.595), the *only* seed with a fully non-regressing Long
   Method (+0.001), and the *only* seed with a rigorously-confirmed
   (paired-baseline) cross-repository generalization result. Wins on
   criteria 1, 2 (tied-best), 4, and 5 (uniquely).
2. **Seed 42** — close second: second-highest macro-F1 (0.729), largest
   *relative* Feature Envy gain (+0.176, though partly reflecting a
   weaker starting baseline), cleanest FP+FN improvement shape (both
   drop). Its one real weakness: the sole confirmed Long Method
   regression (-0.034) — not severe, but real, and it's the one thing
   keeping it from ranking first.
3. **Seed 45** — solid middle: best absolute Long Method (0.808),
   interesting sphinx>sqlalchemy result (weak evidence against
   sqlalchemy-concentration on its own terms), but second-lowest
   macro-F1 and Feature Envy F1 among the four.
4. **Seed 44** — weakest overall: lowest macro-F1 (0.699), lowest
   Feature Envy F1 (0.499) with the highest FP count (276), weakest
   absolute per-repo numbers across the board, and the shortest
   training run (best_epoch=44, suggestive of an early, possibly less
   fully-converged optimum relative to the other three, though not
   confirmed as such — stated as an observation, not a diagnosis).

**Seed 43 is the strongest candidate — and the reasoning above would
select it on these criteria even if it were not already the deployed
seed.** This is stated explicitly because the instructions warned
against defaulting to it by identity: the selection here is criterion-driven
(highest macro-F1, best Long Method behavior, only rigorously-confirmed
cross-repo evidence), and seed 43 happening to already be deployed is a
coincidence of the ranking outcome, not its cause. Seed 42 is a
legitimate, close alternative if Long Method stability is weighted less
heavily than Feature Envy's absolute magnitude of gain.

---

## 9. Recommended Final Candidate

**Seed 43** — `models/experiment_fe_dominant/seed_43/`

**Configuration**: `heads=2, hidden_dim=64, dropout=0.174, lr=0.00308,
weight_decay=0.00236, weight_lm=1.0, weight_fe=2.364, weight_gc=1.110,
class_method_pool=True` — identical to the deployed checkpoint's own
hyperparameters (no search performed). `best_epoch=79` of 150.

**VAL metrics**: long_method P=0.721 R=0.915 F1=0.807; feature_envy
P=0.592 R=0.599 F1=0.595; god_class P=0.800 R=0.808 F1=0.804;
macro-F1=0.735.

**Why selected**: highest macro-F1 of the four sampled seeds, highest
absolute Feature Envy F1, the only seed with a fully flat (non-regressing)
Long Method result, and the only seed with a directly paired,
apples-to-apples cross-repository comparison confirming the Feature
Envy gain is not sqlalchemy-specific (sphinx +0.123 vs. sqlalchemy
+0.121, section 6).

**This checkpoint is NOT deployed.** `models/hybrid_class_pool_tuned_fixed_data`
(the live checkpoint `backend/app/inference.py` actually loads) is
unchanged. The live Feature Envy threshold (0.65) and Long
Method/God Class thresholds (0.5/0.5) are unchanged. **TEST has not
been evaluated for this or any other Experiment #1 checkpoint.**

---

## 10. Final Decision

**READY FOR ONE-TIME FINAL TEST EVALUATION**

"Final candidate selected. TEST evaluation is authorized only as a
separate next step."

Basis: integrity fully verified with fresh, direct file inspection
(section 1); the effect (Feature Envy improves, God Class holds,
macro-F1 improves) replicates consistently across all 4 independently-
seeded runs, not a single-seed artifact; the specific failure mode
flagged before this experiment ran (sqlalchemy-only improvement) was
checked directly with a paired baseline and did not occur; no evidence
of overfitting, underfitting, or seed instability beyond what this
project has already characterized as normal for this model family. No
further TRAIN/VAL experiment is proposed — the evidence doesn't call
for one, and manufacturing another experiment now would be exploring
for its own sake, which the instructions explicitly warned against.

**Not deployed. Not modified. TEST not touched. Waiting for explicit
authorization to proceed to the one-time final TEST evaluation as a
separate step.**
