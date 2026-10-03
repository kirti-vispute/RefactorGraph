# Generalization Investigation: Can Feature Envy Improve Without Overfitting?

Analysis only. No files modified except this report and the read-only
scratch script that produced its fresh numbers
(`scratch_external_eval/feature_envy_data_investigation.py` — TRAIN/VAL
only, never reads `data/raw/test` or any TEST-derived artifact,
confirmed by the script itself only iterating `RAW_DIR/"train"` and
`RAW_DIR/"val"`). No retraining. No model, checkpoint, threshold, or
architecture file touched. TEST not evaluated, not inspected, not used
for any decision below.

---

## 1. Current Performance Analysis

**VAL** (candidate checkpoint, seed=43, `docs/error_analysis_report_candidate.md`):

| task | precision | recall | F1 |
|---|---|---|---|
| long_method | 0.727 | 0.904 | 0.806 |
| feature_envy | 0.374 | 0.653 | 0.475 |
| god_class | 0.776 | 0.817 | 0.796 |

**TEST** (closed, restated only, not re-measured): long_method 0.740,
feature_envy 0.525, god_class 0.767, macro-F1 0.677.

**Train performance: unavailable, stated explicitly rather than
estimated.** Verified again by reading `scripts/tune_hybrid_optuna.py::train_one`
(the training loop behind every `hybrid_class_pool_*` checkpoint,
including the deployed one): each epoch trains on `train_loader` then
calls `run_eval(model, val_loader, device)` for early-stopping — there
is no corresponding evaluation call on `train_loader` anywhere in this
function or its callers. No training-set F1/precision/recall has ever
been computed for this model family. This was already flagged in
`docs/generalization_audit.md` section 2 and remains true.

**Class distributions** (fresh TRAIN/VAL-only count, this investigation,
`scratch_external_eval/feature_envy_data_investigation.py` — counts
*eligible* methods only, i.e. excluding `__init__`/`__new__`/
`staticmethod`/`classmethod`, which `is_feature_envy` always rejects):

- TRAIN: 5219 eligible methods, 176 positive (**3.37%**).
- VAL: 11514 eligible methods, 259 positive (**2.25%**).

(These differ slightly from the raw `dataset_report_train_val_rebuild.md`
counts — 175/258 — because that report counts ALL methods including
the always-ineligible ones in the denominator; both are correct, just
different denominators. Directionally identical.)

**FP/FN patterns** (`docs/error_analysis_report_candidate.md`, VAL):
of 283 false positives, 151 (53%) have `dominant_external_count <=
self_access_count` — i.e. the model flags them despite failing the
label rule's own direction entirely, not a boundary artifact. Of 90
false negatives, 58 sit exactly at the rule's floor
(`dominant_external_count==3`).

**Overfitting / underfitting assessment for Feature Envy specifically**:
TEST F1 (0.525) is *higher* than VAL F1 (0.475) for this task — the
opposite of what overfitting would produce. TEST ROC-AUC=0.952,
PR-AUC=0.546 (`docs/final_test_evaluation_report.md`) — high ranking
quality alongside modest F1@0.5 argues against pure underfitting too
(a genuinely underfit model ranks poorly, not just mis-thresholds).
Classified in `docs/generalization_audit.md` as **LOW-MEDIUM
underfitting, LOW overfitting** — this investigation found no evidence
to revise that.

**Data / representation / loss / threshold / architecture — which is
it?** The evidence below (sections 2, 4, 5, 6) supports: **primarily a
representation/feature-engineering issue, secondarily a data
domain-coverage issue, NOT primarily a loss-weighting issue (already
aggressively applied), and threshold was already the correctly-identified,
already-exploited lever** (deployed threshold=0.65, TEST-confirmed gain
0.412→0.525 was largely already captured that way). Architecture has a
real but harder-to-fix limitation (section 6) that is a *ceiling*, not
the most cost-effective next lever.

---

## 2. Feature Envy Graph Representation — the central finding

Read `ml/graph/graph_builder.py` in full (every edge-creation line, not
a summary) to answer this precisely rather than by inference.

**What the graph DOES capture**:
- `(method, uses, attribute)` — created whenever `a.receiver == "self"`
  (`graph_builder.py` line ~285). This fires for both a genuine
  self-serving read (`self.name`) AND the *inner* half of a chained
  collaborator delegation (`self.database_connection` inside
  `self.database_connection.execute(...)`, since that inner Attribute
  node's own receiver is literally `"self"` — confirmed via
  `ast_parser.py`'s `AttrAccessInfo` construction). So a method DOES get
  an edge to the attribute node representing its collaborator reference.
- `(method, accesses, parameter)` — created when the receiver is one of
  the method's own parameters (line ~289) — the one case of genuinely
  cross-object access the schema links to a real node.
- `(method, calls, method/function)` — only when the callee resolves to
  something in the *same file* (module docstring, line 39-40); a call
  through `self.<collaborator>.method()` does NOT resolve here (the
  callee is an attribute of an unknown type, not a same-file
  method/function name) — confirmed by reading `_resolve_call_target`
  (lines 251-270): it only matches `receiver == "self"` (exact) or
  `receiver is None`, never a dotted receiver like
  `"self.database_connection"`. **This exact call edge is silently
  dropped**, consistent with the module docstring's own stated
  intent: "accesses on an unresolvable receiver ... are not linked to a
  node."

**The load-bearing gap**: every attribute node in the entire graph has
an **identical, constant feature vector**: `data["attribute"].x =
torch.ones((n_attrs, 1))` (line 210). There is no name, no type, no
signal distinguishing "database_connection" (a heavyweight collaborator
object a method might be feature-enviously delegating to) from "id" or
"name" (a trivial scalar field). **No edge type anywhere in the 14-relation
schema connects a method to an *external class* or to another class's
methods/attributes across a class boundary** — this is deliberate
(the module docstring: "fabricating a node for an unknown external
class would be exactly the 'artificial relationship' the project spec
says to avoid") and not a bug, but it means the GAT's message passing
cannot distinguish, at the level of graph topology, a self-serving
attribute use from a collaborator-delegating one — **both produce the
literal same edge shape**, `(method, uses, attribute)`. The only signal
that tells them apart is a *scalar already sitting in the method's own
feature vector* (see below) — something the classifier has direct
access to without needing any message passing at all. Two layers of
GATConv can in principle learn an indirect proxy (e.g. "this attribute
is touched by many different classes' methods elsewhere in the file" —
weak and same-file-only), but nothing in the schema encodes "this
method depends heavily on another class" directly.

**A second, more concrete and more fixable gap — a real label/feature
mismatch, found and quantified in this investigation**: `is_feature_envy`
(`ml/preprocessing/label_rules.py`) keys its decision on
`dominant_external_count` (the single largest-external-receiver count) —
*not* the sum across all external receivers. But
`graph_builder.py` line 183 feeds the method node's feature vector with
`float(fm.external_access_count)` — the **total**, summed across every
distinct external receiver, not the dominant one. These are only
identical when a method has exactly one external collaborator. Measured
fresh this investigation (`scratch_external_eval/feature_envy_data_investigation.py`,
TRAIN+VAL only):

| split | positives where total == dominant | positives where they differ | mean total (when differing) | mean dominant (when differing) |
|---|---|---|---|---|
| TRAIN | 100/176 (56.8%) | 76/176 (43.2%) | 9.55 | 6.39 |
| VAL | 145/259 (56.0%) | 114/259 (44.0%) | 8.53 | 5.75 |

**For ~44% of all positive examples in both TRAIN and VAL, the number
the model actually sees in its input feature vector overstates, by
roughly 33-49% on average, the exact quantity the label itself is
defined by.** This is not noise in the label (the label is computed
correctly from `dominant_external_count`) — it is a real, quantifiable
mismatch between what the classifier is fed and what it is being asked
to predict, for a large minority of the positive class. This is a
concrete, low-risk, purely feature-engineering fix candidate — no
architecture change, no new edge type, no retraining of the graph
construction pipeline's logic beyond which precomputed scalar goes into
one column of an existing 5-column vector.

**Answering the question directly**: information needed to detect
Feature Envy is only **partially** available to the GAT — the two raw
scalars it needs most (self vs. dominant-external count) are available,
but one of them is the *wrong* aggregate for ~44% of positives, and the
graph topology itself provides no additional, non-redundant relational
signal beyond what's already in those scalars.

---

## 3. Label Quality

**Rule** (`ml/preprocessing/label_rules.py::is_feature_envy`, unchanged
this session): excludes `__init__`/`__new__`, `staticmethod`, and
`classmethod` (documented rationale: these have a contract that
legitimately involves reading external state). Requires
`dominant_external_count >= 3` (a floor, not a percentile — the only
Feature Envy threshold that isn't corpus-relative) **and**
`dominant_external_count > self_access_count`.

**Noisy?** Two real label-computation bugs were found and fixed earlier
this project (call+attribute double-counting; the collaborator-chain
double-counting — `docs/dataprocessor_sample_diagnosis.md`). Both are
already fixed and reflected in the current TRAIN/VAL data. No further
label-computation bug was found in this investigation's re-reading of
`is_feature_envy` itself.

**Positive diversity**: real and quantified, not assumed. Statement-count
spread for positives is wide (TRAIN min=1, median=12.0, mean=14.6,
max=111 — this investigation's fresh count) — short and long
Feature Envy examples both exist in volume. The previously-documented
domain-coverage gap stands: `docs/phase8_unseen_repo_validation.md`
found the model misses Feature Envy specifically on
terminal-rendering-widget code, a style absent from all 9 TRAIN repos
(all CLI/web-framework-adjacent). This investigation adds a second,
narrower diversity finding: **per-repo positive rate in TRAIN itself
spans a 12x range** (flask 0.38% — 1 single positive example in the
entire repo — to scrapy 4.62%, see section 4). Flask contributes
essentially nothing to what the model learns about Feature Envy despite
contributing 261 real methods.

**Confusing negatives**: quantified fresh this investigation. TRAIN has
174/5043 negatives (3.5%) and VAL has 197/11255 (1.75%) that are
"near-miss" — real external interaction at or above the floor, but
`self_access_count >= dominant_external_count`, i.e. structurally one
step from being positive. This is a smaller population than the 151/283
VAL false positives that violate the rule's direction entirely
(section 1) — meaning the model's false positives are not *only*
explained by these genuinely-hard near-miss negatives; a meaningful
share of FPs are confident mistakes on negatives that aren't even close
to the boundary (consistent with the previously-documented
CodeBERT-semantic-override pattern, `docs/phase4_6_external_manual_eval.md`).

**Synthetic vs. repository examples**: not applicable — no synthetic
Feature Envy training data exists anywhere in this pipeline (confirmed:
every training label comes from real, cloned open-source repositories
via `scripts/build_dataset.py`; the only "manual" Python files in this
project, `scratch_external_eval/manual/*.py` and the user-supplied
DataProcessor sample, were used exclusively for qualitative
validation/audit, never for training).

---

## 4. Data Distribution (TRAIN + VAL only)

Fresh counts this investigation
(`scratch_external_eval/feature_envy_data_investigation.py`):

**Per-repo positive rate, TRAIN** (9 repos):

| repo | eligible methods | positives | rate |
|---|---|---|---|
| black | 263 | 14 | 5.32% |
| click | 331 | 13 | 3.93% |
| flask | 261 | 1 | **0.38%** |
| gunicorn | 727 | 31 | 4.26% |
| httpx | 310 | 12 | 3.87% |
| pytest | 1193 | 28 | 2.35% |
| requests | 152 | 7 | 4.61% |
| scrapy | 1126 | 52 | **4.62%** |
| tornado | 856 | 18 | 2.10% |

**Per-repo positive rate, VAL** (3 repos):

| repo | eligible methods | positives | rate |
|---|---|---|---|
| sphinx | 3668 | 66 | 1.80% |
| sqlalchemy | 7462 | 188 | **2.52%** |
| starlette | 384 | 5 | 1.30% |

**A real, previously-unquantified concentration risk**: sqlalchemy
alone contributes **188/259 = 72.6% of every Feature Envy positive in
all of VAL**. Every VAL-based decision this project has made about
Feature Envy — the seed=43 selection (`docs/godclass_formula_revision.md`
§10, chosen on macro-F1, which itself is substantially driven by
feature_envy performance on sqlalchemy-shaped code), and the threshold
calibration (`docs/feature_envy_threshold_analysis.md`, `_final_audit.md`) —
is, for this task, disproportionately a referendum on sqlalchemy's own
coding style, not a broad multi-repo measure. This does not invalidate
those decisions (TEST, a genuinely different 3-repo set, confirmed the
threshold gain independently), but it is a real, previously-undocumented
concentration fact worth stating plainly.

**Method-size distribution**: TRAIN positives median=12.0 statements
(mean 14.6), VAL positives median=9.0 (mean 13.3) — VAL's positives skew
slightly shorter than TRAIN's, but not dramatically; no evidence of a
severe TRAIN/VAL size-distribution mismatch.

**Does VAL contain patterns missing from TRAIN?** Partially answerable
from TRAIN/VAL alone: VAL's dominant repo (sqlalchemy, an ORM/SQL
compiler codebase) is a different *style* of code than any single TRAIN
repo, though TRAIN as a *pool* (click/black/httpx/gunicorn/flask/tornado/
requests/scrapy/pytest) is also stylistically heterogeneous — this
investigation cannot cleanly separate "VAL is a novel domain" from "VAL
just happens to be dominated by one repo with an unusually clear
Feature-Envy-rich style (dialect compilers delegating to `sql.*`,
`dialect.*`, etc. — visible directly in the hardest-FP table in
`docs/error_analysis_report_candidate.md`, e.g. `dominant_external_receiver=sql.case`,
`ischema.sys_types`)." The already-completed genuinely-unseen-org
validation (`docs/phase8_unseen_repo_validation.md`: pip/pydantic/rich/attrs)
remains the cleaner test of true domain generalization, and its
conclusion (terminal-rendering-widget code is the specific gap, not
sqlalchemy-style delegation) still stands independently of this
finding.

**Would additional diverse training data plausibly help?** Plausibly
yes, but narrowly: the evidence points at *domain* diversity (more
non-CLI/web-framework repos, ideally something structurally close to
the already-identified gap — GUI/rendering/formatting-heavy code) rather
than *volume* (already established in `docs/phase8_unseen_repo_validation.md`:
34% of positives already have ≤8 statements, ruling out "not enough
short examples"). This is the same conclusion the project already
reached; this investigation did not find new evidence to strengthen or
weaken it, only to reconfirm it with the added repo-concentration
context above.

---

## 5. Loss / Class Imbalance

**Already implemented, verified by reading the code, not assumed**:
`scripts/train_hybrid_baseline.py::pos_weight_for` computes a standard
`neg/pos` ratio and every task's loss is
`nn.BCEWithLogitsLoss(pos_weight=pw_*)` (`scripts/tune_hybrid_optuna.py`
lines 58-69). With TRAIN's ~8479 methods and 175 feature_envy positives,
`pos_weight_fe ≈ (8479-175)/175 ≈ 47.4` — a substantial, automatic,
data-driven upweighting already in force, on top of which
`weight_fe=2.364` (`models/hybrid_class_pool_tuned_fixed_data/best_params.json`,
an Optuna-searched *per-task* multiplier on top of the BCE pos_weight,
found via hyperparameter search, not hand-picked) further upweights
Feature Envy's contribution to the combined multi-task loss relative to
`weight_lm=1.0`. **Class weighting is not an untried lever — it is
already aggressively applied and was already searched over.**

**Focal loss**: not implemented anywhere in this codebase (confirmed,
no `focal` reference in `scripts/train_hybrid_baseline.py` or
`scripts/tune_hybrid_optuna.py`). Conceptually, focal loss down-weights
*easy* examples and concentrates gradient on *hard* ones, which is a
different lever than class-frequency weighting (already applied). Given
the specific FP pattern found in section 1 (53% of VAL FPs violate the
rule's own direction — i.e. the model is confidently wrong on cases a
frequency-based reweighting cannot fix, since they aren't rare, they're
simply mis-learned), focal loss is a *plausible, not yet tried* lever —
but it changes training dynamics globally, is harder to reason about
in a multi-task setup (it would need to be scoped to the feature_envy
head only, which this codebase's training loop does not currently
support per-task loss function *types*, only per-task loss function
*weights*), and its effect is empirically uncertain without a controlled
run. Ranked as a real but second-tier candidate compared to the
feature/label mismatch fix in section 2.

**Balanced sampling** (oversampling positives / undersampling
negatives at the batch level): not implemented (training iterates whole
graphs per batch, `scripts/train_hybrid_baseline.py`'s `DataLoader`
batches *files*, not individual method nodes — so node-level balanced
sampling isn't straightforwardly compatible with this pipeline's
batching unit without a more invasive change). Not recommended as a
first move given this structural mismatch with the existing training
loop.

**Conclusion for this section**: further loss-weighting adjustment has
low expected marginal value (already aggressively tuned); focal loss is
a real but architecturally-secondary, higher-effort option; the
feature/label mismatch (section 2) is a cheaper, more directly-targeted
fix that doesn't touch the loss at all.

---

## 6. Architecture (Design B)

Read `ml/models/gat_baseline.py::HeteroGAT` in full again for this
section specifically.

`predict_feature_envy(h_dict) = self.head_feature_envy(h_dict["method"])`
— a single linear head reading **only** the method's own final-layer
embedding. Unlike `predict_god_class`, there is **no analogous pooling
mechanism** for feature_envy (Design B's `class_method_pool` concatenates
a class's own embedding with a mean-pool of its methods specifically to
solve god_class's own information-flow gap, documented in
`ml/models/gat_baseline.py`'s Phase 13 comment — no equivalent exists
for feature_envy reading anything about the *external* collaborator
class).

The method's embedding, after 2 `HeteroConv`/`GATConv` layers, can
aggregate from its 1-hop neighbors that the deployed model's `EDGE_TYPES`
schema actually wires (confirmed: `(method,uses,attribute)` and
`(method,accesses,parameter)` are both in the base 14-relation list the
deployed checkpoint was built with) and, at the second layer, from
2-hop neighbors-of-neighbors (e.g. other methods that also touch the
same attribute node). But since every attribute/parameter node carries
an **identical placeholder feature** (section 2), what actually
propagates through this path is purely *structural* (which nodes share
which neighbors) — there is nothing semantic to propagate about what
the external collaborator *is*. **The architecture does not prevent
cross-class information from reaching the feature_envy head, but the
graph gives it nothing distinctive to pass along even where the edges
exist.** This is consistent with, and gives a precise mechanism for,
this project's own prior finding
(`docs/robustness_audit_phase_a_c.md` A8 and elsewhere) that CodeBERT's
semantic embedding — not the graph-structural half — appears to be
doing most of the real work on this task, since it's the only signal
carrying anything about *what* is being accessed (the literal source
text of `self.database_connection.execute(...)`).

**Is this an information-flow limitation specifically affecting Feature
Envy?** Yes, in the sense that Feature Envy is *definitionally* a
cross-class relational concept and this schema has no cross-class edge
type by design (section 2) — but this is a genuine, real architectural
ceiling, not a bug, and closing it properly (e.g. resolving `self.<attr>`
to a synthetic "collaborator" pseudo-type, or giving attribute nodes
real identity features) is a bigger, `category 6/7`-risk change per the
ranking below, not a quick fix.

---

## 7. Generalization Risk Ranking

Ranked safest → riskiest, as requested, with the reasoning tied to
evidence gathered above:

1. **Data-quality improvements** (safest) — e.g. targeted domain
   expansion (section 4) or fixing the flask-style near-zero-positive
   repo dilution. Risk: low — more/better data essentially can't cause
   the model to memorize a specific held-out example, and TRAIN/VAL/TEST
   separation is unaffected. Cost: highest (finding and vetting new
   repos, re-running the full label/graph pipeline) relative to the
   other cheap options below.
2. **Feature/representation improvements** (safe, and specifically:
   the `dominant_external_count` vs `external_access_count` fix from
   section 2 is close to the safest possible change in this whole
   ranking — it changes one scalar in an existing 5-column feature
   vector, touches no architecture, no loss, no threshold, and is
   directly evidence-justified by a measured, quantified mismatch
   affecting ~44% of positives). Risk: low — a single corrected input
   feature cannot by itself cause the model to overfit; if it doesn't
   help, it's trivially revertible.
3. **Loss adjustment** — already extensively applied (section 5);
   further adjustment here has a narrower plausible upside and (for
   something like focal loss) real risk of destabilizing the *other*
   two tasks' training dynamics in a shared multi-task loss, exactly
   the failure mode section 9's stopping criteria warn about.
4. **Threshold calibration** — already executed and TEST-confirmed
   (0.412→0.525) before this investigation began; not a remaining lever
   for further gains without a real underlying change, though it
   remains the safest possible category in principle (post-hoc,
   zero-retraining, per-task-isolated, as already demonstrated).
5. **Regularization** (dropout/weight decay changes) — moderate risk:
   changes training dynamics for all three tasks simultaneously via the
   shared backbone; could trade one task's performance for another's
   without a clear, targeted hypothesis for why Feature Envy specifically
   would benefit.
6. **Architecture changes** (e.g. a feature-envy-specific
   cross-class pooling mechanism, analogous to Design B's god_class
   pooling) — higher risk: this project's own history
   (`docs/robustness_audit_phase_a_c.md`, the Phase 13b "negative
   transfer" regression referenced when Design B's god_class pooling
   was first added) already shows that architecture changes on this
   shared backbone can regress *other* tasks even when targeted at one.
   Any such change needs its own controlled before/after comparison on
   all three tasks, not just feature_envy.
7. **Increased model capacity** (riskiest, least justified) — no
   evidence anywhere in this investigation or the prior generalization
   audit suggests the model is capacity-constrained (269K trainable
   parameters, frozen CodeBERT doing most of the semantic heavy
   lifting, `docs/generalization_audit.md` section 8) — Long Method and
   God Class both perform well with this same capacity, so a capacity
   increase aimed at Feature Envy specifically has no mechanism-level
   justification and the highest risk of overfitting the (still small,
   ~2-3% positive-rate) Feature Envy signal specifically.

---

## 8. Experiment Plan (if pursued — not started)

**Baseline**: the currently-deployed seed=43 candidate
(`models/hybrid_class_pool_tuned_fixed_data`), VAL feature_envy F1=0.475,
macro-F1=0.693 (this seed's own VAL number, `docs/godclass_formula_revision.md` §10).

**Change #1 (the one this investigation's evidence most directly
supports — see section 10)**: in `graph_builder.py`'s method/function
feature vector, replace (or augment alongside) `fm.external_access_count`
with `fm.dominant_external_count` — the quantity `is_feature_envy`
actually keys its decision on. This is a single-column data-pipeline
change, not an architecture change.

- **One change at a time**: only this feature substitution — no
  threshold change, no loss change, no architecture change in the same
  run, so any VAL delta can be attributed to this one variable.
- **Fixed random seeds**: reuse the same multi-seed protocol already
  established this project (`scratch_external_eval/godclass_seed_sweep.py`'s
  pattern — seeds 42/43/44/45, monkeypatching `tho.SEED` since
  `train_one` hardcodes its own module-level seed, a real gotcha this
  project already hit and documented in
  `docs/godclass_formula_revision.md` §10).
- **Same architecture/hyperparameters** as the current best_params.json
  (isolates the data/feature change as the only variable, per this
  project's established A/B/C methodology,
  `docs/post_fix_retrain_comparison.md`).
- **Evaluation metrics**: VAL precision/recall/F1 for all three tasks
  (not feature_envy alone), plus VAL macro-F1, across all sampled seeds.
- **Acceptance criteria**: feature_envy VAL F1 improves by a margin
  clearly outside the seed-noise band already characterized for this
  model family (god_class's own 4-seed spread was 0.047,
  `docs/godclass_formula_revision.md` §10 — an analogous or freshly
  re-measured feature_envy seed-spread should be the bar, not a single
  seed's number), **and** long_method/god_class VAL F1 do not regress
  outside their own respective noise bands.
- **Rollback criteria**: if feature_envy improves only by regressing
  long_method or god_class beyond noise, or if the improvement does not
  hold across the seed sweep (i.e. it's a single-seed artifact), the
  change is reverted and not promoted — exactly per this task's explicit
  instruction not to optimize Feature Envy at the other tasks' expense.
- **TEST**: not touched at any point in this experiment. Only after a
  VAL-based decision to promote a new candidate would the existing,
  established one-time final TEST protocol
  (`scripts/phase16_final_test_eval.py`) be considered — a separate,
  later, explicitly-authorized step, not part of this experiment.

No second experiment is proposed here per the task's instruction to
provide only the first one if retraining is recommended.

---

## 9. Stopping Criteria

- No VAL macro-F1 improvement outside the already-characterized 4-seed
  noise band (section 8's acceptance criteria).
- Feature Envy improves only by trading away long_method or god_class
  VAL performance beyond their own noise bands.
- The improvement doesn't replicate across the seed sweep (single-seed
  artifact — this project already caught exactly this failure mode once
  this cycle, section 10 of `docs/godclass_formula_revision.md`, and
  should apply the same discipline here).
- Any sign that a change is fitting VAL specifically rather than
  generalizing — e.g. an improvement that only shows up on
  sqlalchemy-heavy VAL batches specifically (checkable via the same
  per-repo error breakdown technique used in
  `docs/error_analysis_report_candidate.md`), given the 72.6%
  sqlalchemy-concentration finding in section 4.
- The next proposed change would require a new architecture component
  (cross-class pooling, a new edge type) — per the ranking in section 7,
  this is a materially bigger step that should be its own separately-
  scoped decision, not a quick follow-up to this experiment.

---

## 10. Final Recommendation

**B. RETRAIN WITH DATA/LOSS IMPROVEMENT — evidence supports a controlled
experiment.**

Specifically a **feature-representation fix**, not a loss-weighting
change (section 5 shows loss weighting is already aggressively applied)
and not yet an architecture change (section 6 shows a real ceiling
exists, but section 7 ranks architecture changes as meaningfully
riskier than feature fixes, and this investigation found a cheaper,
more directly-justified lever first).

**The exact first experiment** (and only the first — per instruction,
no second experiment is proposed):

> In `ml/graph/graph_builder.py`, change the method/function node
> feature vector to use `fm.dominant_external_count` in place of (or
> alongside) `fm.external_access_count`, so the classifier's input
> matches the quantity `is_feature_envy`'s label rule actually depends
> on. Rebuild TRAIN+VAL structural+hybrid graphs from this one change
> (TEST untouched). Retrain with the exact current best_params.json
> hyperparameters, across the same 4-seed sweep already used for the
> God Class investigation. Compare VAL feature_envy/long_method/god_class
> F1 against the current seed=43 baseline, seed-by-seed. Promote only if
> feature_envy improves outside the seed-noise band without regressing
> the other two tasks outside theirs.

This is not started. No file has been modified toward it. It is
presented as the evidence-justified first move, for explicit user
authorization before any code change, retrain, or TEST access.
