# FINAL TEST LABEL AUDIT

Label audit only. **No model inference. No TEST metrics computed. No
checkpoint loaded for prediction.** Every number below comes from
`scripts/test_label_audit.py` (fingerprint-based label diff, re-parsing
TEST source with the current parser) or direct file inspection — nothing
invented.

## 1. Audit Status

**PASS**

## 2. Current Label Rules

`ml/preprocessing/label_rules.py` (unchanged this entire investigation —
never copied, never modified; the experiment only ever copied
`graph_builder.py`'s *feature* construction, never the label logic):

- **Long Method**: `statement_count >= 15`
- **Feature Envy**: excludes `__init__`/`__new__`, `staticmethod`,
  `classmethod`; `dominant_external_count >= 3 AND dominant_external_count
  > self_access_count`
- **God Class**: `(method_count >= 10 AND loc >= 192) OR (loc >= 192 AND
  field_count >= 1 AND method_count >= 3)` — the OR-rule established in
  `docs/godclass_formula_revision.md` this project cycle.

Thresholds read directly from `configs/label_thresholds.json`:
`long_method_statements=15, god_class_method_count=10, god_class_loc=192,
feature_envy_min_external_calls=3, god_class_min_fields=1,
god_class_min_methods_for_field_branch=3`.

## 3. TRAIN/VAL Consistency

**Identical by construction, not just by inspection.** The corrected
TEST labels were produced by literally importing and calling
`ml.preprocessing.label_rules.is_feature_envy` /
`is_long_method` / `is_god_class` — the exact same functions, same
module, same `configs/label_thresholds.json` file already verified
identical to what produced the current TRAIN/VAL labels
(`scripts/experiment_fe_dominant_rebuild_structural.py`'s own integrity
check, prior turn). There is no second copy of `label_rules.py`
anywhere in this project — only `graph_builder.py` (the *feature*
construction) was ever copied for Experiment #1. **No divergence is
possible between the TEST corrected-label pipeline and the TRAIN/VAL
pipeline.**

## 4. Feature Envy Label Changes: 632 → 554

**Fully reconciled — every one of the 78 net changes is accounted for,
not just plausible.**

| transition (matched, pre-existing entities) | count |
|---|---|
| stayed negative | 19,302 |
| negative → positive | 178 |
| positive → negative | **257** |
| stayed positive | 375 |
| new entity (no stale counterpart), positive | 1 |
| new entity, negative | 307 |

Net: `178 - 257 + 1 = -78`. **Matches the reported delta exactly.**
Sanity check built into the script confirms it: stale positives among
matched pairs = 632 (100% of the original total found a match — nothing
lost to alignment noise for this task), corrected = 553 matched + 1 new
= 554.

**Two opposing, both already-established causes, evidenced by
direction**:
- The **collaborator-chain self-access fix**
  (`docs/dataprocessor_sample_diagnosis.md`) can only *decrease*
  `self_access_count` (it removes a specific double-count, never adds
  one) — this pushes toward *more* positives (the 178 negative→positive
  flips are consistent with this).
- The **call+attribute double-count fix** (pre-existing, attributed in
  this project's own history to a peer session, already reflected in
  TRAIN/VAL) can only *decrease* `dominant_external_count`/
  `external_access_count` for call-heavy methods — this pushes toward
  *fewer* positives (the larger 257 positive→negative count is
  consistent with this being the dominant of the two effects on TEST).
- **Caveat, stated plainly**: this script captured *corrected-side*
  metrics for each flipped example (printed above/in the raw log) but
  did not additionally capture the *stale-side* self/dominant-external
  values for the same entities, so the mechanism is confirmed
  *directionally and by net reconciliation*, not individually
  re-verified for all 435 flipped cases. Both underlying fixes are
  independently documented and already reflected in TRAIN/VAL, and no
  third, unexplained mechanism is needed to make the totals balance.

**Category**: collaborator-chain self-access fix (established) +
call/attribute double-count fix (established) + nested-function
representation (established, contributes only 1 new positive). **No
label-rule change. No unexplained residual.**

## 5. Long Method Label Changes: 2301 → 2367

**Single cause, cleanly and completely explained.**

| transition (matched, pre-existing entities) | count |
|---|---|
| stayed negative | 21,745 |
| stayed positive | 2,301 |
| **any flip among pre-existing entities** | **0** |
| new entity, positive | **66** |
| new entity, negative | 1,487 |

**Zero pre-existing method or function changed its Long Method label.**
100% of the +66 comes from brand-new nodes — nested functions that were
previously "silently dropped entirely" (`ml/graph/graph_builder.py`'s
own module docstring) and now get their own node, their own
`statement_count`, and their own label for the first time. Examples are
all `<locals>`-qualified closures (e.g.
`AMQP._create_task_sender.<locals>.send_task_message`,
`build_tracer.<locals>.trace_task`) — exactly the nested-def shape this
fix targets. **Category: nested-function/class representation fix
(established), 100% of the change. No label-rule change. No
unexplained residual.**

## 6. God Class Label Changes: 295 → 328

**Single cause, cleanly and completely explained, with a clean
mechanistic signature.**

| transition (matched, pre-existing entities) | count |
|---|---|
| stayed negative | 3,305 |
| negative → positive | 30 |
| **positive → negative** | **0** |
| stayed positive | 295 |
| new entity, positive | 3 |
| new entity, negative | 133 |

Net: `30 + 3 = 33`. **Matches exactly.** The **zero** positive→negative
transitions is itself strong mechanistic evidence: the God Class formula
revision (`docs/godclass_formula_revision.md`) added an OR-branch (the
field branch) *on top of* the original AND-rule — a class that was
positive under the old rule is, by construction, still positive under
the new OR-rule (the old condition is now just one of two ways to
qualify, never removed). A rule change that could only ever add
positives, never remove them, is exactly what a 0-value
positive→negative cell demonstrates. The 30 flipped examples all show
the field-branch's own signature: `method_count` between 3-9 (**below**
the old rule's method_count>=10 floor, so the *old* rule could never
have flagged them regardless of size) combined with `loc>=192` and
real `field_count` (2-13) — e.g. `EmailBackend` (loc=239, methods=9,
fields=13), `ManagementUtility` (loc=237, methods=5, fields=4).
`class_metrics.method_count`/`loc`/`field_count` do not depend on the
collaborator-chain or call/attribute fixes at all, so this task's label
changes have no plausible second cause. **Category: God Class formula
revision (established, field-branch OR-rule), 100% of the change. No
unrelated label-rule change.**

**Methodological caveat**: 136 stale class entries could not be
fingerprint-matched to a corrected counterpart at all (class-level
fingerprint = `(loc, method_count)`, a coarser, less unique signal than
methods' 3-tuple fingerprint — some small classes plausibly share
values). **None of these 136 are God-Class-positive on the stale side**
(0 positives among them, confirmed by the script), so they cannot be
hiding any part of the +33 discrepancy — the reconciliation above is
exact regardless. Flagged for completeness, not because it affects the
conclusion.

## 7. Root Cause

**Expected consequences of already-established parser/graph fixes —
for all three tasks, with full or near-full quantitative reconciliation.**

- Long Method: **fully explained**, single cause (nested-function
  representation), exact reconciliation.
- God Class: **fully explained**, single cause (God Class formula
  revision), exact reconciliation, plus a structural sanity check (zero
  positive→negative transitions, exactly as an OR-rule addition
  predicts).
- Feature Envy: **fully reconciled quantitatively** (net -78 exactly
  accounted for), explained by two already-established, independently-documented
  fixes acting in opposite directions; the *individual* mechanism for
  each of the 435 flipped cases was not re-verified one-by-one (stated
  as a real limit of this pass, not glossed over).

No label-rule change was found. No unrelated/unknown mechanism is
required to make any of the three totals balance.

## 8. Experiment #1 Independence

**Confirmed.** `corrected_original` and `corrected_dominant` labels are
identical for all 1574 TEST files — verified in the prior turn
(`scripts/rebuild_test_corrected_structural.py`: "identical everywhere
except (at most) column 4: 1574", "UNEXPECTED differences beyond column
4: 0", label counts printed identical: feature_envy=554,
long_method=2367, god_class=328 for both). Independently reconfirmed by
construction here: `is_feature_envy`/`is_long_method`/`is_god_class`
take a `FunctionInfo`/`ClassInfo` and `LabelThresholds` — they never
read a node's feature vector (the `external_access_count` vs.
`dominant_external_count` column) at all, so no code path exists by
which Experiment #1's feature change could touch a label.

## 9. Leakage Check

**No leakage possible, confirmed by code structure, not just
observation.** `ml/preprocessing/label_rules.py`'s only imports are
`ast_parser` and `metrics` (re-verified this pass) — no `torch` import,
no model/checkpoint reference, no prediction data anywhere in the
module. Labels are a pure function of the parsed source code and the
TRAIN-derived threshold constants. No model was loaded in this audit
(no `torch.load` of any `model.pt` anywhere in
`scripts/test_label_audit.py`) and no candidate/deployed checkpoint was
touched.

## 10. Integrity Check

| Check | Result |
|---|---|
| TEST membership unchanged | PASS |
| 1574 TEST files present | PASS |
| Current label rules verified | PASS |
| TRAIN/VAL label logic matches | PASS |
| FE changes explained | PASS (fully reconciled, mechanism directionally evidenced) |
| LM changes explained | PASS (fully reconciled, single cause) |
| GC changes explained | PASS (fully reconciled, single cause) |
| No label leakage | PASS |
| A/B labels identical | PASS |
| Production untouched | PASS (`graph_builder.py`, `backend/app/inference.py` re-read directly, unchanged) |
| Deployed checkpoint untouched | PASS (`models/hybrid_class_pool_tuned_fixed_data/model.pt`, timestamp Sep 2 19:56, predates this session) |
| Candidate checkpoint untouched | PASS (`models/experiment_fe_dominant/seed_43/model.pt`, timestamp Sep 3 09:40, unchanged since selection) |
| Historical TEST artifacts untouched | PASS (`data/processed/graphs/test`, `data/processed/graphs_hybrid/test` only ever read, never written) |
| No TEST inference performed | PASS |
| No TEST metrics computed | PASS |

## 11. Final Gate

**LABEL AUDIT PASSED — READY FOR FINAL TEST EVALUATION**

Stopping here per instruction. No TEST inference has been run. No
model has been loaded for prediction. No metric has been computed. The
historical baseline (Long Method 0.740, Feature Envy 0.525, God Class
0.767, Macro-F1 0.677) is unmodified and not reinterpreted. Waiting for
explicit authorization before any TEST inference.
