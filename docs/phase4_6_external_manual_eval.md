# Phase 4-6: External + Manual Test Evaluation, Error Analysis

Checkpoint evaluated: `models/hybrid_class_pool_tuned_fixed_data` (Phase 3
retrain, corrected graph representation). This checkpoint is NOT yet
serving the live backend (`backend/app/inference.py` still points at the
frozen `models/hybrid_class_pool_tuned`) -- switching is a deliberate final
decision, not done here. TEST split never read by this evaluation.

Script: `scripts/eval_manual_and_external_examples.py` (read-only, no
training, no writes to any dataset/model file).

## Phase 5 — Manual targeted test cases (`scratch_external_eval/manual/`)

Each case's expected label was independently verified against our own
silver-label rule *before* running the model (see prior turn) -- so this
tests generalization, not whether the model can match a label it never saw.

| case | expected | model result |
|---|---|---|
| A: `OrderReportBuilder.build_report` (32 stmts) | Long Method | **0.999 — correct** |
| B: `ApplicationManager` (35 methods, 198 LOC) | God Class | **1.0 — correct** |
| B: `ApplicationManager.deactivate_user`/`.activate_user` (2 stmts each) | clean (not long) | **0.662 / 0.698 — false positive** |
| C: `DashboardWarningLight.evaluate` (dom_cnt=5 > self=4) | Feature Envy | **0.363 — false negative** (below 0.5 threshold) |
| D: clean class/function set | clean | **correct — nothing predicted positive** |

## Phase 4 — External repo (`ZikaZaki/code-smells-python`, MIT, verified earlier)

| case | silver-rule verdict | model result |
|---|---|---|
| `Company.pay_employee` | Feature Envy (verified genuine TP by reading source) | **0.59 — correct** |
| `VehicleRegistry.online_status` (before AND after) | Feature Envy (silver rule's own **known false positive** — enum-member access, not envy) | **not flagged (<0.2) — model correctly avoided the rule's own bug** |
| `Shell.parseCmd` | NOT Feature Envy (self=0, ext=0 — all calls are denylisted generic string methods `.split/.strip/.upper`) | **0.909 — confident false positive** |
| everything else | no Long Method/God Class examples exist in this corpus under our thresholds | correctly quiet |

## Error analysis — three real findings, causes investigated, not forced to match labels

**1. God-Class methods getting spurious Long Method predictions (false positive).**
`ApplicationManager.deactivate_user`/`.activate_user` are 2-statement
methods -- not remotely long by any definition, and not flagged by our own
rule. Both belong to `ApplicationManager`, a 35-method class correctly
flagged God Class at 1.0 confidence. Investigated whether this is a
message-passing leak: `belongs_to` (method→class) is confirmed NOT wired
into this model's `EDGE_TYPES` (only used via the separate
`class_method_pool` post-hoc mean-pool, which only feeds the **god_class**
head, not method predictions) -- so this isn't a direct architectural leak
through the graph. More likely explanation, consistent with the
already-documented Phase 13 multi-task negative-transfer finding: the
shared 2-layer encoder's representation for methods inside an
unusually-uniform, repetitive God-Class-shaped class may be getting pulled
toward "this class's methods are structurally risky" by gradient sharing
across tasks during training, even though each method's own raw features
don't support it. Not fully proven, reported as a hypothesis grounded in
prior evidence, not asserted as fact.

**2. A genuine Feature Envy case scored below threshold (false negative).**
`DashboardWarningLight.evaluate` is textbook Feature Envy (dominant
external count 5 > self access 4) and scored 0.363 -- correctly directionally
positive relative to a `self`-heavy method, but under the 0.5 cut. Consistent
with Feature Envy's already-documented weakest-of-three-tasks status
(severe ~2% positive-rate imbalance, `docs/robustness_audit_phase_a.md` /
`docs/robustness_audit_phase_a_c.md`) -- one more data point, not a new
root cause.

**3. CodeBERT semantic signal overriding a clean structural signal (false positive), a genuinely new finding.**
`Shell.parseCmd` has `self_access_count=0`, `external_access_count=0` (every
call in the method is `.split`/`.strip`/`.upper` -- all on the project's own
generic-container-method denylist, confirmed by direct inspection) yet the
model scores it 0.909. Since the *structural* features fed to the model are
essentially featureless-negative here, the confident positive prediction is
most plausibly coming from the 768-dim CodeBERT embedding finding this
chained-string-call-heavy code *semantically* similar to genuine
call-heavy Feature Envy code, even though the structural rule (which the
model was trained to approximate) explicitly excludes this exact pattern.
This is a concrete, real-world instance of the dilution hypothesis already
recorded in `docs/hybrid_baseline_report.md` (CodeBERT's 768 dims
dominating a much smaller structural block through the same encoder) --
previously only inferred from aggregate val metrics, now observed directly
on one example.

## Root-cause categorization (Phase 6 requirement)

| finding | insufficient examples? | label quality? | parser/graph? | feature extraction? | model capacity/architecture? | threshold/calibration? |
|---|---|---|---|---|---|---|
| God-Class method false positives | unlikely | no | no | no | **plausible (multi-task interference)** | possible (borderline ~0.66-0.70) |
| Feature Envy false negative | **yes (imbalance)** | partly (known noisy task) | no | no | no | possible (0.363 is close-ish) |
| CodeBERT semantic override | no | no | no | no | **yes (dilution, already documented)** | no |

None of the three findings are explained by "not enough training files."
Two are re-confirmations of already-documented architecture trade-offs
(multi-task interference, CodeBERT dilution); one is the expected
consequence of Feature Envy's severe, already-quantified imbalance. Per
the master prompt's own principle, this does **not** justify Phase 7
dataset expansion on its own evidence -- expanding data volume would not
address a shared-encoder interference pattern or a semantic-vs-structural
conflict already characterized in prior phases. This is reported as the
evidence-based Phase 7 recommendation: **hold off on dataset expansion**;
if these three findings are to be addressed, the load-bearing next steps
are architectural (task-specific capacity, decoupling CodeBERT's
contribution) or calibration work (Phase O), not more raw files.

**No files were modified by this evaluation. TEST untouched.**
