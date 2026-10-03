# Diagnosis: user-supplied `smelly_code.py` (DataProcessor sample)

User-reported live predictions (frozen `models/hybrid_class_pool_tuned`):

- Long Method: `validate_and_process_user_data`, `generate_processing_report`,
  `__init__`, `validate_user_registration`, `save_to_database`
- Feature Envy: `format_phone`, `save_to_database`, `generate_processing_report`
- God Class: `DataProcessor`

Central question: why does `format_phone` (local string manipulation) get
flagged as Feature Envy while `send_welcome_email` (heavy `self.email_service`
delegation) does not?

## Root cause, found by direct AST inspection (not guessed)

`ast_parser.py::visit_Attribute` records a separate `AttrAccessInfo` for
**every** Attribute node whose immediate `.value` is a bare `Name` — this
includes the INNER attribute in a chain. For
`self.database_connection.execute(user_data)`:

- The outer call is correctly recorded once in `fn.calls`
  (`receiver="self.database_connection"`, an external interaction).
- The inner `self.database_connection` (value=`Name('self')`, attr=
  `'database_connection'`) is **also** recorded as a standalone
  `AttrAccessInfo(receiver="self", ...)`, which `metrics.py` counts as a
  **self**-access.

So every single delegated call through an owned collaborator
(`self.<x>.method()`) inflated `self_access_count` by exactly the same
amount as the genuine external interaction it represents. Traced on the
actual sample (`scratch_external_eval` diagnostic, verbatim tool output):

| method | self (before) | ext | dominant | verdict (before) |
|---|---|---|---|---|
| `save_to_database` | 8 | 8 | self.database_connection (5) | **NOT envy** (5 not > 8) |
| `send_welcome_email` | 6 | 6 | self.email_service (6) | **NOT envy** (6 not > 6) |
| `format_phone` | 0 | 0 | none | NOT envy (correctly) |

`is_feature_envy` requires `dominant_external_count > self_access_count` —
this is the single most common real-world Feature Envy shape (a service
class delegating through `self.<collaborator>`), and the bug made it
almost structurally impossible to satisfy.

`format_phone`'s Feature Envy flag is a **separate, unrelated** issue: its
structural features are unambiguously negative (self=0, ext=0 — it only
calls `re.sub`, a module-level function on an imported name, correctly
excluded). The live model flagging it anyway is not explained by this
bug — it matches the CodeBERT-semantic-override pattern already found and
documented in `docs/phase4_6_external_manual_eval.md` (`Shell.parseCmd`,
same shape: confident false positive despite zero structural signal).
**Not fixed by this change** — remains a documented model/architecture
limitation (CodeBERT dilution), not a metrics bug.

## Fix

`ml/preprocessing/ast_parser.py`: `AttrAccessInfo` gains `is_chained_base`,
set when an Attribute node is itself the `.value` of an outer Attribute
(i.e. a stepping stone, not a standalone access).
`ml/preprocessing/metrics.py`: `function_metrics` excludes
`is_chained_base` entries from both self and external counts (the real
interaction is already captured once via `fn.calls`).

Verified directly on the sample after the fix:

| method | self (after) | ext | dominant | verdict (after) |
|---|---|---|---|---|
| `save_to_database` | 0 | 8 | self.database_connection (5) | **Feature Envy** |
| `send_welcome_email` | 0 | 6 | self.email_service (6) | **Feature Envy** |
| `format_phone` | 0 | 0 | none | not envy (unchanged) |
| `__init__` (15 genuine `self.x = y` assignments) | 15 | 0 | — | unaffected |

Regression tests added: `tests/test_metrics.py::
test_delegating_through_owned_collaborator_is_not_counted_as_self_access`,
`::test_genuine_self_attribute_read_still_counts_as_self_access`. Full
suite: 90/90 passing.

## God Class note (separate observation, not fixed here)

`DataProcessor`: 7 methods, 211 LOC, 15 attributes. The current
`is_god_class` rule (`method_count>=10 AND loc>=192`) does **not** use
`field_count` at all, despite `class_metrics` already computing it and the
project's own God Class definition (`docs/robustness_audit_phase_a_c.md`)
listing attribute count as a relevant signal. Under the current rule this
class does **not** cross the method_count threshold (7 < 10) — the silver
rule itself would say NOT God Class, even though the live model (per the
user's report) predicted positive. This is flagged, not changed, in this
pass: changing the God Class threshold formula is a separate, larger
decision (would shift the label distribution and require its own
before/after comparison) and is out of scope for this specific,
precisely-diagnosed Feature Envy fix. Documented here so it isn't lost.

## Status

Structural train/val graphs rebuilt (TEST untouched, verified before and
after). CodeBERT re-augmentation running in the background. Retrain
(same architecture/hyperparameters as the current best config, isolating
this one data fix as the single variable) to follow, same A/B/C
triangulation methodology as `docs/post_fix_retrain_comparison.md`.
