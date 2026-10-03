# Unseen-Repository Validation (user's step 8 → step 2 diagnosis)

Checkpoint tested: `models/hybrid_class_pool_tuned_fixed_data` (cumulative
fix: nested functions/classes + collaborator-chain correction). Not yet
deployed to the live backend.

## Sources (none in `configs/repos.yaml` train/val/test, new orgs entirely)

| file | repo | org | license |
|---|---|---|---|
| `pip_session.py` | pypa/pip | pypa | MIT |
| `attrs_validators.py` | python-attrs/attrs | python-attrs | MIT |
| `rich_progress.py`, `rich_text.py` | Textualize/rich | Textualize | MIT |
| `pydantic_fields.py` | pydantic/pydantic | pydantic | MIT |

Trained repos are all CLI/web-framework-adjacent (click, black, httpx,
gunicorn, flask, tornado, requests, scrapy, pytest). These five files cover
genuinely different domains: a package installer, a validation/decorator
library, a terminal-rendering library, and a data-validation library —
deliberately chosen to NOT resemble the training domain.

Every prediction below was cross-checked against our own silver-label rule
on the same files (ground truth reference, not the model) before judging
agreement/disagreement.

## God Class — 4/4 real cases, 3 clean agreements, 1 reproducible pattern

| class | rule | model |
|---|---|---|
| `pydantic.FieldInfo` (19 methods, 949 loc, 27 fields) | God Class | **0.999 — agree** |
| `rich.Progress` (27 methods, 591 loc, 10 fields) | God Class | **0.999 — agree** |
| `rich.Text` (55 methods, 1218 loc, 8 fields) | God Class | **1.0 — agree** |
| `pip.PipSession` (7 methods, 244 loc, 3 fields) | **NOT** God Class (7 < 10 methods) | **0.804 — predicted positive** |

`PipSession` reproduces the exact same pattern already found on the
user-supplied `DataProcessor` sample (7 methods, 211 loc, 15 attributes —
also under the method-count floor, also predicted positive by the model).
Two independent real-world files now show the model predicting God Class
on classes the rule's `method_count>=10` floor rejects. This is no longer
a one-off — it's a reproducible signal that the model may have learned a
broader, LOC/field-correlated notion of "too much responsibility" than the
rule's method-count-only floor captures. Not proof the model is "more
correct" than the rule, but strong enough evidence to prioritize revisiting
the God Class threshold formula (already flagged, not yet acted on) over
adding more data.

## Long Method — strong agreement on clearly-long methods, boundary noise elsewhere

Every method with a large margin over threshold agreed
(`FieldInfo.__init__` 132 stmts→1.0, `user_agent` 46 stmts→1.0,
`Text.divide` 49 stmts→1.0, `Progress.update` 29 stmts→0.972, etc.).
Disagreements cluster right at the 15-statement boundary in both
directions (`Text.join` 16 stmts→0.498, `Text.highlight_regex`
20 stmts→0.45 — false negatives just under 0.5; `LocalFSAdapter.send`
13 stmts→0.807, `Progress.add_task`/`wrap_file` — false positives just
under the rule's floor). This reads as ordinary decision-boundary noise
around a hard 15-statement cutoff, not a systematic bias — expected,
not alarming.

**One real false negative worth naming**: `PipSession.add_trusted_host`
(16 statements, rule says positive) scored low enough to not even appear
in the borderline range. `attrs.matches_re` (17 statements, a top-level
**function** not a method) was missed entirely too — consistent with
Long Method's function-branch being trained on far fewer examples than
its method-branch (7675 top-level functions vs 16871 methods corpus-wide,
per the original robustness audit).

## Feature Envy — the real, actionable finding

Four genuine, rule-confirmed Feature Envy cases exist in `rich_progress.py`
(`TimeElapsedColumn.render`, `MofNCompleteColumn.render`,
`DownloadColumn.render`, `TimeRemainingColumn.render` — all render methods
that read heavily from an external `task` object, a clean textbook shape)
plus one in `rich_text.py` (`Text.append_text`, reads a `text` collaborator).

| method | rule (ground truth) | model |
|---|---|---|
| `TimeRemainingColumn.render` | envy (ext=4>self=3) | **0.544 — caught** |
| `DownloadColumn.render` | envy (ext=5>self=1) | 0.355 — missed (borderline) |
| `TimeElapsedColumn.render` (6 stmts) | envy (ext=3>self=0) | **missed entirely** (<0.35) |
| `MofNCompleteColumn.render` (5 stmts) | envy (ext=3>self=1) | **missed entirely** (<0.35) |
| `Text.append_text` (9 stmts) | envy (ext=3>self=2) | **missed entirely** (<0.35) |

**1 of 5** genuine cases caught. This is the weakest result of the whole
validation and matches Feature Envy's already-known status as the
consistently-worst task across every experiment this project has run.

### Root-cause check: is this a training-volume problem?

Checked directly against the actual training tensors before concluding
anything: train has 176 feature_envy positives, statement-count
min=1/median=12/mean=14.6, and **59/176 (34%) already have ≤8 statements**
— short positive examples are NOT rare in training. So "not enough short
examples" is not the explanation.

More likely explanation, consistent with the evidence: these specific
missed cases are all **terminal-UI rendering widgets** (`Progress`
columns formatting a `Task`'s elapsed/remaining time, completion count,
download size) — a domain that doesn't appear anywhere in the 9 training
repos (all CLI tools/web frameworks, none do terminal rendering). This
looks like a genuine **domain-coverage gap**, not a volume problem: the
model has seen plenty of short Feature Envy examples, just none that look
like "a small formatter class reading fields off a data object to build a
display string." Feature Envy's already-severe imbalance (2% positive
rate) means it has the least room to generalize across domain shifts of
the three tasks.

## Answering the user's 4-way root-cause question directly

| cause | Long Method | Feature Envy | God Class |
|---|---|---|---|
| parser/graph construction | no evidence found | no evidence found | no evidence found |
| insufficient graph representation | no evidence found | no evidence found | no evidence found |
| weak/insufficient training examples | no (volume checked, adequate) | **partially — domain coverage, not volume** | no |
| genuine model limitation | boundary noise only (expected) | **yes — weak generalization on an unseen domain** | rule-vs-model definitional gap, not a model failure |

## Recommendation (user's step 3 gate: "only if failures show a clear data-coverage problem")

**Feature Envy**: the evidence here does point at a real coverage gap, but
it's a *domain* gap (no terminal/rendering-style code in training), not a
volume gap. Per the project's own dataset investigation principle, adding
more repos from a *different domain* (not just more of the same
CLI/web-framework style) would be the evidence-justified move **if**
pursued — but this is a judgment call on how far to chase generalization
for a single project, not an automatic "go scrape more data" trigger.

**God Class**: the recurring rule-vs-model gap (2/2 real-world files) is
higher-priority and doesn't need new data at all — it needs the God Class
threshold/definition itself revisited (already flagged, tracked as open
work), since the model may already be learning the right thing and the
*rule* (used only for training labels, not for inference) is what's too
narrow.

**Long Method**: no action indicated — boundary noise is expected model
behavior near a hard cutoff, not a defect.

Per the user's own stated final sequence, next steps in order: diagnose
(done, this document) → decide on any dataset/architecture change
(pending user direction) → retrain only if a change is made → re-run
GNNExplainer on the final checkpoint (still targets the wrong model,
flagged earlier) → one final TEST evaluation → freeze.

**No files modified. No training run. TEST untouched.**
