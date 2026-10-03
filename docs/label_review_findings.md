# Silver Label Manual Review — Findings

Per the project's data-quality rules (no fabricated/unvalidated labels), the
rule-based Long Method / God Class / Feature Envy labels produced by
`scripts/build_dataset.py` were manually reviewed against a stratified
sample (`scripts/sample_for_review.py`, 15 positive + 15 negative per smell,
train split only) before being treated as usable for training. This
document records what was found and fixed.

## Method

Five review rounds were run. Each round: generate a fresh stratified sample,
read every item's source against its predicted label, categorize
true/false positives, fix the root cause in `ml/preprocessing/metrics.py` /
`ast_parser.py` if a systematic error was found, add a regression test
locking in the fix, rebuild the full dataset, and re-sample.

## Bugs found and fixed

1. **Module-namespace calls counted as envy** — `click.echo(...)` inside a
   method was flagged as the method "envying" a class named `click`.
   Fixed by excluding imported names (`ModuleInfo.imported_names()`) from
   the Feature Envy receiver set.
2. **Local variables and loop variables counted as external objects** —
   `result = {}；result.update(...)` or `for rule in self.rules: rule.x`
   flagged the locally-created/loop-bound name as a "collaborator" object.
   This was the single largest source of false positives. Fixed by tracking
   every name bound by assignment, augmented assignment, `for`/`async for`,
   `with`/`async with`, comprehension targets, and the walrus operator
   within a function, and excluding them from the external-receiver count.
3. **Long Method inflated by docstrings** — raw `end_lineno - lineno` line
   span counted multi-line docstrings as if they were logic, flagging
   heavily-documented short functions (e.g. a 3-line function with a
   30-line docstring). Fixed by switching the Long Method decision from raw
   LOC to a recursive executable-statement count (`_count_statements`,
   ast.stmt nodes only, not descending into nested def/class bodies), which
   collapses a whole docstring to the single `Expr` statement it actually is.
4. **Own-class-name references counted as envy** — `Subprocess.initialize()`
   inside a method of class `Subprocess` (referring to its own class-level
   state by literal name instead of `self`/`cls`) was counted as external.
   Fixed by excluding the enclosing class's own name from the receiver set.
5. **Generic container methods dominating the signal** — `.get()`,
   `.setdefault()`, `.update()`, etc. called on a `**kwargs` dict or any
   plain parameter were weak/misleading evidence (normal data-structure use,
   not domain-specific behavioral coupling). Fixed by excluding a denylist
   of generic dict/list/str methods from the external-call count — applied
   to BOTH the call-visitor path and the attribute-access-visitor path,
   since `x.get(...)` fires both `visit_Call` and `visit_Attribute` and an
   earlier fix only covered the former (silently leaking through the
   latter until caught in round 4).
6. **`except ... as e:` exception variables counted as envy** —
   `e.args[0]` on a caught exception was flagged as accessing an external
   object. Fixed by tracking `ast.ExceptHandler.name` as a local binding.
7. **Alternate-constructor classmethods flagged** — `from_crawler(cls, crawler)`-style
   factory classmethods exist specifically to read an external object and
   build `cls(...)` from it; that is their contract, not misplaced logic
   (same reasoning already applied to `__init__`). Excluded `classmethod`-
   decorated methods from Feature Envy eligibility.

All fixes have regression tests in `tests/test_label_rules.py` and
`tests/test_ast_parser.py` (23 tests total, all passing).

## Result

- Feature Envy train-split positive count dropped from 450 → 163 across the
  five rounds as false positives were removed (label counts before/after
  are in the git history of `docs/dataset_report.md`; final counts are in
  that file's current version).
- Final round-5 sample review: God/Large Class — 8/8 reviewed positives were
  genuine multi-responsibility classes, no false positives found. Long
  Method — 5/5 spot-checked positives genuine after the statement-count fix.
  Feature Envy — of 15 positives reviewed in the final round, ~9 were clear
  true positives, ~3 were ambiguous-but-plausible (framework/Context-object
  coupling with real complexity), and 2 were residual false positives from
  a still-unfixed pattern (see Known residual limitation below). Negative
  buckets (predicted-False) showed no obvious false negatives across all
  three smells in the samples reviewed.

## Known residual limitation (not fixed — accepted, documented)

Methods implementing the **Visitor / Strategy / Observer / delegate-callback
pattern** (`visit_Expr(self, node)`, `find_handler(self, target, request)`,
`toterminal(self, tw)`) remain a source of false positives: their entire
job, by design, is to drive another object's behavior, which is
structurally identical to Feature Envy's operational definition even though
it is not the code smell — moving such a method to the "other" class would
break the pattern rather than fix a design problem. This is a well-known,
explicitly acknowledged limitation of syntactic Feature Envy detection
(Fowler notes Visitor-style code as a deliberate exception to the smell).
Fixing it in general requires either (a) a curated exception list of
common visitor/callback naming conventions, which is brittle and
corpus-specific, or (b) actual type inference to distinguish "a
domain object whose responsibility is being misplaced" from "a delegate
object this method exists to drive" — out of scope for a rule-based
labeler. This is reported rather than silently accepted: Feature Envy
labels should be treated as noisier than Long Method / God Class labels,
and any model trained on them should be evaluated with this caveat in mind
(consistent with the project's "document limitations, don't hide
weaknesses" requirement).
