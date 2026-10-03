# RefactorGraph Robustness Audit — Phase A/B/C (audit only, no retraining)

Scope: complete pipeline audit (Phase A), Python-representation verification
(Phase B), function-vs-method correctness (Phase C). No code was changed,
no model was retrained, TEST was not touched (confirmed clean at the end —
`data/processed/graphs_hybrid/test` untouched, no TEST file read by any
command run during this audit). Every finding below is backed by either a
direct code read (cited by file/line) or a live, reproducible run against
the actual pipeline code — not speculation.

Final selected model referenced throughout: `models/hybrid_class_pool_tuned`
(Design B, `class_method_pool=True`, `edge_types=None` → the 14-relation
`EDGE_TYPES`, not `EDGE_TYPES_CLASS_AGGREGATION`), per
`docs/final_test_evaluation_report.md`.

---

## 1. Pipeline as it exists today

```
.py source
  -> ml/preprocessing/ast_parser.py   (stdlib ast -> ModuleInfo/ClassInfo/FunctionInfo)
  -> ml/preprocessing/metrics.py      (structural metrics, no thresholds)
  -> ml/preprocessing/label_rules.py  (silver labels from metrics + corpus-percentile thresholds)
  -> ml/graph/graph_builder.py        (HeteroData: 7 node types, up to 14 edge relations)
  -> scripts/augment_graphs_with_codebert.py  (frozen CodeBERT mean-pooled embedding, concatenated onto class/method/function features)
  -> scripts/train_hybrid_baseline.py (per-type z-norm, fit on TRAIN only)
  -> ml/models/gat_baseline.py        (2-layer HeteroGAT + 3 task heads)
  -> backend/app/inference.py         (same pipeline, single in-memory file, FastAPI-served)
  -> frontend                         (React + Cytoscape)
```

13 test files, full suite passing (75 backend + 24 frontend as of the last
full run this session).

---

## 2. Findings, by Phase-A checklist item

### 1–2. Parser / graph-construction limitations (confirmed by direct execution)

**Nested functions and closures are silently dropped from the graph — confirmed, quantified.**

`ast_parser.py::_handle_function` (line 174-199):
```python
is_method = bool(self._class_stack) and not self._func_stack
...
if self._func_stack:
    self._func_stack[-1].nested_functions.append(fn)   # <-- orphaned here
elif is_method:
    self._class_stack[-1].methods.append(fn)
else:
    self.functions.append(fn)
```
A function defined inside another function (a closure/helper) is appended
to `nested_functions` on its parent `FunctionInfo` — a field
`graph_builder.py` never reads. It never becomes a `method` or `function`
graph node, and any call made *to* it is also silently unresolved (its name
never enters `function_index_by_name`).

Verified live (see audit transcript): a `method_with_closure` containing a
nested `def helper(x): ...` produces `helper reachable as graph node? ->
False`, while `mod.functions` and the graph's `function` node count both
omit it entirely — no error, no warning, no node.

**Quantified on the actual train-repo source tree** (`data/raw/train/**/*.py`,
1899 files scanned — note: this is the *unfiltered* tree, before
`build_dataset.py`'s `EXCLUDE_DIR_NAMES` directory filter that produces the
real 547-file trained corpus, so treat this as "how common is the pattern
in real Python," not an exact on-corpus count):

| | count |
|---|---|
| top-level functions (become graph nodes) | 7675 |
| methods (become graph nodes) | 16871 |
| nested/closure functions (become **nothing**) | 3110 |
| files containing ≥1 nested function | 424 / 1899 |

~11% of all function-like definitions in this real Python code are nested
functions, and every one of them is currently invisible to the model —
this is a direct violation of "do not silently discard valid Python
constructs" and the single highest-severity Phase A/B finding.

**Nested classes are flattened, not scoped — confirmed.**

`visit_ClassDef` (line 154-166) pushes/pops `_class_stack` correctly for
method attribution, but unconditionally does `self.classes.append(cls)` on
exit regardless of nesting depth. A class nested inside another class ends
up as a *sibling top-level entry* in `mod.classes`, and `graph_builder.py`
then draws a `(module, contains, class)` edge to it directly — the true
`Outer` → `Inner` containment relationship is not represented at all (there
is no class-contains-class edge type in the schema). Verified live:
`Outer` containing `class Inner: ...` produces `mod.classes == ['Inner',
'Outer']`, both flat. Lower severity than the nested-function bug (Inner
does still become a real, correctly-featured graph node — it's just wired
to the wrong parent), but still an "incorrect relationship" (Phase A item 5).

**The `calls` edge is close to non-functional for the single most common
call pattern in OO code — confirmed, this is the most important structural
finding.**

`graph_builder.py::_resolve_call_target` (line 198-210):
```python
if receiver == "self" and caller.owner_class_idx is not None:
    ...
if receiver is None:
    if method_name in function_index_by_name: ...
    return None
return None   # <-- every other receiver, unconditionally unresolved
```
A call is only ever turned into a `calls` edge when the receiver is
literally `self` (an intra-class call) or the call is a bare name (a
top-level function calling another top-level function by name). **Any call
through a parameter, a local variable, `self.attr.method()`, or any other
receiver produces no edge at all** — it is exactly the receiver pattern
Feature Envy exists to detect (calling methods on a collaborator object
passed in as a parameter) that the graph's `calls` edges cannot see.

Verified live: a class with three calls on a parameter
(`service.validate()/.process()/.finish()`) plus a top-level function
calling a method on its own parameter (`obj.do_something()`) plus a call to
a (dropped) nested function produced **zero** `calls` edges in the built
graph — `[et for et in g.data.edge_types if et[1]=='calls'] == []`.

This is consistent with, and explains, the aggregate numbers already in
`docs/graph_validation_report.md`: every `calls` edge in the entire train/
val/test corpus is either `method→method`/`method→function` (only reachable
via `self.` or a bare name) or `function→function` (bare name only) —
`function→method` never appears in any split's edge-type table, because a
function has no `self` and can only resolve calls to other bare-name
top-level functions.

Practical consequence: the GAT's message passing carries almost no signal
about *which* external object a method/function actually calls into — that
information exists only in the node-level `external_access_count` /
`dominant_external_receiver` scalar features (from `metrics.py`), not as
graph connectivity to the collaborator. This measurably reduces how much
the "graph" part of a graph neural network can contribute over a flat
per-node classifier for exactly the tasks (Feature Envy, and to a lesser
extent God Class) that most depend on knowing *which other object* is
involved — plausibly part of why `docs/hybrid_baseline_report.md` already
found the pure-structural GAT beating the CodeBERT-augmented hybrid on
Feature Envy/God Class, and consistent with the shared-encoder negative-
transfer findings already documented across Phase 13.

**`external_access_count` / `dominant_external_count` double-count call
expressions — confirmed, a genuine measurement bug, not just an
architecture limitation.**

`metrics.py::function_metrics` (line 76-101) increments the same
`external_receivers` Counter from **two separate AST visitor callbacks** for
the same source expression: `visit_Attribute` fires for the `.validate`
part of `service.validate()`, and `visit_Call` fires for the `(...)` part —
both add 1 to `external_receivers["service"]`.

Verified live: three distinct method calls on the same external parameter
(`service.validate()`, `service.process()`, `service.finish()`) produced
`external_access_count = 6`, `dominant_external_count = 6` — not 3.

This metric is both a **node feature** fed directly to the model and the
**quantity the Feature Envy threshold (`>= 3`) is compared against** — and
since `LabelThresholds.from_corpus` derives that "3" from percentiles of
this same inflated metric, the threshold is internally self-consistent for
labeling, but the semantics are not what the code comments/docs claim: a
pure attribute-read pattern (`x.attr` with no call) counts once per access,
while a method-call pattern (`x.method()`) counts twice per call — an
arbitrary 2x asymmetry between "envies via method calls" and "envies via
attribute reads" that was never an intentional design decision. A method
making exactly 2 real external method calls already crosses the labeling
threshold (4 ≥ 3) that was presumably meant to require roughly 3 distinct
interactions. This directly affects label precision and is a concrete,
fixable bug (not requiring more data or a bigger model) — see §7.

### 3–4. Missing / incorrect node & edge types

- No `variable` node type exists, by deliberate design (`graph_builder.py`
  docstring, also confirmed live via the existing frontend test asserting
  `"variable" not in node_types`) — local variables are intentionally out
  of scope, already correctly documented; not a bug, but worth noting the
  originally-proposed schema (Module/Class/Method/Function/**Variable**/
  Attribute/Parameter/Import) was never fully implemented and the frontend
  correctly never claims it is.
- `inherits` edges only connect to a base class defined in the **same
  file** (`graph_builder.py` docstring, confirmed by construction — base-
  name lookup is against `class_index`, built only from `mod.classes` of
  the current file). Inheriting from an external/stdlib/third-party class
  (the overwhelming majority of real inheritance, e.g. `class Foo(BaseModel):`)
  produces zero `inherits` edges. Documented as a known limitation already;
  restated here because it materially limits God Class's ability to see
  inheritance-based responsibility spreading in realistic code.
- `belongs_to` (the `method→class` reverse edge) exists in every built
  graph but is **not** used by the final selected model's backbone
  (`edge_types=None` → plain `EDGE_TYPES`, confirmed against
  `tune_hybrid_class_pool_optuna.py`) — it was found to cause negative
  transfer when wired in (Phase 13b, already documented in
  `gat_baseline.py`'s own comments). Correctly not used; noted here only so
  the frontend's decision to *display* `belongs_to` edges (real, present in
  the served graph) isn't mistaken for something the model's backbone
  reads — it doesn't, `class_method_pool` reads the plain `contains` edge
  directly instead.

### 5. Incorrect relationships

Covered above: nested-class-to-module `contains` edge misattributes the
true parent (§2).

### 6–7. Label / dataset limitations

Already extensively self-documented by a prior, genuinely rigorous manual
gold-review (`docs/label_review_findings.md`, 5 rounds, 7 systematic bugs
found and fixed with regression tests) — re-summarized here rather than
redone, since re-litigating it would not be scientifically honest (the work
already exists and is sound):

- Long Method / God Class: final-round manual spot-check found 0 false
  positives (5/5 and 8/8 clean).
- Feature Envy: final round found ~9/15 (60%) clear true positives, ~3/15
  ambiguous-but-plausible, ~2/15 (13%) residual false positives — the
  **Visitor/Strategy/Observer pattern** (a method whose entire job is to
  drive another object's behavior — `visit_Expr`, `find_handler`, etc.),
  explicitly documented as unfixed by simple heuristics (needs either a
  brittle naming-convention denylist or real type inference). **Feature
  Envy labels are meaningfully noisier than the other two smells** — this
  is already flagged in the existing docs and is reconfirmed here as still
  true; no new work invalidates it.
- **New, not previously documented**: `@property`-decorated methods are
  *not* excluded from Feature Envy eligibility (only `staticmethod`,
  `classmethod`, `__init__`, `__new__` are — `label_rules.py` line 69-78).
  A property getter that legitimately delegates to a wrapped/external
  object (`return self._session.get_current_user()`) is structurally
  identical to genuine Feature Envy under the current rule and is not
  covered by the classmethod-style exception already made for the
  analogous factory-method pattern. Same category of residual limitation
  as the already-documented Visitor pattern; smaller in scope (a quick
  corpus grep would be needed to size it, not done here since Phase A is
  audit-only).

### 8. Class imbalance (from `docs/dataset_report.md`, restated with ratios)

| split | methods | long_method | feature_envy | classes | god_class |
|---|---|---|---|---|---|
| train | 8131 | 904 (11.1%) | 163 (2.0%) | 1436 | 118 (8.2%) |
| val | 13905 | 1160 (8.3%) | 277 (2.0%) | 2575 | 181 (7.0%) |
| test | 23613 | 2301 (9.7%) | 631 (2.7%) | 3718 | 295 (7.9%) |

Feature Envy is the most imbalanced task by a wide margin (~2% positive
rate vs. ~8-11% for the other two) — already mitigated at training time via
per-task `pos_weight` in `BCEWithLogitsLoss` (`train_hybrid_baseline.py`),
not via oversampling (correct choice — oversampling would have risked
duplicating the already-scarce, already-noisier Feature Envy positives
across mini-batches). No formal imbalance-ratio documentation existed
before this audit; now recorded.

### 9–10. Data leakage / duplicate risk

- **Repository-level split is real and enforced before any labeling**
  (`configs/repos.yaml`, fixed train/val/test repo lists; `build_dataset.py`
  never mixes files across the fixed split). No sample-level random split
  exists anywhere in the pipeline — good, matches Phase G's absolute rule.
- **Struct-hash exact/near-duplicate detection already exists and already
  runs cross-split** (`ast_parser.py::_struct_hash`, position/formatting-
  independent `ast.dump`; `build_dataset.py::dedup_cross_split`, confirmed
  by `docs/dataset_report.md`: 124 methods / 17 classes dropped for
  appearing in more than one split, 1326 methods / 73 classes dropped as
  within-split repeats). This already substantially covers Phase F items 1-2.
- **Not covered**: fork/vendored-code detection beyond a directory-name
  denylist (`EXCLUDE_DIR_NAMES` includes `vendor`/`vendored`, but content
  copied into a differently-named directory would not be caught — the
  struct-hash dedup would still catch *identical* vendored functions/
  classes cross-split, but not a vendored file whose only match is at the
  whole-module level, since dedup operates on method/class units, not
  modules). No generated-code filtering (e.g. protobuf/parser-generated
  files with a "DO NOT EDIT" header) and no tiny/uninformative-file
  filtering exist.
- **Mild cross-split homogeneity note, not a leakage bug**: `train`
  includes `click`+`flask` (both Pallets Projects) and `httpx` (Encode);
  `val` includes `starlette` (also Encode). Same-organization projects
  often share contributor style/conventions. The struct-hash dedup already
  guards against literally shared code; this is a softer "style similarity"
  caveat worth keeping in mind when reading validation numbers as a proxy
  for generalization to unrelated codebases — not something to fix by
  itself, just to note.

### 11–14. Model / feature / CodeBERT / GAT limitations

Largely already discovered and documented across Phase 9/10/13 of this
project (not re-litigated, cited for completeness):

- `docs/hybrid_baseline_report.md`: the CodeBERT+GAT hybrid does **not**
  beat the structure-only GAT on any of the 3 tasks; hypothesis recorded
  there (768-dim CodeBERT block dominates a shallow `hidden_dim=32` linear
  encoder's compression budget over the 3-6 raw structural dims).
- **New, from this audit**: decorators (`FunctionInfo.decorators`,
  correctly captured by the parser) are **not** part of the node feature
  vector fed to the model at all (`graph_builder.py`'s method/function
  features are exactly `[loc, statement_count, param_count,
  self_access_count, external_access_count]`). The model has no explicit
  signal that a node is a `@staticmethod`/`@classmethod`/`@property` — it
  must infer this purely from label-time correlation with the other
  structural features, since `is_feature_envy` hard-labels those decorated
  methods as negative during training. This is workable (confirmed
  reasonable via the existing final-round label review) but is an implicit,
  not explicit, signal — a concrete, low-risk feature-engineering
  candidate if Feature Envy precision needs to improve later.
- Multi-task negative transfer already controlled-experiment-verified
  across Phase 13a-e: `EDGE_TYPES_CLASS_AGGREGATION` fixed God Class's
  receptive field but regressed Long Method/Feature Envy (documented root
  cause in `gat_baseline.py`); the shared-encoder Pareto trade-off between
  Long Method and Feature Envy/God Class under loss-reweighting was
  reproduced twice (`docs/class_pool_tuning_report.md`,
  `docs/class_pool_fixed_arch_report.md`) — genuinely a structural limit of
  this backbone at this depth/width, not an unexplored search space.

### 15. Multi-task interference — see above (already answered, Phase 13).

### 16. Inference/training mismatch

Checked directly against `backend/app/inference.py` (which I wrote this
session): same `parse_source` → `build_hetero_graph` → CodeBERT-embed →
`norm_stats` (loaded from the exact training run's saved stats, never
refit) → `HeteroGAT(edge_types=None, class_method_pool=True)` pipeline as
training, confirmed by `tests/test_backend_api.py` (real checkpoint, no
mocking). **No drift found on this axis** — this is the one area of the
Phase-A checklist that comes back clean.

### 17. Frontend/API representation issues

- Confidence is displayed as a raw sigmoid probability percentage
  (`SmellDetailPanel.jsx`, `CodeGraph.jsx`: `(prediction.probability *
  100).toFixed(1)}%`) with **no calibration disclosure**. No reliability
  curve / calibration study has been run at any point in this project.
  Per Phase O, displaying "99.7%" without evidence that 99.7%-confidence
  predictions are actually correct 99.7% of the time is a real gap —
  flagged, not yet fixed (would need a validation-set reliability-curve
  study before any UI wording changes).
- `class_name` fallback (`NOT_AVAILABLE`) and the "no smell prediction
  associated with this node" copy already correctly avoid the Phase S
  trap of implying "no smell shown = no problems" — confirmed by reading
  `SmellDetailPanel.jsx`/`CodeGraph.jsx`, no change needed here.
- Because of the nested-function/nested-class findings above, a real user
  pasting code with closures or nested classes will see an **incomplete**
  graph and get **no predictions at all** for those nodes, with no
  indication in the UI that anything was omitted — the API's own `summary`
  counts (`n_methods`, `n_functions`) are internally consistent with what
  the graph actually contains, but nothing tells the user "3 nested
  functions in this file were not analyzed." This is a direct, user-facing
  consequence of the §2 parser finding and the most concrete near-term fix
  candidate from a UX-correctness standpoint.

### 18. Any way a valid Python program could be incorrectly analyzed

Summary of the concrete failure modes found above:
1. Closures/nested functions: silently absent, no predictions, no warning.
2. Nested classes: present but mis-attributed to the module instead of
   their true parent class.
3. Any parameter-mediated method call (`obj.method()`): invisible to the
   graph's `calls` edges (only self-calls and bare-name calls resolve).
4. Feature Envy's external-access count: silently 2x-inflated for
   call-based envy vs. attribute-read-based envy.
5. Inheritance from any class not defined in the same file: invisible.
6. `@property`-wrapped delegation: not excluded from Feature Envy
   eligibility the way the structurally-similar classmethod pattern is.

None of these are crashes or parse failures — `docs/dataset_report.md` /
`docs/graph_validation_report.md` both report 0 parse errors and 100%
graph-build success across all three splits. The failure mode here is
uniformly "silent under- or mis-representation of valid Python," which is
exactly the class of bug the master prompt calls out as the one to find
before touching data volume or model size.

---

## 3. Phase B — Python representation verification (live-tested)

| Construct | Status | Evidence |
|---|---|---|
| Module | Represented (1 node, scalar `loc` feature only) | `graph_builder.py` |
| Top-level function | Represented | live test |
| Class | Represented | live test |
| Instance method | Represented | live test |
| `@classmethod` | Represented as `method` node; decorator captured for label-time exclusion only, not as a model feature | live test |
| `@staticmethod` | Represented as `method` node; same as above | live test |
| Nested function / closure | **Not represented — silently dropped** | live test, quantified on corpus |
| Nested class | Represented but **mis-parented to the module, not its true enclosing class** | live test |
| Constructors (`__init__`/`__new__`) | Represented as `method`; excluded from Feature Envy eligibility only | `label_rules.py` |
| Decorators (general) | Captured in `FunctionInfo.decorators`, used only for label-time rules, never a model feature | `ast_parser.py`, `graph_builder.py` |
| Imports | Represented (`import` node type + `imports` edge) | `graph_validation_report.md` |
| Parameters | Represented (`parameter` node + `contains`/`accesses` edges) | same |
| Local variables | **Deliberately not represented** (documented design choice) | `graph_builder.py` docstring |
| Attributes (`self.x = ...`) | Represented | `graph_validation_report.md` |
| Function calls (bare name) | Represented | confirmed |
| Method calls via `self` | Represented | confirmed |
| Method calls via any other receiver (parameter, local, chained) | **Not represented as an edge** | live test |
| Inheritance (same-file base) | Represented | confirmed |
| Inheritance (external/stdlib base) | **Not represented** | by construction |
| Module-level code (outside any def/class) | **Not represented** beyond the module's total `loc` scalar | `graph_builder.py` |

---

## 4. Phase C — function vs. method correctness

The parser already keeps `Function` and `Method` as genuinely distinct
`FunctionInfo.is_method` values and distinct graph node types (`function`
vs `method`) — no conflation happens in the graph itself.

Per prediction head, actual current scope:

| Task | Target node type(s) | Training examples | Label definition applies to |
|---|---|---|---|
| Long Method | `method` **and** `function` | Both, real (`is_long_method` takes any `FunctionInfo`, no `is_method` gate) | Method and Function equally |
| Feature Envy | `method` **only** | `method` only — `is_feature_envy` returns `False` immediately if `not fn.is_method` (`label_rules.py` line 69) | Method only, by design (the smell is about *object-oriented* misplaced responsibility; applying it to a free function has no well-defined "which class does this belong to instead" answer) |
| God Class | `class` only | `class` only | Class only, by definition |

This is already scientifically consistent, not a bug: **Feature Envy is
correctly method-only both in training and at inference** (`inference.py`
only calls `predict_feature_envy` on `method` nodes, mirroring
`is_feature_envy`'s `not fn.is_method` gate) — the earlier concern (checked
and disproved during this audit) that classmethod/staticmethod nodes might
be an inference/training distribution mismatch does not hold: they are
included as real negative training examples, not excluded from the graph,
so the model does see them during training as intended.

**Long Method is correctly trained and served on Function nodes too** —
already verified by the existing `backend/app/inference.py` code path
(`if "function" in h ...: predict_long_method(h, "function")`) and by
`train_hybrid_baseline.py`'s function-branch loss term. No gap here.

**No broader "Function-only smell" gap currently exists to fix** — the one
task that is method-only (Feature Envy) is method-only by a real, defensible
domain definition, not an accidental training limitation. This directly
answers Phase C's central question: **no dataset expansion or architecture
change is needed to make Feature Envy apply to functions**, because
applying it to functions would not be a scientifically valid smell
definition in the first place.

---

## 5. What this audit does *not* cover yet

Per the master prompt's explicit phase ordering, this document stops at
Phase C. It does **not** yet include: dataset-source investigation (Phase
D), a data-quality pipeline for any *new* data (Phase F), hyperparameter
search, retraining, baselines re-run, generalization stress tests,
calibration work, or the final TEST evaluation. TEST was not read, touched,
or evaluated at any point during this audit.

## 6. Preliminary decision framing (Phase T) — not yet decided

The findings above point toward **Option C (parser/graph representation
fixes) as the load-bearing next step, ahead of Option B (dataset
expansion)** — per the master prompt's own stated principle, "more data
does not solve incorrect labels, incorrect graph construction, or
training/inference mismatch." Concretely, three of the findings above are
fixable in the existing 547/514/1574-file corpus with **no new data at
all** and would change what the graphs the model already trains on
actually contain:

1. Wire nested functions into the graph (as `function` nodes scoped to
   their true parent, or a documented, deliberate decision to keep
   excluding them — either way, stop the silent drop).
2. Fix the `external_access_count`/`dominant_external_count` double-count
   (a small, local fix in `metrics.py`, would change Feature Envy labels
   and require rebuilding the dataset + retraining, but does not require
   any new source data).
3. Decide whether nested classes should get their own `contains` edge from
   the true parent class, or continue module-attribution with that
   explicitly documented as a known simplification.

Dataset expansion (Option B) may still be warranted afterward — the corpus
is small (9 train repos) and Feature Envy positives are scarce (163
examples) — but doing it before the above would mean training a larger
model on the same structurally-incomplete graphs and the same inflated
Feature Envy metric, which risks looking like an improvement in volume
while leaving the actual representation bugs in place.

**This is a recommendation, not a decision** — Phase D onward (dataset
investigation/expansion), any parser fix, and any retraining are explicitly
gated on user direction before proceeding, consistent with "do not
automatically choose dataset expansion" and this project's established
practice of confirming direction at phase boundaries.
