# RefactorGraph Robustness Audit — Phase A/B/C

Audit-only. No retraining, no dataset changes, no TEST access performed while
producing this document. Every finding below is backed by either a source
read (file:line) or a live empirical test run against the actual code in
this repo (commands and output kept in this session's log). TEST split
confirmed untouched before and after this audit (`data/processed/graphs_hybrid/test`
still 1574 files, `models/hybrid_class_pool_tuned/model.pt` timestamp
unchanged, `docs/final_test_evaluation_report.md` unchanged).

## Pipeline as it exists today

```
.py source --ast_parser.py--> ModuleInfo (classes/functions/imports)
          --graph_builder.py--> HeteroData (7 node types, structural features)
          --augment_graphs_with_codebert.py--> + frozen CodeBERT 768-dim embedding
          --train_hybrid_baseline.py normalize--> z-scored per node type
          --HeteroGAT (2-layer GATConv, class_method_pool=True)--> per-task heads
          --inference.py (backend)--> same pipeline, single file, live
```

Final selected checkpoint: `models/hybrid_class_pool_tuned` (Phase 13c
free-arch search winner: heads=2, hidden_dim=64). TEST macro-F1=0.650
(`docs/final_test_evaluation_report.md`).

---

## PHASE A — Pipeline audit findings

### 1–2. Parser / graph-construction limitations (confirmed empirically)

**A1. Nested functions/closures are silently dropped from the graph — the single largest representational gap found.**

`ast_parser.py::_handle_function`:
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
A function nested inside another function (a closure/local helper) is
appended to `nested_functions` on its parent `FunctionInfo`, but
`graph_builder.py` only ever iterates `mod.classes[*].methods` and
`mod.functions` — `nested_functions` is never read anywhere in the graph
construction path. Verified live:
```
mod = parse_source(SRC)              # SRC defines `helper` inside a method
closure_method.nested_functions      # -> ['helper']   (captured)
'helper' in all_reachable_names      # -> False         (never a graph node)
```
**Scanned the real training corpus** (1899 train `.py` files):
- 424/1899 files (22.3%) contain at least one nested/closure function.
- **3110 nested functions total are silently dropped** — for comparison,
  the graph currently has 7675 top-level function nodes and 16871 method
  nodes. Nested functions are ~14% of the volume that top-level functions
  contribute, entirely invisible to the model, the graph, and any smell
  prediction (long_method is never checked on a closure regardless of its
  own length; a closure could itself be a very long/feature-envious
  function and it is simply never asked about).
- Not a crash, not a warning — a silent omission of a valid Python
  construct. Directly violates the "do not silently discard valid Python
  constructs" requirement.

**A2. Nested classes are flattened, not scoped, producing an incorrect `module→contains→class` edge.**

`visit_ClassDef` always does `self.classes.append(cls)` regardless of
nesting depth. Verified live: a class `Inner` defined inside class `Outer`
appears in `mod.classes` as a plain top-level sibling of `Outer`
(`['Inner', 'Outer']`). `graph_builder.py` then wires
`(module, contains, class)` from the module directly to `Inner`, which
misrepresents the real containment (`Outer` contains `Inner`, not the
module). `Inner`'s own methods/attributes are still correctly scoped to
`Inner` (not folded into `Outer`), so god_class/long_method metrics for
existing classes are not corrupted by this — only the class↔module edge
is factually wrong, and no `class-contains-class` relation exists to
represent real nesting even though the schema is silent on it.

**A3. The `calls` edge is close to non-functional for the single most common real call pattern.**

`graph_builder.py::_resolve_call_target`:
```python
if receiver == "self" and caller.owner_class_idx is not None: ...
if receiver is None: ...                       # bare name call
return None                                      # everything else: unresolved
```
Any call through a **named receiver that is not literally `self`** —
i.e. calling a method on a parameter, a local variable, `self.attr`, or
any object reference — is unconditionally unresolved and produces no edge.
Verified live on a snippet with 4 real calls
(`service.validate()/process()/finish()`, `obj.do_something()`):
**zero `calls` edges were produced.** This is exactly the pattern Feature
Envy is defined on (repeated calls to methods on an external
collaborator/parameter) — the GAT's message-passing structure never sees
that collaboration as a graph edge; the corpus-wide numbers in
`docs/graph_validation_report.md` confirm this isn't a fluke: every
`calls` edge in the entire train/val/test corpus is either a same-class
`self.foo()` call or a bare top-level-function call — never a
parameter-mediated or attribute-chain call. This materially limits what
the GAT's structural message-passing can contribute beyond the raw
per-node metrics already in the feature vector, and is a plausible
contributing explanation (see A5, and `docs/hybrid_baseline_report.md`)
for why the structure-only GAT did not clearly outperform simpler
baselines on this graph schema.

**A4. `external_access_count` / `dominant_external_count` double-count call-based access relative to attribute-only access.**

`metrics.py::function_metrics` increments the same `external_receivers`
Counter from **two separate AST visitor passes**: once from
`fn.attr_accesses` (fires for `x.foo` as an Attribute node) and once from
`fn.calls` (fires for `x.foo(...)` as a Call node wrapping that same
Attribute). A plain method call `x.foo()` therefore contributes **2** to
the counter, while a pure attribute read `x.foo` (no call) contributes 1.
Verified live: 3 distinct method calls on the same receiver
(`service.validate()/process()/finish()`) produced
`external_access_count = 6`, `dominant_external_count = 6`, not 3.
Consequence: the documented Feature Envy rule ("dominant external
receiver access count >= 3") is **not "3 distinct interactions"** for
call-heavy code — 2 real calls already clears the threshold (2×2=4 ≥ 3),
while 3 real *plain attribute reads* (no calls) exactly meets it at
3×1=3. The rule is internally self-consistent (the 90th-percentile
threshold in `configs/label_thresholds.json` was itself derived using
this same inflated counter, so labels are not "wrong" relative to their
own calibration), but the semantic meaning documented everywhere
("count of external accesses") is misleading, and the asymmetry between
call-style and attribute-style envy is unintended, not a deliberate
design choice recorded anywhere.

**A5. Decorators are parsed but never reach the model as a feature.**

`FunctionInfo.decorators` (e.g. `['staticmethod']`, `['classmethod']`,
`['property']`) is used only inside `label_rules.py::is_feature_envy` at
**label-generation time** (excludes `staticmethod`/`classmethod` from
positive eligibility, assigning them a hard negative label instead — this
is correct and already covered by 5 rounds of manual label review, see
Phase E below). It is never concatenated into the method/function node
feature vector (`ml/graph/graph_builder.py`'s method features are exactly
`[loc, statement_count, param_count, self_access_count,
external_access_count]`). The model has no explicit signal for "this is a
classmethod/staticmethod/property" — it must infer that indirectly from
correlated structural patterns. Not a correctness bug (classmethod/
staticmethod nodes are still correctly labeled negative during training),
but a real feature-completeness limitation worth listing under "model
limitations."

**A6. Multi-inheritance and cross-file inheritance are only partially represented — already documented, confirmed correct as designed.** `class_index` lookup only connects `inherits` edges when the base class is defined in the *same file* (`graph_builder.py` docstring says so explicitly). External/stdlib bases (the common case) never get an edge. This is a known, already-documented limitation, not a new finding — listed here only for completeness of the audit checklist.

**A7. Module-level code (statements outside any class/def) has no representation beyond a single scalar `loc` feature on the module node.** No smell head targets "module" today, so this isn't blocking current functionality, but a module doing significant top-level work (script-style files) is structurally invisible beyond line count.

### 3–6. Missing node/edge types, incorrect relationships
Covered above (A1–A3 are the substantive findings). No entirely-missing
node *type* was found relative to the documented 7-type schema; the gaps
are within existing types (nested functions never becoming nodes; nested
classes becoming wrongly-parented nodes) and within the `calls` edge's
resolution logic, not missing edge *types*.

### 7–8. Label limitations, class imbalance
Already extensively investigated by an earlier phase of this project —
see `docs/label_review_findings.md` (5 rounds of manual review, 7 concrete
bugs found and fixed with regression tests) and `docs/dataset_report.md`.
Re-verified the referenced files still exist and match; summarizing +
adding class-imbalance ratios (computed from `docs/dataset_report.md`
counts, not previously stated as ratios anywhere):

| task | train pos-rate | val pos-rate | test pos-rate | imbalance |
|---|---|---|---|---|
| long_method | 904/8131 = 11.1% | 1160/13905 = 8.3% | 2301/23613 = 9.7% | ~9:1 |
| god_class | 118/1436 = 8.2% | 181/2575 = 7.0% | 295/3718 = 7.9% | ~12:1 |
| feature_envy | 163/8131 = 2.0% | 277/13905 = 2.0% | 631/23613 = 2.7% | **~49:1** |

Feature Envy's imbalance is far more severe than the other two tasks, and
consistent across all three splits (not an artifact of one split). This,
combined with the label-noise finding already on record (Visitor/Strategy
pattern residual false positives, `docs/label_review_findings.md`), is
the most plausible compounding explanation for why feature_envy has had
the lowest F1 of the three tasks in *every* model variant tried so far
(0.31–0.42 across all baseline/tuning reports) — it is simultaneously the
rarest and the noisiest label.

### 9–10. Data leakage risk, duplicate/near-duplicate risk
**Already substantially addressed, verified by reading the actual code
(not just trusting the report):** `scripts/build_dataset.py::dedup_cross_split`
drops any method/class whose position-independent structural hash
(`ast.dump(annotate_fields=False, include_attributes=False)`, see
`ast_parser.py::_struct_hash`) appears in more than one split (124 methods
+ 17 classes dropped this way per `docs/dataset_report.md`), and collapses
duplicates within a split (1326 methods + 73 classes). Splitting is
strictly repository-level (`configs/repos.yaml`: 9 train / 3 val / 3 test
repos, fixed before any labeling). This already satisfies most of Phase
F/G's stated requirements. Residual gaps (not yet implemented):
- No generated-code filtering beyond directory-name exclusion (`migrations`,
  `vendor` etc. — content-based generated-file detection, e.g. "DO NOT
  EDIT" headers or `.pyi` stub patterns, is not implemented).
- No tiny/uninformative-file filtering (a near-empty `__init__.py` still
  contributes a module node).
- `httpx` (train) and `starlette` (val) share the same GitHub org/author
  (`encode`) and coding conventions — not literal duplication (struct-hash
  dedup would catch that), but a mild stylistic-homogeneity risk worth
  naming explicitly since it wasn't called out anywhere before.

### 11–16. Model/architecture/multi-task limitations
Already extensively, empirically investigated in this project's own prior
phases — re-verified the claims still hold by re-reading the relevant
code and reports rather than re-deriving from scratch:
- `docs/hybrid_baseline_report.md`: CodeBERT-augmented hybrid does **not**
  beat the structure-only GAT on long_method or god_class on the same val
  split — documented dilution hypothesis (24:1 compression ratio between a
  768-dim CodeBERT block and a 3–6-dim structural block through the same
  Linear encoder). Given finding A3 above (calls edges rarely fire), this
  dilution is compounded by structure itself carrying less signal than
  the architecture assumes.
- `docs/class_pool_report.md` / `class_pool_tuning_report.md` /
  `class_pool_fixed_arch_report.md`: multi-task negative transfer between
  long_method and feature_envy/god_class already confirmed via controlled
  experiments (three separate Optuna searches), concluded as a genuine
  Pareto trade-off under the shared 2-layer backbone, not resolvable by
  loss-weighting alone.
- **New finding (A8): god_class's method-level signal is an unweighted mean, not a distribution.** `HeteroGAT._pool_methods_to_class` (`ml/models/gat_baseline.py`) mean-pools all of a class's *final* method embeddings into one vector for the god_class head. A class with one very long method and nine short ones produces the same pooled vector shape/scale as one with ten medium methods — the pooling discards variance/outlier information that could matter for God Class specifically (a class dominated by one giant method is a different smell shape than one with uniformly bloated methods). `method_count` itself is still a raw class feature, so this is a nuance, not a blind spot, but worth recording as an architecture limitation (Phase I item 6).

### 17. Frontend/API representation issues
- **Confidence is displayed as a raw, uncalibrated sigmoid output** (`frontend/src/components/SmellDetailPanel.jsx`: `{(p.probability * 100).toFixed(1)}%`) labeled simply "Confidence." No calibration study has been performed (Phase O of the master prompt asks for this explicitly) — the UI does not currently caveat that 91% is a raw model score, not a measured empirical hit-rate. This is a real, fixable gap; flagged for Phase O, not fixed in this audit pass.
- No other API/frontend representation issues found: `/analyze`'s `graph.nodes`/`graph.edges` are read directly from the same `HeteroData` object built for inference (not re-derived), node ids are index-matched to the tensors the model actually scored (`ml/graph/node_naming.py`), and `NOT_AVAILABLE` fallbacks are used everywhere real data is absent (verified in Phase 17's own test suite, `tests/test_backend_api.py`).

### 18. Any reason a valid Python program could be incorrectly analyzed
Consolidates A1–A5 above: a closure-heavy file will under-report long
methods/feature envy (closures invisible); a file with nested classes
will show a structurally-wrong module→class edge for the inner class; a
file whose objects communicate primarily through parameters (very common)
will show almost no `calls` edges regardless of how much real
collaboration exists; and Feature Envy's magnitude signal is inflated
~2x for call-heavy code relative to attribute-read-heavy code.

---

## PHASE B — Python representation audit (empirically verified, not assumed)

| construct | supported? | evidence |
|---|---|---|
| modules | yes | `ModuleInfo`, 1 node/file, always present |
| top-level functions | yes | `mod.functions`, verified in live test |
| classes | yes (top-level); **nested classes flattened, see A2** | live test: `Inner`+`Outer` both appear in `mod.classes` |
| instance methods | yes | `is_method` logic verified |
| classmethods | yes, decorator captured; **excluded from graph node features (A5)** | live test: `from_config` decorators=['classmethod'] |
| staticmethods | yes, decorator captured; same A5 caveat | live test: `static_thing` decorators=['staticmethod'], is_method=True |
| **nested/closure functions** | **NO — silently dropped (A1)** | live test + corpus scan: 3110 dropped in train alone |
| nested classes | partially — becomes a node, but wrongly parented (A2) | live test |
| constructors (`__init__`/`__new__`) | yes, explicitly excluded from Feature Envy eligibility (correct per Fowler's own exception) | `label_rules.py` |
| decorators (general) | parsed, stored, used only for label eligibility, not fed to model (A5) | — |
| imports (`import`, `from...import`, `as`) | yes | `visit_Import`/`visit_ImportFrom`, alias handling confirmed |
| parameters | yes | node type + `contains`/`accesses` edges |
| variables (local) | **deliberately not a node type** — tracked only to *exclude* them from Feature Envy's external-receiver count (`local_names`) | by design, documented in `graph_builder.py` |
| attributes (`self.x`) | yes | node type, `uses` edge (method/function→attribute) |
| function calls | yes for bare-name calls; **no edge for parameter/variable/attribute-chain-mediated calls (A3)** | live test |
| method calls | yes only for literal `self.foo()`; same A3 caveat otherwise | live test + corpus-wide edge stats |
| inheritance | yes, same-file bases only (documented, not a new finding) | `graph_builder.py` docstring |
| module-level code | only a scalar `loc`, no structural detail (A7) | — |

**No valid Python construct causes a parse failure or a silent full-file
skip** — `parse_error_summary` in `build_dataset.py` reports 0 parse
errors across all three splits, and `parse_source` degrades gracefully
(returns a `ModuleInfo` with `parse_error` set, never raises). The gaps
found are all *within* a successfully-parsed file (specific constructs
becoming invisible or mis-scoped), not whole-file failures.

---

## PHASE C — Function vs Method correctness

The system already keeps Function and Method as genuinely distinct
concepts end-to-end — this was checked carefully since it's exactly the
kind of "convert Function→Method to satisfy an existing model" shortcut
the master prompt warns against, and **no such shortcut exists here**:

| smell | trained on | inference support | evidence |
|---|---|---|---|
| Long Method | **both** method and function nodes (`y_long_method` computed for both in `graph_builder.py`) | both — `inference.py` calls `predict_long_method(h, "method")` and `predict_long_method(h, "function")` separately | verified in `backend/app/inference.py` and its test `test_analyze_returns_predictions_for_every_node` |
| Feature Envy | **method nodes only** — `is_feature_envy` returns `False` unconditionally for `not fn.is_method` | correctly method-only — `HeteroGAT.predict_feature_envy(h)` reads `h_dict["method"]` only, never called for functions in `inference.py` or `train_hybrid_baseline.py` | `label_rules.py:69`, `gat_baseline.py:153` |
| God Class | class nodes only (no function/method equivalent smell exists) | class-only | — |

This is scientifically consistent: the model does **not** claim
Feature Envy support for arbitrary top-level functions (correctly
omitted from both training labels and the inference response's
`feature_envy` array), and Long Method is legitimately trained and
served for both callable types since "a function/method body that is too
long" is the same underlying structural claim regardless of whether the
callable happens to belong to a class. No architecture change is needed
here — Phase C's job was to verify, not fix, and the existing design
already reflects the right decision (train Long Method jointly on the
`_MethodLikeNode` abstraction that already unifies method/function
feature extraction in `graph_builder.py`, keep Feature Envy method-only
by construction of the label rule, not a post-hoc filter).

---

## Evidence-based decision (Phase T framing, informational — no phase D–S work has been performed yet)

Per the master prompt's own instruction ("more data does not solve
incorrect labels, incorrect graph construction, or training/inference
mismatch"): the findings above are dominated by **graph-construction and
feature-measurement issues (A1–A5)**, not by dataset size or label
quality — the existing 5-round gold review already shows Long
Method/God Class labels are clean, and Feature Envy's weakness is better
explained by severe imbalance (~49:1) plus a documented, hard-to-fix
labeling ambiguity (Visitor/Strategy pattern) than by needing more raw
files. This points toward **Option C (fix parser/graph representation
first)** as the technically justified next step before any Option B
(dataset expansion) work — expanding the dataset on top of a graph
construction that drops 22% of files' closures and produces almost no
`calls` edges for parameter-mediated collaboration would mostly train the
model harder on the same structural blind spots, not fix them.

This is a recommendation, not an executed decision — per the master
prompt's Phase U rule and this project's own established practice, next
steps (which of A1–A5 to fix, whether to fix A4's double-counting given
it may shift label thresholds, whether to pursue Phase D dataset work at
all) should be confirmed before any code/data/training change is made.

**No files were modified in this audit.** TEST split, `models/hybrid_class_pool_tuned`, and all training/dataset/graph-construction code remain exactly as they were before Phase A began.
