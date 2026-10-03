# Dataset Report -- Post-Fix Rebuild (TRAIN/VAL only, TEST excluded)

Rebuilt after the robustness-audit parser/graph fixes (docs/robustness_audit_phase_a.md): nested function/class representation, scoped call resolution, external_access_count double-count fix. TEST deliberately not touched -- see this script's module docstring. Compare against docs/dataset_report.md (the original, TEST-inclusive baseline) for what changed.

## Files parsed

- train: 547 .py files, 0 parse errors
- val: 514 .py files, 0 parse errors

## Deduplication (train<->val only -- see module docstring)

- Methods: 50 dropped (cross-split collision), 990 dropped (within-split duplicate)
- Classes: 0 dropped (cross-split collision), 34 dropped (within-split duplicate)

## Label thresholds (derived from TRAIN split only, 90th percentile, floors applied)

- Long Method: statement_count >= 15
- God/Large Class: (method_count >= 10 AND LOC >= 192) OR (LOC >= 192 AND field_count >= 1 AND method_count >= 3)  [see docs/godclass_formula_revision.md]
- Feature Envy: dominant external receiver access count >= 3 AND > self access count

## Label counts per split

- train: 8479 methods (long_method=920, feature_envy=175); 1436 classes (god_class=137)
- val: 14652 methods (long_method=1203, feature_envy=258); 2583 classes (god_class=208)