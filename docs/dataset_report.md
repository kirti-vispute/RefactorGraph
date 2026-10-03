# RefactorGraph Dataset Report

## Files parsed

- train: 547 .py files, 0 parse errors
- val: 514 .py files, 0 parse errors
- test: 1574 .py files, 0 parse errors

## Deduplication

- Methods: 124 dropped (cross-split collision), 1326 dropped (within-split duplicate)
- Classes: 17 dropped (cross-split collision), 73 dropped (within-split duplicate)

## Label thresholds (derived from TRAIN split only, 90th percentile, floors applied)

- Long Method: statement_count >= 15
- God/Large Class: method_count >= 10 AND LOC >= 192
- Feature Envy: dominant external receiver access count >= 3 AND > self access count

## Label counts per split

- train: 8131 methods (long_method=904, feature_envy=163); 1436 classes (god_class=118)
- val: 13905 methods (long_method=1160, feature_envy=277); 2575 classes (god_class=181)
- test: 23613 methods (long_method=2301, feature_envy=631); 3718 classes (god_class=295)

## Known limitations

- Labels are RULE-BASED (silver), derived from our own AST metrics, not human-annotated. A stratified manual-review subset must be scored for precision before trusting these as gold.
- Feature Envy heuristic excludes imported-module and builtin receivers but has no real type inference; a parameter reassigned to a different type mid-method, or attribute access through a deep alias, can still be mis-attributed.
- Thresholds are percentile-based on this specific 9-repo train pool; not claimed universal.