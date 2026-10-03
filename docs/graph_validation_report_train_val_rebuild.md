# Graph Construction Validation Report -- Post-Fix Rebuild (TRAIN/VAL only, TEST excluded)

Thresholds used: LabelThresholds(long_method_statements=15, god_class_method_count=10, god_class_loc=192, feature_envy_min_external_calls=3, god_class_min_fields=1, god_class_min_methods_for_field_branch=3)


## train

- graphs built: 547
- total n_module: 547 (avg 1.00/graph)
- total n_class: 1441 (avg 2.63/graph)
- total n_method: 6256 (avg 11.44/graph)
- total n_function: 2339 (avg 4.28/graph)
- total n_attribute: 2341 (avg 4.28/graph)
- total n_parameter: 10805 (avg 19.75/graph)
- total n_import: 8362 (avg 15.29/graph)
- total edges: 58720 (avg 107.35/graph)
- edge types present (count of graphs containing at least one edge of this type):
  - ('module', 'imports', 'import'): 509 graphs, 8362 total edges
  - ('module', 'contains', 'class'): 374 graphs, 1408 total edges
  - ('class', 'contains', 'method'): 360 graphs, 6256 total edges
  - ('method', 'belongs_to', 'class'): 360 graphs, 6256 total edges
  - ('method', 'contains', 'parameter'): 344 graphs, 7269 total edges
  - ('module', 'contains', 'function'): 316 graphs, 2045 total edges
  - ('function', 'contains', 'parameter'): 306 graphs, 3536 total edges
  - ('method', 'accesses', 'parameter'): 264 graphs, 3340 total edges
  - ('class', 'contains', 'attribute'): 262 graphs, 2341 total edges
  - ('method', 'uses', 'attribute'): 261 graphs, 10128 total edges
  - ('method', 'calls', 'method'): 227 graphs, 2836 total edges
  - ('function', 'accesses', 'parameter'): 202 graphs, 2365 total edges
  - ('function', 'calls', 'function'): 175 graphs, 1176 total edges
  - ('method', 'calls', 'function'): 128 graphs, 596 total edges
  - ('class', 'inherits', 'class'): 76 graphs, 479 total edges
  - ('function', 'contains', 'function'): 75 graphs, 156 total edges
  - ('method', 'contains', 'function'): 57 graphs, 138 total edges
  - ('function', 'contains', 'class'): 9 graphs, 14 total edges
  - ('method', 'contains', 'class'): 7 graphs, 11 total edges
  - ('class', 'contains', 'class'): 6 graphs, 8 total edges

## val

- graphs built: 514
- total n_module: 514 (avg 1.00/graph)
- total n_class: 2612 (avg 5.08/graph)
- total n_method: 12966 (avg 25.23/graph)
- total n_function: 2610 (avg 5.08/graph)
- total n_attribute: 3410 (avg 6.63/graph)
- total n_parameter: 20925 (avg 40.71/graph)
- total n_import: 13151 (avg 25.59/graph)
- total edges: 98437 (avg 191.51/graph)
- edge types present (count of graphs containing at least one edge of this type):
  - ('module', 'imports', 'import'): 502 graphs, 13151 total edges
  - ('module', 'contains', 'class'): 377 graphs, 2538 total edges
  - ('class', 'contains', 'method'): 366 graphs, 12966 total edges
  - ('method', 'belongs_to', 'class'): 366 graphs, 12966 total edges
  - ('method', 'contains', 'parameter'): 347 graphs, 16339 total edges
  - ('function', 'contains', 'parameter'): 347 graphs, 4586 total edges
  - ('module', 'contains', 'function'): 307 graphs, 1866 total edges
  - ('class', 'contains', 'attribute'): 277 graphs, 3410 total edges
  - ('method', 'uses', 'attribute'): 277 graphs, 14122 total edges
  - ('function', 'accesses', 'parameter'): 260 graphs, 3251 total edges
  - ('method', 'accesses', 'parameter'): 260 graphs, 6042 total edges
  - ('method', 'calls', 'method'): 222 graphs, 3882 total edges
  - ('function', 'calls', 'function'): 142 graphs, 1041 total edges
  - ('method', 'contains', 'function'): 125 graphs, 491 total edges
  - ('method', 'calls', 'function'): 123 graphs, 534 total edges
  - ('class', 'inherits', 'class'): 121 graphs, 925 total edges
  - ('function', 'contains', 'function'): 77 graphs, 253 total edges
  - ('class', 'contains', 'class'): 22 graphs, 62 total edges
  - ('method', 'contains', 'class'): 6 graphs, 9 total edges
  - ('function', 'contains', 'class'): 3 graphs, 3 total edges

## Failures

None.
