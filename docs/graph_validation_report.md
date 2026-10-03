# Graph Construction Validation Report

Thresholds used: LabelThresholds(long_method_statements=15, god_class_method_count=10, god_class_loc=192, feature_envy_min_external_calls=3)


## train

- graphs built: 547
- total n_module: 547 (avg 1.00/graph)
- total n_class: 1441 (avg 2.63/graph)
- total n_method: 6206 (avg 11.35/graph)
- total n_function: 2045 (avg 3.74/graph)
- total n_attribute: 2341 (avg 4.28/graph)
- total n_parameter: 10428 (avg 19.06/graph)
- total n_import: 8362 (avg 15.29/graph)
- total edges: 57518 (avg 105.15/graph)
- edge types present (count of graphs containing at least one edge of this type):
  - ('module', 'imports', 'import'): 509 graphs, 8362 total edges
  - ('module', 'contains', 'class'): 378 graphs, 1441 total edges
  - ('class', 'contains', 'method'): 356 graphs, 6206 total edges
  - ('method', 'belongs_to', 'class'): 356 graphs, 6206 total edges
  - ('method', 'contains', 'parameter'): 341 graphs, 7201 total edges
  - ('module', 'contains', 'function'): 316 graphs, 2045 total edges
  - ('function', 'contains', 'parameter'): 289 graphs, 3227 total edges
  - ('class', 'contains', 'attribute'): 262 graphs, 2341 total edges
  - ('method', 'accesses', 'parameter'): 262 graphs, 3328 total edges
  - ('method', 'uses', 'attribute'): 258 graphs, 10096 total edges
  - ('method', 'calls', 'method'): 227 graphs, 2835 total edges
  - ('function', 'accesses', 'parameter'): 193 graphs, 2211 total edges
  - ('function', 'calls', 'function'): 165 graphs, 1033 total edges
  - ('method', 'calls', 'function'): 116 graphs, 507 total edges
  - ('class', 'inherits', 'class'): 76 graphs, 479 total edges

## val

- graphs built: 514
- total n_module: 514 (avg 1.00/graph)
- total n_class: 2612 (avg 5.08/graph)
- total n_method: 12936 (avg 25.17/graph)
- total n_function: 1866 (avg 3.63/graph)
- total n_attribute: 3410 (avg 6.63/graph)
- total n_parameter: 19834 (avg 38.59/graph)
- total n_import: 13151 (avg 25.59/graph)
- total edges: 95441 (avg 185.68/graph)
- edge types present (count of graphs containing at least one edge of this type):
  - ('module', 'imports', 'import'): 502 graphs, 13151 total edges
  - ('module', 'contains', 'class'): 377 graphs, 2612 total edges
  - ('class', 'contains', 'method'): 366 graphs, 12936 total edges
  - ('method', 'belongs_to', 'class'): 366 graphs, 12936 total edges
  - ('method', 'contains', 'parameter'): 347 graphs, 16300 total edges
  - ('module', 'contains', 'function'): 307 graphs, 1866 total edges
  - ('function', 'contains', 'parameter'): 291 graphs, 3534 total edges
  - ('class', 'contains', 'attribute'): 277 graphs, 3410 total edges
  - ('method', 'uses', 'attribute'): 277 graphs, 14082 total edges
  - ('method', 'accesses', 'parameter'): 259 graphs, 6040 total edges
  - ('method', 'calls', 'method'): 222 graphs, 3882 total edges
  - ('function', 'accesses', 'parameter'): 214 graphs, 2539 total edges
  - ('function', 'calls', 'function'): 129 graphs, 855 total edges
  - ('class', 'inherits', 'class'): 121 graphs, 949 total edges
  - ('method', 'calls', 'function'): 100 graphs, 349 total edges

## test

- graphs built: 1574
- total n_module: 1574 (avg 1.00/graph)
- total n_class: 3766 (avg 2.39/graph)
- total n_method: 20112 (avg 12.78/graph)
- total n_function: 3934 (avg 2.50/graph)
- total n_attribute: 5245 (avg 3.33/graph)
- total n_parameter: 29254 (avg 18.59/graph)
- total n_import: 18708 (avg 11.89/graph)
- total edges: 148789 (avg 94.53/graph)
- edge types present (count of graphs containing at least one edge of this type):
  - ('module', 'imports', 'import'): 1275 graphs, 18708 total edges
  - ('module', 'contains', 'class'): 900 graphs, 3766 total edges
  - ('class', 'contains', 'method'): 844 graphs, 20112 total edges
  - ('method', 'belongs_to', 'class'): 844 graphs, 20112 total edges
  - ('method', 'contains', 'parameter'): 765 graphs, 21331 total edges
  - ('module', 'contains', 'function'): 652 graphs, 3934 total edges
  - ('function', 'contains', 'parameter'): 610 graphs, 7923 total edges
  - ('class', 'contains', 'attribute'): 580 graphs, 5245 total edges
  - ('method', 'uses', 'attribute'): 573 graphs, 21874 total edges
  - ('method', 'accesses', 'parameter'): 558 graphs, 8110 total edges
  - ('method', 'calls', 'method'): 493 graphs, 7950 total edges
  - ('function', 'accesses', 'parameter'): 396 graphs, 5105 total edges
  - ('function', 'calls', 'function'): 291 graphs, 2039 total edges
  - ('class', 'inherits', 'class'): 218 graphs, 1187 total edges
  - ('method', 'calls', 'function'): 203 graphs, 1393 total edges

## Failures

None — every parsed module was converted to a graph successfully.
