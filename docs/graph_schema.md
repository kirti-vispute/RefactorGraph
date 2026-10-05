# Production Graph Schema

The parser and `ml/graph/graph_builder.py` construct one heterogeneous
graph per Python source file. The deployed model is
`models/experiment_fe_dominant/seed_43/`.

## Node Features

| Node type | Raw features | Deployed input width |
|---|---|---:|
| Module | LOC | 1 |
| Class | LOC, method count, field count | 771 |
| Method | LOC, statement count, parameter count, self-access count, dominant external-access count | 773 |
| Function | Same five features as method | 773 |
| Attribute | Constant 1 | 1 |
| Parameter | Constant 1 | 1 |
| Import | Constant 1 | 1 |

Class, method, and function inputs concatenate 768 frozen CodeBERT
embedding values with their raw features. The backend applies saved
TRAIN normalization statistics. Labels are stored separately from
input features; they are not fed to the classifier.

## Model Relationships

The deployed two-layer HeteroGAT uses these 14 configured edge types
from `ml/models/gat_baseline.py::EDGE_TYPES`:

- module imports import
- module contains class
- module contains function
- class contains method
- class contains attribute
- class inherits class
- method contains parameter
- function contains parameter
- method calls method
- method calls function
- function calls function
- method uses attribute
- method accesses parameter
- function accesses parameter

The graph builder also records nested containment, method ownership
(`method belongs_to class`), and additional resolvable calls/accesses.
These extra edge types can appear in the UI without being configured in
the deployed GAT backbone. Design B gives the God Class head a mean pool
of final method embeddings using `class contains method` edges after
message passing; it does not add the ownership edge to the backbone.

## Limits

Calls and inheritance are linked only when they can be resolved within
the analyzed file. Parameter accesses and self-attribute accesses are
represented, but the system does not perform full Python type inference
or construct a complete graph across repositories. Unknown external
classes are not invented as nodes. Local variables are not a separate
node type. External interactions that cannot be linked reliably are
still summarized by structural access counts.

The live UI displays this source-derived graph and structural metrics.
Offline GNNExplainer studies are listed separately in the
[documentation guide](README.md).
