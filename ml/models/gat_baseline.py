# -*- coding: utf-8 -*-
"""Phase 7: shallow (2-layer, to avoid over-smoothing per the staged training
plan) heterogeneous GAT baseline — structure-only, same placeholder metrics
as node features (no CodeBERT semantics yet). Tests whether propagating
structural context (calls/uses/contains/inherits) beyond a node's own raw
metrics adds anything over the Phase 6 per-node metrics baseline.

Per-node-type raw feature widths come directly from ml/graph/graph_builder.py
(module=1, class=3, method=5, function=5, attribute=1, parameter=1, import=1).
A per-type Linear encoder projects every type into a shared hidden_dim before
message passing, since GATConv requires matching dimensions across an edge.

Leaf/source-only node types (module, attribute, parameter, import) are never
the destination of every edge they need to be a source for at the next layer
— HeteroConv only returns entries for node types that received at least one
message. `h_dict.update(...)` (not reassignment) is used after each conv so
untouched types keep their previous-layer embedding instead of vanishing.

Residual connections are required, not optional here: with add_self_loops=
False (necessary since most edge types are bipartite, e.g. class->method),
GATConv's output for a destination type is a pure function of its incoming
neighbors, with no path back to the node's own pre-conv features. A method
node's only incoming edge type is (class, contains, method) — without a
residual add, a method's own encoded statement_count/loc signal would be
fully overwritten by its parent class's embedding after just one layer.

HeteroGAT accepts an optional `edge_types` override (default: EDGE_TYPES).
Pass EDGE_TYPES_CLASS_AGGREGATION to additionally wire (method, belongs_to,
class), the reverse of (class, contains, method) — see that constant's
comment for why class nodes otherwise never see their methods' embeddings.

`class_method_pool=True` is a second, non-backbone way to give god_class
visibility into its methods — see the comment above the constructor for why
it exists and how it differs from EDGE_TYPES_CLASS_AGGREGATION.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.utils import scatter

NODE_FEATURE_DIMS: Dict[str, int] = {
    "module": 1, "class": 3, "method": 5, "function": 5,
    "attribute": 1, "parameter": 1, "import": 1,
}

EDGE_TYPES = [
    ("module", "imports", "import"),
    ("module", "contains", "class"),
    ("module", "contains", "function"),
    ("class", "contains", "method"),
    ("class", "contains", "attribute"),
    ("class", "inherits", "class"),
    ("method", "contains", "parameter"),
    ("function", "contains", "parameter"),
    ("method", "calls", "method"),
    ("method", "calls", "function"),
    ("function", "calls", "function"),
    ("method", "uses", "attribute"),
    ("method", "accesses", "parameter"),
    ("function", "accesses", "parameter"),
]

# Phase 13: (class, contains, method) only ever makes method the destination
# — class is never a destination of any edge, so under EDGE_TYPES a class
# node's embedding can only ever come from its own encoded metrics, module,
# and inherited classes, never from its methods (found via the Phase 12
# GNNExplainer receptive-field audit, god_class predictions). The reverse
# edge below is the minimal fix: it lets class aggregate its methods'
# embeddings without adding any other relation. Kept as a separate constant
# rather than folded into EDGE_TYPES so existing checkpoints trained under
# the original schema (models/hybrid_tuned, models/hybrid_baseline,
# models/gat_baseline) keep loading unmodified — HeteroConv only wires
# convolutions for the edge types it's constructed with, and silently
# ignores any other edge type keys present in a batch's edge_index_dict
# (see torch_geometric.nn.HeteroConv.forward), so graphs may carry this
# edge regardless of which EDGE_TYPES list the model consuming them uses.
EDGE_TYPES_CLASS_AGGREGATION = EDGE_TYPES + [("method", "belongs_to", "class")]

# Phase 13b: EDGE_TYPES_CLASS_AGGREGATION fixed the god_class receptive-field
# gap but regressed long_method/feature_envy too (not just god_class) on the
# same val split, same hyperparameters. Root cause: (class, contains, method)
# is also method's ONLY incoming edge type, so at conv2 a method node reads
# back its own class's embedding — and once belongs_to lets that class embed-
# ding absorb every sibling method's conv1 output, conv2 folds that sibling-
# aggregate signal straight back into each method's own embedding. That
# embedding is shared by ALL THREE heads, so long_method/feature_envy lose
# per-method discriminativeness even though neither task's own edges changed.
# CLASS_METHOD_POOL avoids this by never adding an edge the shared 2-layer
# backbone sees at all: it reuses the existing (class, contains, method)
# edge_index — present in every graph already — to mean-pool the BACKBONE's
# final method embeddings into a per-class vector *after* conv2, purely for
# the god_class head's input (concatenated with class's own conv2 embedding).
# method/function embeddings themselves are never touched, so long_method and
# feature_envy are structurally guaranteed to see the exact Phase 10 pathway.


class HeteroGAT(nn.Module):
    def __init__(self, hidden_dim: int = 32, heads: int = 2, dropout: float = 0.2,
                 node_feature_dims: Dict[str, int] = None, edge_types: list = None,
                 class_method_pool: bool = False):
        super().__init__()
        assert hidden_dim % heads == 0
        node_feature_dims = node_feature_dims or NODE_FEATURE_DIMS
        edge_types = edge_types or EDGE_TYPES
        self.edge_types = edge_types
        self.class_method_pool = class_method_pool
        self.encoders = nn.ModuleDict({
            nt: nn.Linear(dim, hidden_dim) for nt, dim in node_feature_dims.items()
        })
        conv_kwargs = dict(out_channels=hidden_dim // heads, heads=heads, add_self_loops=False)
        self.conv1 = HeteroConv(
            {et: GATConv(hidden_dim, **conv_kwargs) for et in edge_types}, aggr="sum"
        )
        self.conv2 = HeteroConv(
            {et: GATConv(hidden_dim, **conv_kwargs) for et in edge_types}, aggr="sum"
        )
        self.dropout = nn.Dropout(dropout)
        self.head_long_method = nn.Linear(hidden_dim, 1)
        self.head_feature_envy = nn.Linear(hidden_dim, 1)
        god_class_in = hidden_dim * 2 if class_method_pool else hidden_dim
        self.head_god_class = nn.Linear(god_class_in, 1)

    def encode(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {nt: torch.relu(self.encoders[nt](x)) for nt, x in x_dict.items() if x.shape[0] > 0}

    def _pool_methods_to_class(self, h_dict, edge_index_dict):
        num_class = h_dict["class"].shape[0]
        key = ("class", "contains", "method")
        if key not in edge_index_dict or "method" not in h_dict or edge_index_dict[key].shape[1] == 0:
            return h_dict["class"].new_zeros(num_class, h_dict["class"].shape[1])
        class_idx, method_idx = edge_index_dict[key]
        return scatter(h_dict["method"][method_idx], class_idx, dim=0, dim_size=num_class, reduce="mean")

    def forward(self, x_dict, edge_index_dict) -> Dict[str, torch.Tensor]:
        h_dict = self.encode(x_dict)
        out = self.conv1(h_dict, edge_index_dict)
        h_dict = {k: torch.relu(v + out[k]) if k in out else v for k, v in h_dict.items()}
        h_dict = {k: self.dropout(v) for k, v in h_dict.items()}
        out = self.conv2(h_dict, edge_index_dict)
        h_dict = {k: torch.relu(v + out[k]) if k in out else v for k, v in h_dict.items()}
        if self.class_method_pool and "class" in h_dict:
            h_dict = dict(h_dict)
            h_dict["class_method_pool"] = self._pool_methods_to_class(h_dict, edge_index_dict)
        return h_dict

    def predict_long_method(self, h_dict, node_type):
        return self.head_long_method(h_dict[node_type]).squeeze(-1)

    def predict_feature_envy(self, h_dict):
        return self.head_feature_envy(h_dict["method"]).squeeze(-1)

    def predict_god_class(self, h_dict):
        feat = h_dict["class"]
        if self.class_method_pool:
            feat = torch.cat([feat, h_dict["class_method_pool"]], dim=-1)
        return self.head_god_class(feat).squeeze(-1)
