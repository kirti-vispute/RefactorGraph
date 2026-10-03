# -*- coding: utf-8 -*-
"""Build one heterogeneous PyTorch Geometric graph per source file.

Node types: module, class, method, function, attribute, parameter, import.
("Variable" — local variables — is deliberately dropped from the schema:
local names are already excluded from the Feature Envy signal in
metrics.py, and turning every local variable into a graph node would blow
up graph size for no signal a GNN could use; see docs/graph_schema.md.)

Edge types (only relations that can be resolved reliably WITHIN a single
file, without full type inference — see docs/graph_schema.md for what was
deliberately left out and why):
  (module, contains, class) / (module, contains, function) / (module, imports, import)
                                    — used for a TOP-LEVEL class/function (parent_qualname is
                                      None); a nested class/function instead gets a `contains`
                                      edge from its REAL parent (class, method, or function --
                                      see the two entries below), never from the module.
  (class, contains, class)         — a class nested inside another class (e.g. `class Meta:`)
  (method, contains, function) / (function, contains, function) / (method, contains, class) /
  (function, contains, class)      — a function/closure or class nested inside a function or
                                      method body. Previously such a nested def was silently
                                      dropped entirely (never a graph node at all) -- see
                                      docs/robustness_audit_phase_a.md for how this was found
                                      and quantified. Reuses the same `contains` relation
                                      already used for every other containment pair rather than
                                      inventing a new relationship type.
  (class, contains, method) / (class, contains, attribute)
  (method, belongs_to, class)      — reverse of (class, contains, method); added so a class
                                      node's embedding can aggregate its methods' embeddings
                                      during message passing (class is otherwise never the
                                      destination of any method-derived edge — see
                                      EDGE_TYPES_CLASS_AGGREGATION in ml/models/gat_baseline.py).
                                      Always present in the graph; whether a model's own
                                      HeteroConv actually wires it depends on the edge_types
                                      the model was constructed with, since HeteroConv ignores
                                      edge types it wasn't built with.
  (method, contains, parameter) / (function, contains, parameter)
  (class, inherits, class)        — only when the base class is defined in the same file
  (method, calls, method) / (method, calls, function) / (function, calls, function) / (function, calls, method)
                                    — only when the callee resolves to a method/function in the same file
  (method, uses, attribute) / (function, uses, attribute)
                                    — self.<attr> access, i.e. intra-class use
  (method, accesses, parameter) / (function, accesses, parameter)
                                    — access on a name that IS one of the method's own
                                      parameters, i.e. the reliably-resolvable half of the
                                      Feature Envy signal (accesses on an unresolvable
                                      receiver — a bare global, an unannotated local typed
                                      externally — are not linked to a node: fabricating a
                                      node for an unknown external class would be exactly the
                                      "artificial relationship" the project spec says to avoid)

Node features are placeholder structural metrics (pre-CodeBERT — semantic
embeddings are wired in during the CodeBERT phase, per the staged training
plan: validate the graph pipeline before adding the transformer). None of
the label booleans (long_method/god_class/feature_envy) appear in the
feature vectors — they are stored separately as `y` tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from ml.preprocessing.ast_parser import ClassInfo, FunctionInfo, ModuleInfo
from ml.preprocessing.label_rules import LabelThresholds, is_feature_envy, is_god_class, is_long_method
from ml.preprocessing.metrics import class_metrics, function_metrics


@dataclass
class _MethodLikeNode:
    fn: FunctionInfo
    node_type: str  # "method" or "function"
    index: int
    owner_class_idx: Optional[int]  # None for module-level functions


@dataclass
class GraphBuildResult:
    data: HeteroData
    stats: Dict[str, int] = field(default_factory=dict)


def build_hetero_graph(mod: ModuleInfo, thresholds: LabelThresholds) -> GraphBuildResult:
    if not mod.ok:
        raise ValueError(f"cannot build a graph from a module with a parse error: {mod.parse_error}")

    data = HeteroData()

    # Keyed by the class's unique qualname (not simple name) so two
    # unrelated nested classes sharing a simple name (e.g. two different
    # `class Meta:` inner classes) never collide -- for a true top-level
    # class qualname == name, so this is unchanged for the common case.
    class_index: Dict[str, int] = {c.qualname: i for i, c in enumerate(mod.classes)}
    method_like: List[_MethodLikeNode] = []
    method_index_by_qualname: Dict[str, int] = {}
    # For (contains) parent-edge lookups by full qualname -- includes
    # nested closures, unlike the by-name index below.
    function_index_by_qualname: Dict[str, int] = {}
    # For bare-name call resolution among genuinely top-level functions only.
    top_level_function_index_by_name: Dict[str, int] = {}
    # For bare-name call resolution to a closure from within its own
    # immediate enclosing function/method -- keyed by (enclosing qualname,
    # simple name), so same-named closures in different scopes (a very
    # common pattern: many methods each define their own local "helper")
    # never resolve to the wrong one.
    nested_function_index: Dict[Tuple[str, str], int] = {}

    for ci, cls in enumerate(mod.classes):
        for fn in cls.methods:
            idx = len(method_like)
            method_like.append(_MethodLikeNode(fn=fn, node_type="method", index=idx, owner_class_idx=ci))
            method_index_by_qualname[fn.qualname] = idx
    method_count = len(method_like)

    function_nodes: List[_MethodLikeNode] = []
    for fn in mod.functions:
        idx = len(function_nodes)
        function_nodes.append(_MethodLikeNode(fn=fn, node_type="function", index=idx, owner_class_idx=None))
        function_index_by_qualname[fn.qualname] = idx
        if fn.parent_qualname is None:
            top_level_function_index_by_name[fn.name] = idx
        else:
            nested_function_index[(fn.parent_qualname, fn.name)] = idx

    def _resolve_parent_node(qualname: Optional[str]) -> Optional[Tuple[str, int]]:
        """Generic (node_type, index) lookup for a `contains` parent edge --
        the parent of a nested class/function may itself be a class, a
        method, or a top-level/nested function."""
        if qualname is None:
            return None
        if qualname in class_index:
            return "class", class_index[qualname]
        if qualname in method_index_by_qualname:
            return "method", method_index_by_qualname[qualname]
        if qualname in function_index_by_qualname:
            return "function", function_index_by_qualname[qualname]
        return None

    attribute_index: Dict[Tuple[int, str], int] = {}
    for ci, cls in enumerate(mod.classes):
        for attr in sorted(cls.class_attrs):
            attribute_index[(ci, attr)] = len(attribute_index)

    parameter_index: Dict[Tuple[str, int, str], int] = {}  # (owner_type, owner_idx, param_name) -> idx
    for m in method_like:
        for p in m.fn.params:
            if p == "self" or p == "cls":
                continue
            parameter_index[("method", m.index, p)] = len(parameter_index)
    for f in function_nodes:
        for p in f.fn.params:
            parameter_index[("function", f.index, p)] = len(parameter_index)

    import_index: Dict[str, int] = {}
    for imp in mod.imports:
        for name in imp.names:
            if name not in import_index:
                import_index[name] = len(import_index)

    data["module"].x = torch.tensor([[float(mod.loc)]], dtype=torch.float)

    if mod.classes:
        class_feats = []
        class_labels = []
        for cls in mod.classes:
            cm = class_metrics(cls)
            class_feats.append([float(cm.loc), float(cm.method_count), float(cm.field_count)])
            class_labels.append(1.0 if is_god_class(cls, thresholds) else 0.0)
        data["class"].x = torch.tensor(class_feats, dtype=torch.float)
        data["class"].y = torch.tensor(class_labels, dtype=torch.float)
    else:
        data["class"].x = torch.zeros((0, 3), dtype=torch.float)
        data["class"].y = torch.zeros((0,), dtype=torch.float)

    def _method_like_features(nodes: List[_MethodLikeNode], parent_mod: ModuleInfo):
        feats, long_method_labels, feature_envy_labels = [], [], []
        for n in nodes:
            fm = function_metrics(n.fn, parent_mod)
            feats.append([
                float(fm.loc), float(fm.statement_count), float(fm.param_count),
                float(fm.self_access_count), float(fm.dominant_external_count),
            ])
            long_method_labels.append(1.0 if is_long_method(n.fn, parent_mod, thresholds) else 0.0)
            feature_envy_labels.append(1.0 if is_feature_envy(n.fn, parent_mod, thresholds) else 0.0)
        return feats, long_method_labels, feature_envy_labels

    if method_like:
        feats, lm, fe = _method_like_features(method_like, mod)
        data["method"].x = torch.tensor(feats, dtype=torch.float)
        data["method"].y_long_method = torch.tensor(lm, dtype=torch.float)
        data["method"].y_feature_envy = torch.tensor(fe, dtype=torch.float)
    else:
        data["method"].x = torch.zeros((0, 5), dtype=torch.float)
        data["method"].y_long_method = torch.zeros((0,), dtype=torch.float)
        data["method"].y_feature_envy = torch.zeros((0,), dtype=torch.float)

    if function_nodes:
        feats, lm, fe = _method_like_features(function_nodes, mod)
        data["function"].x = torch.tensor(feats, dtype=torch.float)
        data["function"].y_long_method = torch.tensor(lm, dtype=torch.float)
        data["function"].y_feature_envy = torch.tensor(fe, dtype=torch.float)  # always 0: not eligible
    else:
        data["function"].x = torch.zeros((0, 5), dtype=torch.float)
        data["function"].y_long_method = torch.zeros((0,), dtype=torch.float)
        data["function"].y_feature_envy = torch.zeros((0,), dtype=torch.float)

    n_attrs = len(attribute_index)
    data["attribute"].x = torch.ones((n_attrs, 1), dtype=torch.float)
    n_params = len(parameter_index)
    data["parameter"].x = torch.ones((n_params, 1), dtype=torch.float)
    n_imports = len(import_index)
    data["import"].x = torch.ones((n_imports, 1), dtype=torch.float)

    edges: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}

    def add_edge(src_type, rel, dst_type, src_idx, dst_idx):
        key = (src_type, rel, dst_type)
        edges.setdefault(key, []).append((src_idx, dst_idx))

    for ci, cls in enumerate(mod.classes):
        parent = _resolve_parent_node(cls.parent_qualname)
        if parent is not None:
            add_edge(parent[0], "contains", "class", parent[1], ci)
        else:
            add_edge("module", "contains", "class", 0, ci)
    for fi, fnode in enumerate(function_nodes):
        parent = _resolve_parent_node(fnode.fn.parent_qualname)
        if parent is not None:
            add_edge(parent[0], "contains", "function", parent[1], fi)
        else:
            add_edge("module", "contains", "function", 0, fi)
    for name, ii in import_index.items():
        add_edge("module", "imports", "import", 0, ii)

    for m in method_like:
        add_edge("class", "contains", "method", m.owner_class_idx, m.index)
        add_edge("method", "belongs_to", "class", m.index, m.owner_class_idx)
    for (ci, attr), ai in attribute_index.items():
        add_edge("class", "contains", "attribute", ci, ai)

    for (owner_type, owner_idx, pname), pi in parameter_index.items():
        add_edge(owner_type, "contains", "parameter", owner_idx, pi)

    for ci, cls in enumerate(mod.classes):
        for base in cls.bases:
            if base in class_index:
                add_edge("class", "inherits", "class", ci, class_index[base])

    def _resolve_call_target(caller: _MethodLikeNode, callee_repr: str, receiver: Optional[str]):
        method_name = callee_repr.rsplit(".", 1)[-1]
        if receiver == "self" and caller.owner_class_idx is not None:
            owner_cls = mod.classes[caller.owner_class_idx]
            for other in owner_cls.methods:
                if other.name == method_name:
                    return "method", method_index_by_qualname.get(other.qualname)
            return None
        if receiver is None:
            # Prefer a closure defined directly inside the caller's own
            # scope (correct Python name resolution -- a nested function is
            # only visible by bare name within its own enclosing def) over
            # a same-named top-level function elsewhere in the file.
            local_key = (caller.fn.qualname, method_name)
            if local_key in nested_function_index:
                return "function", nested_function_index[local_key]
            if method_name in top_level_function_index_by_name:
                return "function", top_level_function_index_by_name[method_name]
            return None
        return None

    all_method_like = method_like + function_nodes
    for n in all_method_like:
        for c in n.fn.calls:
            target = _resolve_call_target(n, c.callee, c.receiver)
            if target is not None:
                dst_type, dst_idx = target
                if dst_idx is not None:
                    add_edge(n.node_type, "calls", dst_type, n.index, dst_idx)

    for n in all_method_like:
        owner_ci = n.owner_class_idx
        param_names = {p for p in n.fn.params if p not in ("self", "cls")}
        for a in n.fn.attr_accesses:
            if a.receiver == "self" and owner_ci is not None:
                key = (owner_ci, a.attr)
                if key in attribute_index:
                    add_edge(n.node_type, "uses", "attribute", n.index, attribute_index[key])
            elif a.receiver in param_names:
                pkey = (n.node_type, n.index, a.receiver)
                if pkey in parameter_index:
                    add_edge(n.node_type, "accesses", "parameter", n.index, parameter_index[pkey])

    for key, pairs in edges.items():
        src_type, rel, dst_type = key
        if not pairs:
            continue
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        data[src_type, rel, dst_type].edge_index = torch.stack([src, dst], dim=0)

    stats = {
        "n_module": 1, "n_class": len(mod.classes), "n_method": method_count,
        "n_function": len(function_nodes), "n_attribute": n_attrs,
        "n_parameter": n_params, "n_import": n_imports,
        "n_edge_types": len(edges), "n_edges": sum(len(v) for v in edges.values()),
    }
    return GraphBuildResult(data=data, stats=stats)
