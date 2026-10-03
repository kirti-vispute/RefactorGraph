# -*- coding: utf-8 -*-
"""Display names (and, where genuinely available, source line ranges) for
every node type produced by graph_builder.build_hetero_graph, in the EXACT
construction order that function uses -- so index i in a node type's
feature tensor maps to records[node_type][i].

The name-only ordering here is a standalone copy of the node-index-to-name
logic already verified by
tests/test_explain.py::test_build_name_index_matches_construction_order
(scripts/explain.py::build_name_index) -- duplicated rather than imported
so scripts/explain.py (Phase 12 explainability tooling) stays untouched
while the same verified ordering is reused for the backend API (Phase 17).
Any consumer needing node identity from a built graph should use this
module rather than re-deriving node order independently.
"""
from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from ml.preprocessing.ast_parser import ModuleInfo


class NodeRecord(TypedDict):
    name: str
    lineno: Optional[int]
    end_lineno: Optional[int]


def build_node_records(mod: ModuleInfo, relpath: str) -> Dict[str, List[NodeRecord]]:
    """module/attribute/parameter/import have no source-text unit of their
    own (see ml/graph/graph_builder.py docstring) so lineno/end_lineno are
    None for those -- never fabricated."""
    records: Dict[str, List[NodeRecord]] = {
        "module": [{"name": relpath, "lineno": 1 if mod.loc else None, "end_lineno": mod.loc or None}],
        "class": [{"name": c.name, "lineno": c.lineno, "end_lineno": c.end_lineno} for c in mod.classes],
        "method": [
            {"name": fn.qualname, "lineno": fn.lineno, "end_lineno": fn.end_lineno}
            for c in mod.classes for fn in c.methods
        ],
        "function": [{"name": fn.name, "lineno": fn.lineno, "end_lineno": fn.end_lineno} for fn in mod.functions],
    }

    attrs: List[NodeRecord] = []
    for cls in mod.classes:
        for attr in sorted(cls.class_attrs):
            attrs.append({"name": f"{cls.name}.{attr}", "lineno": None, "end_lineno": None})
    records["attribute"] = attrs

    params: List[NodeRecord] = []
    for cls in mod.classes:
        for fn in cls.methods:
            for p in fn.params:
                if p in ("self", "cls"):
                    continue
                params.append({"name": f"{fn.qualname}.{p}", "lineno": None, "end_lineno": None})
    for fn in mod.functions:
        for p in fn.params:
            params.append({"name": f"{fn.name}.{p}", "lineno": None, "end_lineno": None})
    records["parameter"] = params

    imports: List[NodeRecord] = []
    seen = set()
    for imp in mod.imports:
        for name in imp.names:
            if name not in seen:
                seen.add(name)
                imports.append({"name": name, "lineno": None, "end_lineno": None})
    records["import"] = imports

    return records


def build_name_index(mod: ModuleInfo, relpath: str) -> Dict[str, List[str]]:
    records = build_node_records(mod, relpath)
    return {node_type: [r["name"] for r in recs] for node_type, recs in records.items()}
