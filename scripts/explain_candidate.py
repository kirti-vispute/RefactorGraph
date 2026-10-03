# -*- coding: utf-8 -*-
"""GNNExplainer for the CURRENT CANDIDATE checkpoint
(models/hybrid_class_pool_tuned_fixed_data, Design B / class_method_pool=True,
seed=43). Closes the previously-flagged gap that scripts/explain.py's
published explanations (docs/explainability_report.md) were for
models/hybrid_tuned, a different, older, non-pooled architecture -- the
Phase 12 report predates the Design B (class_method_pool) change entirely.

Same GNNExplainer methodology and heterogeneous-graph pruning technique as
scripts/explain.py (see that module's docstring for the general rationale) --
one real, necessary difference, found and fixed here rather than blindly
reused:

**Design-B-specific receptive-field bug, found before running anything**:
scripts/explain.py's receptive_field() does a backward BFS restricted to
edges whose DESTINATION is in the current frontier. But HeteroGAT's
class_method_pool mechanism (ml/models/gat_baseline.py::_pool_methods_to_class)
reads the ("class","contains","method") edge directly -- and that edge's
destination is "method", not "class" (see gat_baseline.py's Phase 13
comment: "class is never a destination of any edge" under the standard
2-layer backbone). So a plain reuse of receptive_field("class") would
prune this edge away for a "class"-node explanation, silently zeroing out
HALF of what predict_god_class actually computes for that node (the
class_method_pool half of the hidden_dim*2 concatenation) -- not just an
incomplete explanation, an outright WRONG single_graph_prob relative to
real inference. class_method_pool_receptive_field() below is the standard
receptive_field() unioned with (a) "method"'s own 2-hop receptive field
(pooling reads h_dict["method"] AFTER its own 2 GAT layers, so method's
own dependencies matter too) and (b) the pooling edge and "method" node
type explicitly, whenever target_type=="class" and the model uses
class_method_pool. Only applies to "class" targets; method/function
targets are unaffected and use the plain receptive_field() unchanged.

Output: docs/explainability_report_candidate.md,
models/hybrid_class_pool_tuned_fixed_data/explanations_val.json. TEST
split is never loaded by this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.algorithm.utils import clear_masks

from ml.models.gat_baseline import EDGE_TYPES, HeteroGAT
from ml.preprocessing.ast_parser import ModuleInfo, parse_file
from scripts.build_dataset import RAW_DIR
from scripts.error_analysis_candidate import HYBRID_GRAPH_DIR, MODEL_DIR, apply_norm, load_model, render_examples
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
SPLIT = "val"
EXAMPLES_PER_BUCKET = 3
GNNEXPLAINER_EPOCHS = 100
TOP_EDGES = 5
POOL_EDGE = ("class", "contains", "method")

STRUCT_FEATURE_NAMES = {
    "method": ["loc", "statement_count", "param_count", "self_access_count", "external_access_count"],
    "function": ["loc", "statement_count", "param_count", "self_access_count", "external_access_count"],
    "class": ["loc", "method_count", "field_count"],
}


class TaskWrapper(nn.Module):
    def __init__(self, base: HeteroGAT, task: str, node_type: str):
        super().__init__()
        self.base = base
        self.task = task
        self.node_type = node_type

    def forward(self, x_dict, edge_index_dict):
        h = self.base(x_dict, edge_index_dict)
        if self.task == "long_method":
            return self.base.predict_long_method(h, self.node_type)
        if self.task == "feature_envy":
            return self.base.predict_feature_envy(h)
        if self.task == "god_class":
            return self.base.predict_god_class(h)
        raise ValueError(self.task)


def receptive_field(target_type: str, edge_types=EDGE_TYPES, hops: int = 2) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    frontier = {target_type}
    all_edges: Set[Tuple[str, str, str]] = set()
    for _ in range(hops):
        hop_edges = [et for et in edge_types if et[2] in frontier]
        all_edges.update(hop_edges)
        frontier = frontier | {et[0] for et in hop_edges}
    return frontier, all_edges


def class_method_pool_receptive_field(
    target_type: str, edge_types, hops: int = 2
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """See module docstring. For non-"class" targets this is identical to
    receptive_field(). For "class" targets, additionally unions in method's
    own receptive field plus the explicit pooling edge, so the pruned graph
    passed to GNNExplainer matches what HeteroGAT.forward() actually
    computes for a class_method_pool=True model."""
    nodes, edges = receptive_field(target_type, edge_types=edge_types, hops=hops)
    if target_type != "class":
        return nodes, edges
    if POOL_EDGE not in edge_types:
        return nodes, edges
    method_nodes, method_edges = receptive_field("method", edge_types=edge_types, hops=hops)
    nodes = nodes | method_nodes | {"method"}
    edges = edges | method_edges | {POOL_EDGE}
    return nodes, edges


def prune_to_receptive_field(data, node_types: Set[str], edge_types: Set[Tuple[str, str, str]]):
    x_dict = {nt: data[nt].x for nt in node_types if data[nt].x.shape[0] > 0}
    edge_index_dict = {
        et: data[et].edge_index
        for et in edge_types
        if et in data.edge_types and et[0] in x_dict and et[2] in x_dict
    }
    return x_dict, edge_index_dict


def find_node_index(mod: ModuleInfo, node_type: str, qualname: str, lineno: int) -> int:
    if node_type == "class":
        items = [(c.name, c.lineno) for c in mod.classes]
    elif node_type == "method":
        items = [(fn.qualname, fn.lineno) for c in mod.classes for fn in c.methods]
    elif node_type == "function":
        items = [(fn.name, fn.lineno) for fn in mod.functions]
    else:
        raise ValueError(node_type)
    for i, (name, ln) in enumerate(items):
        if name == qualname and ln == lineno:
            return i
    for i, (name, ln) in enumerate(items):
        if name == qualname:
            return i
    raise ValueError(f"{node_type} node {qualname!r} (line {lineno}) not found on re-parse")


def build_name_index(mod: ModuleInfo, relpath: str) -> Dict[str, List[str]]:
    names: Dict[str, List[str]] = {
        "module": [relpath],
        "class": [c.name for c in mod.classes],
        "method": [fn.qualname for c in mod.classes for fn in c.methods],
        "function": [fn.name for fn in mod.functions],
    }
    attrs = []
    for cls in mod.classes:
        for attr in sorted(cls.class_attrs):
            attrs.append(f"{cls.name}.{attr}")
    names["attribute"] = attrs

    params = []
    for cls in mod.classes:
        for fn in cls.methods:
            for p in fn.params:
                if p in ("self", "cls"):
                    continue
                params.append(f"{fn.qualname}.{p}")
    for fn in mod.functions:
        for p in fn.params:
            params.append(f"{fn.name}.{p}")
    names["parameter"] = params

    imports, seen = [], set()
    for imp in mod.imports:
        for name in imp.names:
            if name not in seen:
                seen.add(name)
                imports.append(name)
    names["import"] = imports
    return names


def explain_record(record: dict, model: HeteroGAT, norm_stats: dict, explainer_cache: dict) -> dict:
    task, node_type = record["task"], record["node_type"]
    repo, relpath = record["repo"], record["file"]

    mod = parse_file(RAW_DIR / SPLIT / repo / relpath)
    if not mod.ok:
        return {**record, "error": f"re-parse failed: {mod.parse_error}"}

    graph_path = HYBRID_GRAPH_DIR / SPLIT / repo / (relpath.replace("\\", "/") + ".pt")
    data = torch.load(graph_path, weights_only=False)
    apply_norm(data, norm_stats)

    idx = find_node_index(mod, node_type, record["qualname"], record["lineno"])
    # Restrict candidate edges to what the MODEL actually registered a
    # relation for (model.edge_types, i.e. what conv1/conv2 were built
    # with), not just what this file's graph happens to contain. Found
    # necessary here: graph_builder.py emits a (method,belongs_to,class)
    # reverse edge into every saved graph, but this model's own edge_types
    # (plain EDGE_TYPES, no class_method_pool changes that) never
    # registered a GATConv for it -- HeteroConv.forward() silently ignores
    # any edge_index_dict key it has no conv for, so that edge type
    # guarantees zero gradient every time it's included, which is exactly
    # what crashed all 6 god_class examples on the first attempt at this
    # fix (before this line existed). The class_method_pool POOL_EDGE is
    # exempt from this filter -- _pool_methods_to_class reads it directly
    # from edge_index_dict, independent of conv1/conv2's registered set.
    usable_edge_types = [et for et in data.edge_types if et in model.edge_types or et == POOL_EDGE]
    reach_nodes, reach_edges = class_method_pool_receptive_field(node_type, edge_types=usable_edge_types)
    x_sub, ei_sub = prune_to_receptive_field(data, reach_nodes, reach_edges)

    key = (task, node_type)
    if key not in explainer_cache:
        wrapper = TaskWrapper(model, task, node_type)
        explainer_cache[key] = Explainer(
            model=wrapper,
            algorithm=GNNExplainer(epochs=GNNEXPLAINER_EPOCHS),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
        )
    explainer = explainer_cache[key]

    with torch.no_grad():
        logits = explainer.model(x_sub, ei_sub)
    single_prob = torch.sigmoid(logits[idx]).item()

    try:
        explanation = explainer(x_sub, ei_sub, index=idx)
    except ValueError as e:
        # Known heterogeneous-GNNExplainer limitation (see module docstring
        # and scripts/explain.py's docstring): receptive_field()/
        # class_method_pool_receptive_field() prune by TYPE-level 2-hop
        # reachability, but a specific file can have an edge of a reachable
        # type that is not actually on any gradient path to THIS instance
        # (e.g. a nested class elsewhere in the same file connected via
        # (class,contains,class) to a class unrelated to the target method).
        # GNNExplainer hard-errors on that rather than skipping the type.
        # Reported as an explicit per-record failure, not silently retried
        # with a weakened/different explanation that would misrepresent
        # what was actually explained.
        #
        # Also found (second crash, caught before trusting this fix): PyG's
        # GNNExplainer.forward() calls self._clean_model(model) (which
        # clears the edge/node masks it attached to model's conv layers via
        # set_masks) only AFTER _train() returns normally -- no try/finally.
        # An exception inside _train (as above) skips that cleanup, leaving
        # the PREVIOUS graph's edge_mask (wrong shape for the NEXT graph)
        # attached to model.base.conv1/conv2's GATConv layers. The very
        # next call on this same cached explainer -- even a plain no_grad
        # forward pass -- then hits
        # `assert inputs.size(self.node_dim) == edge_mask.size(0)` inside
        # torch_geometric's own explain_message hook. Must clear_masks(model)
        # explicitly here or every subsequent record silently corrupts.
        clear_masks(model)
        return {**record, "error": f"GNNExplainer failed: {e}"}
    names = build_name_index(mod, relpath)

    edges_out = []
    for et in explanation.edge_types:
        mask = explanation[et].edge_mask
        ei = ei_sub[et]
        for e in range(ei.shape[1]):
            s, d = ei[0, e].item(), ei[1, e].item()
            edges_out.append({
                "importance": mask[e].item(),
                "src_type": et[0], "src_name": names[et[0]][s] if s < len(names[et[0]]) else f"{et[0]}#{s}",
                "rel": et[1],
                "dst_type": et[2], "dst_name": names[et[2]][d] if d < len(names[et[2]]) else f"{et[2]}#{d}",
            })
    edges_out.sort(key=lambda e: e["importance"], reverse=True)
    top_edges = edges_out[:TOP_EDGES]

    own_mask = explanation[node_type].node_mask[idx]
    n_struct = len(STRUCT_FEATURE_NAMES[node_type])
    struct_importance = [
        {"feature": name, "importance": own_mask[i].item()}
        for i, name in enumerate(STRUCT_FEATURE_NAMES[node_type])
    ]
    struct_importance.sort(key=lambda f: f["importance"], reverse=True)
    codebert_importance = own_mask[n_struct:].mean().item() if own_mask.numel() > n_struct else 0.0

    return {
        **record,
        "single_graph_prob": single_prob,
        "receptive_field_node_types": sorted(reach_nodes),
        "receptive_field_edge_types": sorted(f"{s}->{r}->{d}" for s, r, d in reach_edges),
        "top_edges": top_edges,
        "struct_feature_importance": struct_importance,
        "codebert_importance_mean": codebert_importance,
    }


def select_examples(errors: List[dict]) -> Dict[str, List[dict]]:
    by_task: Dict[str, List[dict]] = {}
    for r in errors:
        by_task.setdefault(r["task"], []).append(r)
    picked = {}
    for task, recs in by_task.items():
        fp = render_examples(recs, label=0, top_k=EXAMPLES_PER_BUCKET, reverse=True)
        fn = render_examples(recs, label=1, top_k=EXAMPLES_PER_BUCKET, reverse=False)
        picked[task] = fp + fn
    return picked


def render_report(explained: Dict[str, List[dict]]) -> str:
    lines = [
        "# Explainability Report -- CANDIDATE checkpoint (validation split only)",
        "",
        "GNNExplainer applied to the candidate model "
        "(models/hybrid_class_pool_tuned_fixed_data/, Design B "
        "class_method_pool=True, seed=43), on the hardest false "
        "positives/negatives per task from docs/error_analysis_report_candidate.md. "
        "The original Phase 12 report (docs/explainability_report.md) targeted "
        "models/hybrid_tuned, a different, older, non-pooled architecture -- this "
        "is the first GNNExplainer run actually against the model this project has "
        "been evaluating since the Design B change. See scripts/explain_candidate.py "
        "module docstring for a real receptive-field bug (class_method_pool's "
        "pooling edge has method, not class, as its destination) found and fixed "
        "before running this, not blindly inherited from scripts/explain.py. TEST "
        "split is never loaded by this script.",
        "",
        "Each example is explained from a single-graph forward pass; "
        "`single_graph_prob` is reported alongside the batched `prob` from error "
        "analysis rather than assumed equal (floating-point batching "
        "non-associativity) -- a large gap is flagged.",
        "",
    ]
    for task, examples in explained.items():
        lines.append(f"## {task}")
        lines.append("")
        for r in examples:
            if "error" in r:
                lines.append(f"- **{r['repo']}/{r['file']}:{r['qualname']}** -- {r['error']}")
                lines.append("")
                continue
            kind = "false positive" if r["label"] == 0 else "false negative"
            gap = abs(r["single_graph_prob"] - r["prob"])
            gap_note = "" if gap < 0.05 else f" **(gap={gap:.3f}, flagged)**"
            lines.append(
                f"### {r['repo']}/{r['file']}:{r['qualname']} (line {r['lineno']}) -- {kind}"
            )
            lines.append("")
            lines.append(
                f"- label={r['label']} batched_prob={r['prob']:.3f} "
                f"single_graph_prob={r['single_graph_prob']:.3f}{gap_note}"
            )
            lines.append(f"- own raw metrics: {r['metric']}")
            lines.append(
                f"- receptive field: node types {r['receptive_field_node_types']}, "
                f"edge types {r['receptive_field_edge_types']}"
            )
            lines.append("- top contributing edges:")
            for e in r["top_edges"]:
                lines.append(
                    f"  - {e['importance']:.3f}: {e['src_type']}:{e['src_name']} "
                    f"--{e['rel']}--> {e['dst_type']}:{e['dst_name']}"
                )
            if not r["top_edges"]:
                lines.append("  - (none -- pruned graph had no edges reaching this node)")
            lines.append("- own structural feature importance:")
            for f in r["struct_feature_importance"]:
                lines.append(f"  - {f['feature']}: {f['importance']:.3f}")
            lines.append(f"- own CodeBERT embedding importance (mean over 768 dims): {r['codebert_importance_mean']:.3f}")
            lines.append("")
    return "\n".join(lines)


def main():
    print("[load] candidate model (Design B, seed=43) ...")
    model = load_model()
    raw_norm_stats = torch.load(MODEL_DIR / "norm_stats.pt", weights_only=False)
    norm_stats = {nt: (torch.tensor(m), torch.tensor(s)) for nt, (m, s) in raw_norm_stats.items()}

    errors_path = MODEL_DIR / "errors_val.json"
    print(f"[load] {errors_path} ...")
    all_records = json.loads(errors_path.read_text())
    errors = [r for r in all_records if (r["label"] == 1) != (r["prob"] >= 0.5)]
    print(f"[select] {len(errors)} total val errors across all tasks")

    examples = select_examples(errors)
    for task, recs in examples.items():
        print(f"[select] {task}: {len(recs)} examples to explain")

    explainer_cache: dict = {}
    explained: Dict[str, List[dict]] = {}
    for task, recs in examples.items():
        out = []
        for i, r in enumerate(recs):
            print(f"[explain] {task} {i + 1}/{len(recs)}: {r['repo']}/{r['file']}:{r['qualname']}")
            out.append(explain_record(r, model, norm_stats, explainer_cache))
        explained[task] = out

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "explanations_val.json").write_text(json.dumps(explained, indent=2))

    report = render_report(explained)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "explainability_report_candidate.md").write_text(report, encoding="utf-8")
    print(f"[done] wrote docs/explainability_report_candidate.md and {MODEL_DIR / 'explanations_val.json'}")


if __name__ == "__main__":
    main()
