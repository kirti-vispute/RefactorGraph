# -*- coding: utf-8 -*-
"""Phase 12: Explainability (GNNExplainer) for the Phase 10 tuned hybrid model.

Explains concrete predictions on the VAL split only -- TEST is never loaded
by this script. Examples are picked from Phase 11's already-computed error
list (models/hybrid_tuned/errors_val.json): the hardest false positives and
false negatives per task, i.e. the same node-to-name attribution already
verified in scripts/error_analysis.py, reused rather than re-derived.

Why GNNExplainer and not PGExplainer:
torch_geometric.explain.PGExplainer is a *parametric* explainer -- it trains
a small MLP to predict edge masks, and that MLP itself must be trained first
over a batch of instances with its own optimizer/epoch loop (effectively a
second small model with its own train/eval discipline). This PyG version
(2.8) also restricts PGExplainer to a single scalar node index per forward
call, which would force a slow one-index-at-a-time training loop for every
(task, node_type) combination. Given the project's "no fabricated/rushed
results" rule, building and validating that second training pipeline properly
is out of scope for this phase; GNNExplainer is a per-instance optimizer
(no separate training set needed) and is natively supported for
heterogeneous graphs in this PyG version, so it is used as the sole explainer
here. PGExplainer is left as documented future work in the report.

Heterogeneous GNNExplainer pitfall (found and fixed here): calling the
explainer with the FULL x_dict/edge_index_dict of a graph fails with
"Could not compute gradients for node/edge masks of type X" whenever a node
or edge type has no path to the explained node within the model's receptive
field (e.g. an unrelated top-level function in the same file as the method
being explained) -- the mask parameter for that disconnected type gets zero
gradient, which GNNExplainer treats as a hard error rather than skipping it.
receptive_field() below computes, per target node type, exactly the node/edge
types that CAN reach it in a 2-layer HeteroConv stack (matching HeteroGAT's
depth) via backward BFS over (src, rel, dst) edges with dst in the current
frontier; prune_to_receptive_field() then restricts the graph passed to the
explainer to that set (further intersected with what the graph actually has).
This is not just a workaround -- the pruned sets are themselves an
interpretability finding, see docs/explainability_report.md.

Output: docs/explainability_report.md, models/hybrid_tuned/explanations_val.json
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

from ml.models.gat_baseline import EDGE_TYPES, HeteroGAT
from ml.preprocessing.ast_parser import ModuleInfo, parse_file
from scripts.build_dataset import RAW_DIR
from scripts.error_analysis import HYBRID_GRAPH_DIR, MODEL_DIR, apply_norm, load_model, render_examples
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
SPLIT = "val"
EXAMPLES_PER_BUCKET = 3
GNNEXPLAINER_EPOCHS = 100
TOP_EDGES = 5

STRUCT_FEATURE_NAMES = {
    "method": ["loc", "statement_count", "param_count", "self_access_count", "external_access_count"],
    "function": ["loc", "statement_count", "param_count", "self_access_count", "external_access_count"],
    "class": ["loc", "method_count", "field_count"],
}

class TaskWrapper(nn.Module):
    """Wraps HeteroGAT so forward(x_dict, edge_index_dict) returns a single
    raw-logit Tensor for one (task, node_type) -- torch_geometric's Explainer
    requires a plain Tensor output so it can index into its first dim."""

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


def receptive_field(
    target_type: str, edge_types=EDGE_TYPES, hops: int = 2
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """Node/edge types that can influence target_type's final embedding in a
    `hops`-layer HeteroConv stack: backward BFS over (src, rel, dst) with dst
    in the current frontier. Anything outside this set has zero gradient path
    to the target and must be pruned before calling GNNExplainer (see module
    docstring)."""
    frontier = {target_type}
    all_edges: Set[Tuple[str, str, str]] = set()
    for _ in range(hops):
        hop_edges = [et for et in edge_types if et[2] in frontier]
        all_edges.update(hop_edges)
        frontier = frontier | {et[0] for et in hop_edges}
    return frontier, all_edges


def prune_to_receptive_field(data, node_types: Set[str], edge_types: Set[Tuple[str, str, str]]):
    x_dict = {nt: data[nt].x for nt in node_types if data[nt].x.shape[0] > 0}
    edge_index_dict = {
        et: data[et].edge_index
        for et in edge_types
        if et in data.edge_types and et[0] in x_dict and et[2] in x_dict
    }
    return x_dict, edge_index_dict


def find_node_index(mod: ModuleInfo, node_type: str, qualname: str, lineno: int) -> int:
    """Recovers a node's index within its type's array by re-parsing the same
    file with the exact node order ml/graph/graph_builder.py uses (class
    enumerate order for classes; nested class.methods order for methods;
    mod.functions order for functions) -- same technique as
    scripts/error_analysis.py, applied in reverse (name -> index instead of
    index -> name)."""
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
    """Display names for every node type, in the exact construction order
    ml/graph/graph_builder.py uses, so index i in an edge_mask/node_mask
    array maps to names[node_type][i]."""
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
    # Use THIS graph's own edge types, not the model's static full schema:
    # a type can be reachable in principle (EDGE_TYPES) yet absent from this
    # particular file (e.g. no method calls any module-level function here),
    # in which case including it anyway leaves its node-feature encoding
    # computed but never consumed downstream -- GNNExplainer's node_mask
    # parameter for that type then gets zero gradient and hard-errors,
    # exactly like the disconnected-type case receptive_field already
    # guards against, just one level more specific (per-graph, not per-model).
    reach_nodes, reach_edges = receptive_field(node_type, edge_types=list(data.edge_types))
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

    explanation = explainer(x_sub, ei_sub, index=idx)
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
        "# Explainability Report (Phase 12 -- validation split only)",
        "",
        "GNNExplainer (torch_geometric.explain) applied to the Phase 10 tuned "
        "hybrid model (models/hybrid_tuned/), on the hardest false "
        "positives/negatives per task from Phase 11 (docs/error_analysis_report.md). "
        "TEST split is never loaded by this script. See scripts/explain.py module "
        "docstring for why GNNExplainer (not PGExplainer) was used, and for the "
        "heterogeneous-graph receptive-field pruning this required.",
        "",
        "Each example is explained from a single-graph forward pass (GNNExplainer "
        "optimizes a mask per instance), while Phase 11's reported `prob` came from "
        "batch_size=64 inference -- `single_graph_prob` below is usually within "
        "~0.01 of `prob` (floating-point batching non-associativity, see Phase 11) "
        "but is reported explicitly rather than silently assumed equal; a "
        "large gap is flagged.",
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
    print("[load] hybrid_tuned model ...")
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
    (DOCS_DIR / "explainability_report.md").write_text(report, encoding="utf-8")
    print(f"[done] wrote docs/explainability_report.md and {MODEL_DIR / 'explanations_val.json'}")


if __name__ == "__main__":
    main()
