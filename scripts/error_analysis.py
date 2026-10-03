# -*- coding: utf-8 -*-
"""Phase 11: error analysis on VAL predictions from the best model so far
(the Phase 10 tuned hybrid, models/hybrid_tuned/). Re-parses every VAL
source file with the same iteration order ml/graph/graph_builder.py uses
(mod.classes enumerate order for "class"; mod.classes -> cls.methods order
for "method"; mod.functions order for "function"), so each node's model
prediction can be attributed back to a concrete (repo, file, qualname,
lineno) plus its raw structural metrics (statement_count / method_count /
loc / dominant_external_count / self_access_count) from ml/preprocessing/
metrics.py -- the same metrics label_rules.py thresholds against.

Only the VAL split is loaded (both the hybrid graphs used for inference and
the raw source re-parsed for names/metrics). TEST is never referenced by
this script.

Inference is run batched (batch_size=64, shuffle=False -- identical to the
val_loader in scripts/train_hybrid_baseline.py / tune_hybrid_optuna.py) and
NOT one file at a time: PyTorch's matmul/softmax kernels are not exactly
associative across different batch shapes, so a lone GATConv forward pass
over a single small graph produces logits that differ from the same graph's
forward pass when batched alongside others by a few units in the last
float32 bits -- close, but occasionally enough to flip a prediction that
sits right at the model's own 0.5 boundary. Batching identically to Phase 10
makes this script's confusion counts bit-reproduce the already-published
Phase 10 report instead of silently publishing a second, slightly different
"official" number for the same model. Verified against Phase 10:
long_method tp=989 fp=229 fn=173 tn=13411 (n=14802) match exactly.

Output: docs/error_analysis_report.md, models/hybrid_tuned/errors_val.json
(full per-node FP/FN list, for the Phase 12 explainability step to pick
concrete examples from without re-deriving this mapping).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader

from ml.models.gat_baseline import HeteroGAT
from ml.preprocessing.ast_parser import parse_file
from ml.preprocessing.metrics import class_metrics, function_metrics
from scripts.build_dataset import RAW_DIR
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS

ROOT = Path(__file__).resolve().parents[1]
HYBRID_GRAPH_DIR = ROOT / "data" / "processed" / "graphs_hybrid"
MODEL_DIR = ROOT / "models" / "hybrid_tuned"
DOCS_DIR = ROOT / "docs"
SPLIT = "val"
TOP_K = 15
BATCH_SIZE = 64


def iter_val_modules():
    """(repo, relpath, mod) for every VAL hybrid graph file, iterated in
    the EXACT same order as scripts/train_hybrid_baseline.py's load_split
    (sorted rglob over the hybrid graph directory) -- not parse_split's
    order, which walks the raw-source directory instead and does not sort
    within a repo. Same graph order -> same DataLoader batch composition
    -> bit-identical predictions to the officially reported Phase 10 eval
    (see module docstring)."""
    for p in sorted((HYBRID_GRAPH_DIR / SPLIT).rglob("*.pt")):
        rel = p.relative_to(HYBRID_GRAPH_DIR / SPLIT)
        repo = rel.parts[0]
        relpath = "/".join(rel.parts[1:])[: -len(".pt")]
        mod = parse_file(RAW_DIR / SPLIT / repo / relpath)
        if not mod.ok:
            print(f"[warn] re-parse failed for {repo}/{relpath}: {mod.parse_error}")
            continue
        yield repo, relpath, mod


def load_model() -> HeteroGAT:
    params = json.loads((MODEL_DIR / "best_params.json").read_text())
    model = HeteroGAT(
        hidden_dim=params["hidden_dim"], heads=params["heads"], dropout=params["dropout"],
        node_feature_dims=HYBRID_NODE_DIMS,
    )
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", weights_only=True))
    model.eval()
    return model


def apply_norm(data, norm_stats: dict):
    for nt, (mean, std) in norm_stats.items():
        if data[nt].x.shape[0] > 0:
            data[nt].x = (data[nt].x - mean) / std


@torch.no_grad()
def predict_file(model: HeteroGAT, data) -> dict:
    """Single-graph forward pass. Used by the unit test (small synthetic
    graph, no batching-precision concern there); the real pipeline uses
    predict_batch below so its numbers bit-match the batched Phase 10 eval."""
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    h = model(data.x_dict, edge_index_dict)
    out = {}
    if "method" in h and data["method"].x.shape[0] > 0:
        out["method_lm"] = torch.sigmoid(model.predict_long_method(h, "method")).tolist()
        out["method_fe"] = torch.sigmoid(model.predict_feature_envy(h)).tolist()
    if "function" in h and data["function"].x.shape[0] > 0:
        out["function_lm"] = torch.sigmoid(model.predict_long_method(h, "function")).tolist()
    if "class" in h and data["class"].x.shape[0] > 0:
        out["class_gc"] = torch.sigmoid(model.predict_god_class(h)).tolist()
    return out


@torch.no_grad()
def predict_batch(model: HeteroGAT, batch) -> dict:
    """Batched forward pass, sliced back to per-graph prediction lists via
    each node type's batch-assignment vector (torch_geometric concatenates
    node types across graphs in list order, so a per-graph split of that
    concatenation recovers per-file predictions in original order)."""
    edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
    h = model(batch.x_dict, edge_index_dict)
    num_graphs = batch.num_graphs

    def split_by_graph(probs: torch.Tensor, node_type: str) -> list:
        counts = torch.bincount(batch[node_type].batch, minlength=num_graphs).tolist()
        return [t.tolist() for t in torch.split(probs, counts)]

    out = {"method_lm": [[]] * num_graphs, "method_fe": [[]] * num_graphs,
           "function_lm": [[]] * num_graphs, "class_gc": [[]] * num_graphs}
    if "method" in h and batch["method"].x.shape[0] > 0:
        out["method_lm"] = split_by_graph(torch.sigmoid(model.predict_long_method(h, "method")), "method")
        out["method_fe"] = split_by_graph(torch.sigmoid(model.predict_feature_envy(h)), "method")
    if "function" in h and batch["function"].x.shape[0] > 0:
        out["function_lm"] = split_by_graph(torch.sigmoid(model.predict_long_method(h, "function")), "function")
    if "class" in h and batch["class"].x.shape[0] > 0:
        out["class_gc"] = split_by_graph(torch.sigmoid(model.predict_god_class(h)), "class")
    return out


def collect_records(modules, model, norm_stats) -> list:
    """One record per (class|method|function) node with its prediction(s),
    label(s), qualname, lineno, and raw structural metrics. Node order
    within each file matches ml/graph/graph_builder.py exactly, so the i-th
    prediction for a file corresponds to the i-th method_like node built in
    that same nested-loop order.

    Predictions come from batched inference (batch_size=64, shuffle=False)
    to bit-match the officially reported Phase 10 val metrics -- see module
    docstring. Labels are read from each graph's own (unbatched) tensors, so
    batching only affects predictions, never the ground truth being compared
    against."""
    entries = []  # (repo, relpath, mod, data)
    n_skipped = 0
    for repo, relpath, mod in modules:
        graph_path = HYBRID_GRAPH_DIR / SPLIT / repo / (relpath.replace("\\", "/") + ".pt")
        if not graph_path.exists():
            n_skipped += 1
            continue
        data = torch.load(graph_path, weights_only=False)
        apply_norm(data, norm_stats)
        entries.append((repo, relpath, mod, data))
    if n_skipped:
        print(f"[warn] {n_skipped} parsed files had no matching hybrid graph, skipped")

    loader = DataLoader([e[3] for e in entries], batch_size=BATCH_SIZE, shuffle=False)
    records = []
    ptr = 0
    for batch in loader:
        preds = predict_batch(model, batch)
        for g in range(batch.num_graphs):
            repo, relpath, mod, data = entries[ptr + g]
            method_idx = 0
            for cls in mod.classes:
                for fn in cls.methods:
                    fm = function_metrics(fn, mod)
                    records.append({
                        "task": "long_method", "node_type": "method",
                        "repo": repo, "file": relpath, "qualname": fn.qualname, "lineno": fn.lineno,
                        "label": int(data["method"].y_long_method[method_idx].item()),
                        "prob": preds["method_lm"][g][method_idx],
                        "metric": {"statement_count": fm.statement_count},
                    })
                    # No eligibility filter here (e.g. classmethod/staticmethod/
                    # __init__) even though label_rules.is_feature_envy excludes
                    # those categories when deriving the label itself: run_eval
                    # (scripts/train_hybrid_baseline.py) scores every "method"
                    # node's feature_envy prediction unconditionally -- for an
                    # ineligible method the label is definitionally 0, but the
                    # model still emits a prediction for it, and that prediction
                    # still counts as a real FP if the model says otherwise.
                    # Filtering those nodes out here would silently drop real
                    # FPs from the count and stop this report's numbers from
                    # matching the officially reported Phase 10 val metrics.
                    records.append({
                        "task": "feature_envy", "node_type": "method",
                        "repo": repo, "file": relpath, "qualname": fn.qualname, "lineno": fn.lineno,
                        "label": int(data["method"].y_feature_envy[method_idx].item()),
                        "prob": preds["method_fe"][g][method_idx],
                        "metric": {
                            "dominant_external_count": fm.dominant_external_count,
                            "self_access_count": fm.self_access_count,
                            "dominant_external_receiver": fm.dominant_external_receiver,
                        },
                    })
                    method_idx += 1

            for fi, fn in enumerate(mod.functions):
                fm = function_metrics(fn, mod)
                records.append({
                    "task": "long_method", "node_type": "function",
                    "repo": repo, "file": relpath, "qualname": fn.name, "lineno": fn.lineno,
                    "label": int(data["function"].y_long_method[fi].item()),
                    "prob": preds["function_lm"][g][fi],
                    "metric": {"statement_count": fm.statement_count},
                })

            for ci, cls in enumerate(mod.classes):
                cm = class_metrics(cls)
                records.append({
                    "task": "god_class", "node_type": "class",
                    "repo": repo, "file": relpath, "qualname": cls.name, "lineno": cls.lineno,
                    "label": int(data["class"].y[ci].item()),
                    "prob": preds["class_gc"][g][ci],
                    "metric": {"method_count": cm.method_count, "loc": cm.loc},
                })
        ptr += batch.num_graphs
    return records


def confusion_counts(recs: list) -> dict:
    tp = sum(1 for r in recs if r["label"] == 1 and r["prob"] >= 0.5)
    fp = sum(1 for r in recs if r["label"] == 0 and r["prob"] >= 0.5)
    fn = sum(1 for r in recs if r["label"] == 1 and r["prob"] < 0.5)
    tn = sum(1 for r in recs if r["label"] == 0 and r["prob"] < 0.5)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"n": len(recs), "positives": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def fmt_metric(m: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in m.items())


def render_examples(recs: list, label: int, top_k: int, reverse: bool) -> list:
    pool = [r for r in recs if r["label"] == label]
    pool.sort(key=lambda r: r["prob"], reverse=reverse)
    return pool[:top_k]


def boundary_stats(recs: list, task: str) -> list:
    """Class-specific failure pattern notes computed over the FULL error set
    (not just the top-15 tables), tied to the actual rule each task's label
    was derived from (ml/preprocessing/label_rules.py) -- quantifies
    whether errors cluster at the threshold boundary (expected, harder for
    any model) or violate the labeling rule's own direction entirely
    (a real signal-learning gap, not just borderline uncertainty)."""
    fns = [r for r in recs if r["label"] == 1 and r["prob"] < 0.5]
    fps = [r for r in recs if r["label"] == 0 and r["prob"] >= 0.5]
    notes = []

    if task == "long_method":
        near = sum(1 for r in fns if r["metric"]["statement_count"] < 20)
        far = sum(1 for r in fns if r["metric"]["statement_count"] >= 25)
        notes.append(
            f"Of {len(fns)} false negatives, {near} have statement_count < 20 "
            f"(within 5 statements of the >=15 threshold -- a near-boundary miss) "
            f"vs {far} with statement_count >= 25 (comfortably over threshold, a "
            f"clearer miss, not just a close call)."
        )
        short_fp = sum(1 for r in fps if r["metric"]["statement_count"] < 10)
        notes.append(
            f"Of {len(fps)} false positives, {short_fp} have statement_count < 10 "
            f"-- well under the >=15 threshold, so these are not near-miss "
            f"boundary cases but methods the model flagged as long despite a "
            f"short body, likely picking up on graph-structural context (e.g. "
            f"calling into large surrounding code) rather than the method's own length."
        )
    elif task == "feature_envy":
        rule_violating_fp = sum(
            1 for r in fps if r["metric"]["dominant_external_count"] <= r["metric"]["self_access_count"]
        )
        notes.append(
            f"Of {len(fps)} false positives, {rule_violating_fp} have "
            f"dominant_external_count <= self_access_count -- meaning the model "
            f"flagged them as feature envy even though they fail the label "
            f"rule's own direction (external access must exceed self access). "
            f"This is a genuine signal-learning gap, not threshold noise."
        )
        # Every FN already satisfies the label rule by construction (label=1
        # requires it) -- checking that again would be tautological. The
        # real question is whether misses cluster right at the rule's floor
        # (feature_envy_min_external_calls=3) or are comfortably above it.
        at_floor_fn = sum(1 for r in fns if r["metric"]["dominant_external_count"] == 3)
        well_above_fn = sum(1 for r in fns if r["metric"]["dominant_external_count"] >= 5)
        notes.append(
            f"Of {len(fns)} false negatives, {at_floor_fn} sit right at the "
            f"label rule's floor (dominant_external_count==3, the minimum to "
            f"qualify) vs {well_above_fn} with dominant_external_count >= 5 -- "
            f"feature_envy remains the hardest task (rarest class, "
            f"{sum(1 for r in recs if r['label'] == 1)} positives in {len(recs)})."
        )
    elif task == "god_class":
        one_sided_fp = sum(
            1 for r in fps
            if (r["metric"]["method_count"] < 10) != (r["metric"]["loc"] < 192)
        )
        notes.append(
            f"Of {len(fps)} false positives, {one_sided_fp} satisfy only ONE of "
            f"the label rule's two AND-conditions (method_count>=10, loc>=192) "
            f"-- the model appears to respond to the two structural signals "
            f"somewhat independently rather than strictly requiring both, "
            f"unlike the rule that generated the label."
        )
        near_fn = sum(
            1 for r in fns
            if r["metric"]["method_count"] < 15 and r["metric"]["loc"] < 250
        )
        notes.append(
            f"Of {len(fns)} false negatives, {near_fn} are within a modest margin "
            f"of both thresholds (method_count<15, loc<250) -- near-boundary "
            f"misses rather than large, unambiguous god classes being missed outright."
        )

    return notes


def build_report(all_records: dict) -> str:
    lines = [
        "# Error Analysis Report (Phase 11 -- validation split only)",
        "",
        "Model: Phase 10 tuned hybrid (models/hybrid_tuned/, heads=4 hidden_dim=128 "
        "dropout=0.103). Predictions re-derived per VAL source file by re-parsing "
        "with ml.preprocessing.ast_parser in the same node order "
        "ml/graph/graph_builder.py uses, so every prediction is attributed back to "
        "a concrete repo/file/qualname/lineno. TEST split is never loaded by this "
        "script.",
        "",
        "False positive (FP) = predicted smelly, label says not smelly. "
        "False negative (FN) = predicted not smelly, label says smelly. "
        "\"Hardest\" FPs = highest-confidence wrong positives (prob closest to "
        "1.0 while label=0). \"Hardest\" FNs = most confidently missed positives "
        "(prob closest to 0.0 while label=1) -- these are true smells the model "
        "was most wrong about, not just borderline 0.5 cases.",
        "",
    ]

    for task in ("long_method", "feature_envy", "god_class"):
        recs = all_records[task]
        cc = confusion_counts(recs)
        lines.append(f"## {task}")
        lines.append("")
        lines.append(
            f"- n={cc['n']} positives={cc['positives']} precision={cc['precision']:.3f} "
            f"recall={cc['recall']:.3f} f1={cc['f1']:.3f} "
            f"(tp={cc['tp']} fp={cc['fp']} fn={cc['fn']} tn={cc['tn']})"
        )
        lines.append("")

        hardest_fp = render_examples(recs, label=0, top_k=TOP_K, reverse=True)
        hardest_fn = render_examples(recs, label=1, top_k=TOP_K, reverse=False)

        lines.append(f"### Hardest false positives (top {len(hardest_fp)} by confidence)")
        lines.append("")
        lines.append("| repo | file | qualname | line | prob | metrics |")
        lines.append("|---|---|---|---|---|---|")
        for r in hardest_fp:
            lines.append(
                f"| {r['repo']} | {r['file']} | {r['qualname']} | {r['lineno']} | "
                f"{r['prob']:.3f} | {fmt_metric(r['metric'])} |"
            )
        lines.append("")

        lines.append(f"### Hardest false negatives (top {len(hardest_fn)} by confidence)")
        lines.append("")
        lines.append("| repo | file | qualname | line | prob | metrics |")
        lines.append("|---|---|---|---|---|---|")
        for r in hardest_fn:
            lines.append(
                f"| {r['repo']} | {r['file']} | {r['qualname']} | {r['lineno']} | "
                f"{r['prob']:.3f} | {fmt_metric(r['metric'])} |"
            )
        lines.append("")

        lines.append("### Class-specific failure pattern")
        lines.append("")
        for note in boundary_stats(recs, task):
            lines.append(f"- {note}")
        lines.append("")

        errs = [r for r in recs if (r["label"] == 1) != (r["prob"] >= 0.5)]
        by_repo = {}
        for r in errs:
            by_repo[r["repo"]] = by_repo.get(r["repo"], 0) + 1
        if by_repo:
            ranked = sorted(by_repo.items(), key=lambda kv: kv[1], reverse=True)
            repo_str = ", ".join(f"{repo} ({n})" for repo, n in ranked[:5])
            label = "Errors by repo" if len(ranked) <= 5 else f"Errors by repo (top 5 of {len(ranked)})"
            lines.append(f"- {label}: {repo_str}.")
            lines.append("")

    return "\n".join(lines)


def main():
    print(f"[parse] {SPLIT} ...")
    modules = list(iter_val_modules())
    print(f"[parse] {SPLIT}: {len(modules)} modules")

    print("[load] hybrid_tuned model ...")
    model = load_model()
    raw_norm_stats = torch.load(MODEL_DIR / "norm_stats.pt", weights_only=False)
    norm_stats = {nt: (torch.tensor(m), torch.tensor(s)) for nt, (m, s) in raw_norm_stats.items()}

    print("[predict] running inference per VAL file ...")
    records = collect_records(modules, model, norm_stats)
    by_task = {
        "long_method": [r for r in records if r["task"] == "long_method"],
        "feature_envy": [r for r in records if r["task"] == "feature_envy"],
        "god_class": [r for r in records if r["task"] == "god_class"],
    }
    for task, recs in by_task.items():
        cc = confusion_counts(recs)
        print(f"[check] {task}: precision={cc['precision']:.3f} recall={cc['recall']:.3f} f1={cc['f1']:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "errors_val.json").write_text(json.dumps(records, indent=2))

    report = build_report(by_task)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "error_analysis_report.md").write_text(report, encoding="utf-8")
    print(f"[done] wrote docs/error_analysis_report.md and {MODEL_DIR / 'errors_val.json'}")


if __name__ == "__main__":
    main()
