# -*- coding: utf-8 -*-
"""Error analysis for the CURRENT CANDIDATE checkpoint
(models/hybrid_class_pool_tuned_fixed_data, Design B / class_method_pool=True,
seed=43 -- see docs/godclass_formula_revision.md sections 9-11), addressing
the previously-flagged gap that scripts/error_analysis.py and scripts/explain.py
both hardcode models/hybrid_tuned (a DIFFERENT, older, non-pooled architecture)
-- their explanations were never actually for the model this project has been
evaluating/candidate-selecting since Phase 13's Design B change.

Otherwise identical methodology to scripts/error_analysis.py (same VAL-only
scope, same batched-inference bit-reproducibility discipline, same node
attribution technique) -- see that script's module docstring for the parts
not repeated here. Key differences:
  - MODEL_DIR points at the candidate, not models/hybrid_tuned.
  - load_model() passes class_method_pool=True, edge_types=None to match
    how this checkpoint was actually trained (scripts/promote_god_class_retrain_seed43.py).
  - god_class records also carry field_count (relevant to the revised
    is_god_class formula, docs/godclass_formula_revision.md) and
    boundary_stats() checks against the two-branch OR-rule instead of the
    old single AND-rule.
  - Output goes to docs/error_analysis_report_candidate.md and
    models/hybrid_class_pool_tuned_fixed_data/errors_val.json -- the
    original Phase 11 report/file for hybrid_tuned is untouched.

TEST is never loaded by this script.
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
from ml.preprocessing.label_rules import LabelThresholds
from ml.preprocessing.metrics import class_metrics, function_metrics
from scripts.build_dataset import RAW_DIR
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS

ROOT = Path(__file__).resolve().parents[1]
HYBRID_GRAPH_DIR = ROOT / "data" / "processed" / "graphs_hybrid"
MODEL_DIR = ROOT / "models" / "hybrid_class_pool_tuned_fixed_data"
DOCS_DIR = ROOT / "docs"
SPLIT = "val"
TOP_K = 15
BATCH_SIZE = 64

# Same floors as the revised rule -- for the boundary_stats note only, not
# used to derive labels here (labels already come from the graph's own y).
_TH = LabelThresholds(
    long_method_statements=15, god_class_method_count=10, god_class_loc=192,
    god_class_min_fields=1, god_class_min_methods_for_field_branch=3,
)


def iter_val_modules():
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
        node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
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
    entries = []
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
                    "metric": {"method_count": cm.method_count, "loc": cm.loc, "field_count": cm.field_count},
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
    fns = [r for r in recs if r["label"] == 1 and r["prob"] < 0.5]
    fps = [r for r in recs if r["label"] == 0 and r["prob"] >= 0.5]
    notes = []

    if task == "long_method":
        near = sum(1 for r in fns if r["metric"]["statement_count"] < 20)
        far = sum(1 for r in fns if r["metric"]["statement_count"] >= 25)
        notes.append(
            f"Of {len(fns)} false negatives, {near} have statement_count < 20 "
            f"(near-boundary miss) vs {far} with statement_count >= 25 (comfortably "
            f"over threshold, a clearer miss)."
        )
        short_fp = sum(1 for r in fps if r["metric"]["statement_count"] < 10)
        notes.append(
            f"Of {len(fps)} false positives, {short_fp} have statement_count < 10 "
            f"-- not near-miss boundary cases."
        )
    elif task == "feature_envy":
        rule_violating_fp = sum(
            1 for r in fps if r["metric"]["dominant_external_count"] <= r["metric"]["self_access_count"]
        )
        notes.append(
            f"Of {len(fps)} false positives, {rule_violating_fp} have "
            f"dominant_external_count <= self_access_count -- flagged despite failing "
            f"the label rule's own direction."
        )
        at_floor_fn = sum(1 for r in fns if r["metric"]["dominant_external_count"] == 3)
        well_above_fn = sum(1 for r in fns if r["metric"]["dominant_external_count"] >= 5)
        notes.append(
            f"Of {len(fns)} false negatives, {at_floor_fn} sit right at the rule's "
            f"floor (dominant_external_count==3) vs {well_above_fn} with "
            f"dominant_external_count >= 5 -- still the hardest task "
            f"({sum(1 for r in recs if r['label'] == 1)} positives in {len(recs)})."
        )
    elif task == "god_class":
        # Revised OR-rule (docs/godclass_formula_revision.md): size branch
        # (method_count>=10 AND loc>=192) OR field branch (loc>=192 AND
        # field_count>=1 AND method_count>=3). Categorize each FP by which
        # branch it satisfies, if any -- a FP satisfying NEITHER branch is
        # a genuine signal-learning gap; one satisfying the field branch
        # only is the model picking up the exact structural shape the
        # formula revision targeted.
        size_only, field_only, neither, both = 0, 0, 0, 0
        for r in fps:
            mc, loc, fc = r["metric"]["method_count"], r["metric"]["loc"], r["metric"]["field_count"]
            size_branch = mc >= _TH.god_class_method_count and loc >= _TH.god_class_loc
            field_branch = loc >= _TH.god_class_loc and fc >= _TH.god_class_min_fields and mc >= _TH.god_class_min_methods_for_field_branch
            if size_branch and field_branch:
                both += 1
            elif size_branch:
                size_only += 1
            elif field_branch:
                field_only += 1
            else:
                neither += 1
        notes.append(
            f"Of {len(fps)} false positives: {both} satisfy both rule branches, "
            f"{size_only} the size branch only, {field_only} the field branch only, "
            f"{neither} satisfy NEITHER branch (a genuine signal-learning gap, not "
            f"just a formula-boundary artifact)."
        )
        near_fn = sum(
            1 for r in fns
            if r["metric"]["method_count"] < 15 and r["metric"]["loc"] < 250
        )
        notes.append(
            f"Of {len(fns)} false negatives, {near_fn} are within a modest margin "
            f"of both thresholds (method_count<15, loc<250) -- near-boundary misses."
        )

    return notes


def build_report(all_records: dict) -> str:
    lines = [
        "# Error Analysis Report -- CANDIDATE checkpoint (validation split only)",
        "",
        "Model: candidate (models/hybrid_class_pool_tuned_fixed_data/, Design B "
        "class_method_pool=True, seed=43 -- see docs/godclass_formula_revision.md "
        "sections 9-11 for how this checkpoint was selected). Same methodology as "
        "the original Phase 11 report (docs/error_analysis_report.md), but that "
        "report was for models/hybrid_tuned, a different (non-pooled) architecture "
        "-- this is the first error analysis actually run against the model this "
        "project has been evaluating since the Design B change. TEST split is "
        "never loaded by this script.",
        "",
        "False positive (FP) = predicted smelly, label says not smelly. "
        "False negative (FN) = predicted not smelly, label says smelly.",
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

    print("[load] candidate model (Design B, seed=43) ...")
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
    (DOCS_DIR / "error_analysis_report_candidate.md").write_text(report, encoding="utf-8")
    print(f"[done] wrote docs/error_analysis_report_candidate.md and {MODEL_DIR / 'errors_val.json'}")


if __name__ == "__main__":
    main()
