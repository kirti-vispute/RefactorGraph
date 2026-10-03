# -*- coding: utf-8 -*-
"""Phase 4/5: build one heterogeneous graph per source file across all three
splits, save to data/processed/graphs/{split}/{repo}/{relpath}.pt, and
report aggregate structural stats (this is the "graph validation" step —
sanity-check node/edge counts on the real corpus, not just synthetic
snippets in unit tests).

Reuses the exact same train-only threshold computation as build_dataset.py
so graph labels are consistent with the flat-record dataset already built.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.graph.graph_builder import build_hetero_graph
from ml.preprocessing.label_rules import LabelThresholds
from scripts.build_dataset import DOCS_DIR, RAW_DIR, parse_split

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "data" / "processed" / "graphs"


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    all_modules = {}
    for split in ("train", "val", "test"):
        print(f"[parse] {split} ...")
        modules, parse_errors, file_count = parse_split(split)
        all_modules[split] = modules
        print(f"[parse] {split}: {file_count} files, {len(modules)} parsed ok, {len(parse_errors)} errors")

    thresholds_path = ROOT / "configs" / "label_thresholds.json"
    if not thresholds_path.exists():
        raise SystemExit(
            f"{thresholds_path} not found — run scripts/build_dataset.py first so both the "
            "flat dataset and the graph dataset use the same thresholds."
        )
    thresholds = LabelThresholds(**json.loads(thresholds_path.read_text(encoding="utf-8")))
    print(f"[thresholds] loaded from {thresholds_path}: {thresholds}")

    agg = Counter()
    per_split_counts = {}
    failures = []

    for split in ("train", "val", "test"):
        n_graphs = 0
        node_totals = Counter()
        edge_totals = Counter()
        edge_type_presence = Counter()
        for repo, relpath, mod in all_modules[split]:
            try:
                result = build_hetero_graph(mod, thresholds)
            except Exception as e:  # noqa: BLE001 - validation pass, record and continue
                failures.append((split, repo, relpath, f"{type(e).__name__}: {e}"))
                continue
            out_path = GRAPH_DIR / split / repo / (relpath.replace("\\", "/") + ".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result.data, out_path)
            n_graphs += 1
            for k, v in result.stats.items():
                node_totals[k] += v
            for et in result.data.edge_types:
                edge_type_presence[et] += 1
                edge_totals[et] += result.data[et].edge_index.shape[1]
        per_split_counts[split] = (n_graphs, node_totals, edge_totals, edge_type_presence)
        agg[split] = n_graphs
        print(f"[graphs] {split}: {n_graphs} graphs saved, {len(failures)} cumulative failures")

    report = ["# Graph Construction Validation Report\n"]
    report.append(f"Thresholds used: {thresholds}\n")
    for split in ("train", "val", "test"):
        n_graphs, node_totals, edge_totals, edge_presence = per_split_counts[split]
        report.append(f"\n## {split}\n")
        report.append(f"- graphs built: {n_graphs}")
        for k in ("n_module", "n_class", "n_method", "n_function", "n_attribute", "n_parameter", "n_import"):
            total = node_totals[k]
            avg = total / n_graphs if n_graphs else 0
            report.append(f"- total {k}: {total} (avg {avg:.2f}/graph)")
        report.append(f"- total edges: {node_totals['n_edges']} (avg {node_totals['n_edges']/n_graphs if n_graphs else 0:.2f}/graph)")
        report.append("- edge types present (count of graphs containing at least one edge of this type):")
        for et, cnt in sorted(edge_presence.items(), key=lambda x: -x[1]):
            report.append(f"  - {et}: {cnt} graphs, {edge_totals[et]} total edges")

    if failures:
        report.append(f"\n## Failures ({len(failures)})\n")
        for split, repo, relpath, err in failures[:50]:
            report.append(f"- {split}/{repo}/{relpath}: {err}")
    else:
        report.append("\n## Failures\n\nNone — every parsed module was converted to a graph successfully.\n")

    (DOCS_DIR / "graph_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved graphs under {GRAPH_DIR}, report at {DOCS_DIR / 'graph_validation_report.md'}")


if __name__ == "__main__":
    main()
