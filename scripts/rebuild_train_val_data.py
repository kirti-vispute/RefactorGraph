# -*- coding: utf-8 -*-
"""Robustness-audit fix rebuild (docs/robustness_audit_phase_a.md).

Re-runs the flat-dataset extraction (scripts/build_dataset.py) and
structural-graph construction (scripts/build_graphs.py) using the corrected
ml/preprocessing/ast_parser.py + ml/graph/graph_builder.py (nested
function/class representation, scoped call resolution, the
external_access_count double-count fix) -- for TRAIN and VAL ONLY.

TEST IS DELIBERATELY EXCLUDED. Per the project's frozen-TEST rule, TEST
(data/raw/test, data/processed/graphs/test, data/processed/graphs_hybrid/test)
is not read, re-parsed, re-labeled, or re-built here. This means:

  - Label thresholds are still derived from TRAIN only (unchanged
    methodology), now fed by the corrected metrics.
  - Cross-split duplicate detection here only guards TRAIN <-> VAL. It does
    NOT re-check TRAIN/VAL against TEST's existing (frozen, unchanged)
    content -- that full three-way check will be re-run, once, at the final
    TEST evaluation step, exactly mirroring how scripts/phase16_final_test_eval.py
    already treats TEST as a one-time final touch.

Output (separate from the original full-corpus reports so the historical
TEST-inclusive baseline in docs/dataset_report.md / docs/graph_validation_report.md
is preserved, not overwritten):
  data/processed/{train,val}_methods.jsonl, {train,val}_classes.jsonl (overwritten)
  data/processed/graphs/{train,val}/**/*.pt (overwritten)
  configs/label_thresholds.json (overwritten -- build_graphs-style scripts
    downstream must use this new file)
  docs/dataset_report_train_val_rebuild.md
  docs/graph_validation_report_train_val_rebuild.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.graph.graph_builder import build_hetero_graph
from ml.preprocessing.label_rules import (
    LabelThresholds,
    collect_corpus_stats,
    is_feature_envy,
    is_god_class,
    is_long_method,
)
from scripts.build_dataset import dedup_cross_split, extract_units, parse_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
GRAPH_DIR = ROOT / "data" / "processed" / "graphs"
DOCS_DIR = ROOT / "docs"

SPLITS = ("train", "val")  # TEST intentionally excluded -- see module docstring


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    all_method_units, all_class_units = [], []
    all_modules_by_split = {}
    file_counts, parse_error_summary = {}, {}

    for split in SPLITS:
        print(f"[parse] {split} ...")
        modules, parse_errors, file_count = parse_split(split)
        file_counts[split] = file_count
        parse_error_summary[split] = parse_errors
        all_modules_by_split[split] = modules
        method_units, class_units = extract_units(split, modules)
        all_method_units.extend(method_units)
        all_class_units.extend(class_units)
        print(f"[parse] {split}: {file_count} files, {len(modules)} parsed ok, "
              f"{len(parse_errors)} parse errors, {len(method_units)} methods, {len(class_units)} classes")

    method_units, m_cross, m_within = dedup_cross_split(all_method_units)
    class_units, c_cross, c_within = dedup_cross_split(all_class_units)
    print(f"[dedup] methods: dropped {m_cross} cross-split (train<->val), {m_within} within-split duplicates")
    print(f"[dedup] classes: dropped {c_cross} cross-split (train<->val), {c_within} within-split duplicates")

    train_mods = [mod for _repo, _relpath, mod in all_modules_by_split["train"]]
    method_statement_counts, class_method_counts, class_locs = collect_corpus_stats(train_mods)
    thresholds = LabelThresholds.from_corpus(method_statement_counts, class_method_counts, class_locs)
    print(f"[thresholds] {thresholds}")

    thresholds_path = ROOT / "configs" / "label_thresholds.json"
    thresholds_path.write_text(
        json.dumps({
            "long_method_statements": thresholds.long_method_statements,
            "god_class_method_count": thresholds.god_class_method_count,
            "god_class_loc": thresholds.god_class_loc,
            "feature_envy_min_external_calls": thresholds.feature_envy_min_external_calls,
            "god_class_min_fields": thresholds.god_class_min_fields,
            "god_class_min_methods_for_field_branch": thresholds.god_class_min_methods_for_field_branch,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[thresholds] saved to {thresholds_path}")

    for u in method_units:
        fn, mod = u["_fn"], u["_mod"]
        u["long_method"] = is_long_method(fn, mod, thresholds)
        u["feature_envy"] = is_feature_envy(fn, mod, thresholds)
        del u["_fn"]
        del u["_mod"]
    for u in class_units:
        u["god_class"] = is_god_class(u["_cls"], thresholds)
        del u["_cls"]

    by_split_methods, by_split_classes = defaultdict(list), defaultdict(list)
    for u in method_units:
        by_split_methods[u["split"]].append(u)
    for u in class_units:
        by_split_classes[u["split"]].append(u)

    label_counts = {}
    for split in SPLITS:
        ms, cs = by_split_methods[split], by_split_classes[split]
        lm = sum(1 for u in ms if u["long_method"])
        fe = sum(1 for u in ms if u["feature_envy"])
        gc = sum(1 for u in cs if u["god_class"])
        label_counts[split] = {
            "methods_total": len(ms), "long_method": lm, "feature_envy": fe,
            "classes_total": len(cs), "god_class": gc,
        }
        with open(PROCESSED_DIR / f"{split}_methods.jsonl", "w", encoding="utf-8") as f:
            for u in ms:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        with open(PROCESSED_DIR / f"{split}_classes.jsonl", "w", encoding="utf-8") as f:
            for u in cs:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")

    report = ["# Dataset Report -- Post-Fix Rebuild (TRAIN/VAL only, TEST excluded)\n"]
    report.append(
        "Rebuilt after the robustness-audit parser/graph fixes "
        "(docs/robustness_audit_phase_a.md): nested function/class "
        "representation, scoped call resolution, external_access_count "
        "double-count fix. TEST deliberately not touched -- see this "
        "script's module docstring. Compare against docs/dataset_report.md "
        "(the original, TEST-inclusive baseline) for what changed.\n"
    )
    report.append("## Files parsed\n")
    for split in SPLITS:
        report.append(f"- {split}: {file_counts[split]} .py files, {len(parse_error_summary[split])} parse errors")
    report.append("\n## Deduplication (train<->val only -- see module docstring)\n")
    report.append(f"- Methods: {m_cross} dropped (cross-split collision), {m_within} dropped (within-split duplicate)")
    report.append(f"- Classes: {c_cross} dropped (cross-split collision), {c_within} dropped (within-split duplicate)")
    report.append("\n## Label thresholds (derived from TRAIN split only, 90th percentile, floors applied)\n")
    report.append(f"- Long Method: statement_count >= {thresholds.long_method_statements}")
    report.append(
        f"- God/Large Class: (method_count >= {thresholds.god_class_method_count} AND LOC >= {thresholds.god_class_loc}) "
        f"OR (LOC >= {thresholds.god_class_loc} AND field_count >= {thresholds.god_class_min_fields} "
        f"AND method_count >= {thresholds.god_class_min_methods_for_field_branch})  "
        "[see docs/godclass_formula_revision.md]"
    )
    report.append(f"- Feature Envy: dominant external receiver access count >= {thresholds.feature_envy_min_external_calls} AND > self access count")
    report.append("\n## Label counts per split\n")
    for split in SPLITS:
        lc = label_counts[split]
        report.append(
            f"- {split}: {lc['methods_total']} methods "
            f"(long_method={lc['long_method']}, feature_envy={lc['feature_envy']}); "
            f"{lc['classes_total']} classes (god_class={lc['god_class']})"
        )
    (DOCS_DIR / "dataset_report_train_val_rebuild.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    # --- structural graphs (train/val only) ---
    print("\n[graphs] building structural graphs for train/val ...")
    agg_report = ["# Graph Construction Validation Report -- Post-Fix Rebuild (TRAIN/VAL only, TEST excluded)\n"]
    agg_report.append(f"Thresholds used: {thresholds}\n")
    failures = []
    for split in SPLITS:
        n_graphs = 0
        node_totals, edge_totals, edge_type_presence = Counter(), Counter(), Counter()
        for repo, relpath, mod in all_modules_by_split[split]:
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
        print(f"[graphs] {split}: {n_graphs} graphs saved")
        agg_report.append(f"\n## {split}\n")
        agg_report.append(f"- graphs built: {n_graphs}")
        for k in ("n_module", "n_class", "n_method", "n_function", "n_attribute", "n_parameter", "n_import"):
            total = node_totals[k]
            agg_report.append(f"- total {k}: {total} (avg {total / n_graphs if n_graphs else 0:.2f}/graph)")
        agg_report.append(f"- total edges: {node_totals['n_edges']} (avg {node_totals['n_edges']/n_graphs if n_graphs else 0:.2f}/graph)")
        agg_report.append("- edge types present (count of graphs containing at least one edge of this type):")
        for et, cnt in sorted(edge_type_presence.items(), key=lambda x: -x[1]):
            agg_report.append(f"  - {et}: {cnt} graphs, {edge_totals[et]} total edges")

    if failures:
        agg_report.append(f"\n## Failures ({len(failures)})\n")
        for split, repo, relpath, err in failures[:50]:
            agg_report.append(f"- {split}/{repo}/{relpath}: {err}")
    else:
        agg_report.append("\n## Failures\n\nNone.\n")

    (DOCS_DIR / "graph_validation_report_train_val_rebuild.md").write_text("\n".join(agg_report), encoding="utf-8")
    print("\n".join(agg_report))
    print(f"\nDone. Graphs under {GRAPH_DIR}/{{train,val}}, reports in {DOCS_DIR}. TEST untouched.")


if __name__ == "__main__":
    main()
