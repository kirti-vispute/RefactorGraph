# -*- coding: utf-8 -*-
"""EXPERIMENT-ONLY rebuild (authorized: docs/feature_envy_generalization_investigation.md
section 10). Rebuilds TRAIN+VAL structural graphs using the ONE-LINE-CHANGED
graph_builder copy (scratch_external_eval/graph_builder_dominant_ext.py:
`dominant_external_count` replaces `external_access_count` in the
method/function feature vector). TEST is never read.

Output goes to a NEW, separate directory
(data/processed/graphs_experiment_fe_dominant/{train,val}) -- the existing
data/processed/graphs/{train,val} (which backs the currently-deployed
candidate's own reproducibility/audit trail) is never touched or
overwritten.

Also verifies, before any training happens:
  - TRAIN repo set and VAL repo set are unchanged from configs/repos.yaml
    and remain disjoint (no repo appears in both).
  - Label thresholds re-derived from this rebuild match the currently
    active configs/label_thresholds.json exactly (is_feature_envy's own
    logic is untouched by this experiment -- only the graph's INPUT
    feature changed, never the label -- so an exact match here is the
    expected, correct outcome, not a coincidence).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from scratch_external_eval.graph_builder_dominant_ext import build_hetero_graph
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
GRAPH_DIR = ROOT / "data" / "processed" / "graphs_experiment_fe_dominant"
DOCS_DIR = ROOT / "docs"

SPLITS = ("train", "val")


def verify_repo_split():
    cfg = yaml.safe_load((ROOT / "configs" / "repos.yaml").read_text(encoding="utf-8"))
    train_repos = {r["name"] for r in cfg["train"]}
    val_repos = {r["name"] for r in cfg["val"]}
    test_repos = {r["name"] for r in cfg["test"]}
    assert train_repos.isdisjoint(val_repos), "TRAIN/VAL repo overlap!"
    assert train_repos.isdisjoint(test_repos), "TRAIN/TEST repo overlap!"
    assert val_repos.isdisjoint(test_repos), "VAL/TEST repo overlap!"
    on_disk_train = {p.name for p in (ROOT / "data" / "raw" / "train").iterdir() if p.is_dir()}
    on_disk_val = {p.name for p in (ROOT / "data" / "raw" / "val").iterdir() if p.is_dir()}
    assert on_disk_train == train_repos, f"TRAIN dir mismatch: {on_disk_train} vs {train_repos}"
    assert on_disk_val == val_repos, f"VAL dir mismatch: {on_disk_val} vs {val_repos}"
    print(f"[verify] TRAIN repos ({len(train_repos)}): {sorted(train_repos)}")
    print(f"[verify] VAL repos ({len(val_repos)}): {sorted(val_repos)}")
    print(f"[verify] TEST repos ({len(test_repos)}) -- NOT read, NOT touched: {sorted(test_repos)}")
    print("[verify] TRAIN/VAL/TEST repo-disjoint: PASS")


def main():
    verify_repo_split()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_method_units, all_class_units = [], []
    all_modules_by_split = {}

    for split in SPLITS:
        print(f"[parse] {split} ...")
        modules, parse_errors, file_count = parse_split(split)
        all_modules_by_split[split] = modules
        method_units, class_units = extract_units(split, modules)
        all_method_units.extend(method_units)
        all_class_units.extend(class_units)
        print(f"[parse] {split}: {file_count} files, {len(modules)} parsed ok, "
              f"{len(parse_errors)} parse errors, {len(method_units)} methods, {len(class_units)} classes")

    method_units, m_cross, m_within = dedup_cross_split(all_method_units)
    class_units, c_cross, c_within = dedup_cross_split(all_class_units)
    print(f"[dedup] methods: dropped {m_cross} cross-split, {m_within} within-split")
    print(f"[dedup] classes: dropped {c_cross} cross-split, {c_within} within-split")

    train_mods = [mod for _repo, _relpath, mod in all_modules_by_split["train"]]
    method_statement_counts, class_method_counts, class_locs = collect_corpus_stats(train_mods)
    thresholds = LabelThresholds.from_corpus(method_statement_counts, class_method_counts, class_locs)
    print(f"[thresholds] {thresholds}")

    current = json.loads((ROOT / "configs" / "label_thresholds.json").read_text(encoding="utf-8"))
    matches = (
        thresholds.long_method_statements == current["long_method_statements"]
        and thresholds.god_class_method_count == current["god_class_method_count"]
        and thresholds.god_class_loc == current["god_class_loc"]
    )
    print(f"[verify] thresholds match currently-active configs/label_thresholds.json: "
          f"{'PASS' if matches else 'FAIL -- ' + str(current)}")
    print("[verify] configs/label_thresholds.json NOT written by this script (read-only check).")

    print("\n[graphs] building structural graphs (experiment feature) for train/val ...")
    failures = []
    label_counts = {}
    for split in SPLITS:
        n_graphs = 0
        fe_count, lm_count, gc_count = 0, 0, 0
        for repo, relpath, mod in all_modules_by_split[split]:
            try:
                result = build_hetero_graph(mod, thresholds)
            except Exception as e:  # noqa: BLE001
                failures.append((split, repo, relpath, f"{type(e).__name__}: {e}"))
                continue
            out_path = GRAPH_DIR / split / repo / (relpath.replace("\\", "/") + ".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result.data, out_path)
            n_graphs += 1
            fe_count += int(result.data["method"].y_feature_envy.sum().item())
            lm_count += int(result.data["method"].y_long_method.sum().item()) + int(result.data["function"].y_long_method.sum().item())
            gc_count += int(result.data["class"].y.sum().item())
        label_counts[split] = {"graphs": n_graphs, "feature_envy": fe_count, "long_method": lm_count, "god_class": gc_count}
        print(f"[graphs] {split}: {n_graphs} graphs, feature_envy={fe_count} long_method={lm_count} god_class={gc_count}")

    if failures:
        print(f"\n[FAILURES] {len(failures)} -- see below")
        for f in failures[:20]:
            print(" ", f)
    else:
        print("\n[graphs] no failures")

    print(f"\nDone. Structural graphs under {GRAPH_DIR}/{{train,val}}. "
          f"data/processed/graphs/{{train,val}} (production) untouched. TEST untouched.")


if __name__ == "__main__":
    main()
