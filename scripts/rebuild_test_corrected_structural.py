# -*- coding: utf-8 -*-
"""ONE-TIME, deliberate, user-authorized TEST structural rebuild
(authorized this session, after `scripts/experiment_fe_dominant_test_eval.py`
found the existing data/processed/graphs_hybrid/test predates two parser
fixes already reflected in TRAIN/VAL -- docs/experiment_fe_dominant_test_eval_STOPPED.md).

Builds TWO parallel TEST structural graph sets, both using the CURRENT
parser (ml/preprocessing/ast_parser.py, metrics.py -- the already-established
fixes, unchanged by this script), differing in EXACTLY one thing:

  - "corrected_original": production ml/graph/graph_builder.py
    (external_access_count feature) -- this is what the TEST structural
    data SHOULD already look like under the current parser, had TEST
    ever been rebuilt like TRAIN/VAL was.
  - "corrected_dominant": the Experiment #1 feature
    (scratch_external_eval/graph_builder_dominant_ext.py,
    dominant_external_count) -- reuses the build already made in
    scripts/experiment_fe_dominant_test_eval.py (same parser, same
    thresholds, so node order/count is guaranteed identical to
    corrected_original by construction; not rebuilt a second time here,
    per the "rebuild TEST exactly once" instruction).

Neither existing data/processed/graphs/test nor
data/processed/graphs_hybrid/test (the original, stale, historical
baseline data) is modified or overwritten -- both new sets go to
separate, clearly-named directories.

Does NOT run CodeBERT augmentation (a separate script) and does NOT run
any evaluation/inference -- stops after producing + verifying the two
structural sets, per explicit instruction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ml.graph.graph_builder import build_hetero_graph as build_hetero_graph_original
from ml.preprocessing.label_rules import LabelThresholds
from scripts.build_dataset import parse_split

ORIGINAL_OUT = ROOT / "data" / "processed" / "graphs_test_corrected_original"
DOMINANT_DIR = ROOT / "data" / "processed" / "graphs_experiment_fe_dominant" / "test"  # already built


def main():
    cfg = json.loads((ROOT / "configs" / "label_thresholds.json").read_text())
    thresholds = LabelThresholds(
        long_method_statements=cfg["long_method_statements"],
        god_class_method_count=cfg["god_class_method_count"],
        god_class_loc=cfg["god_class_loc"],
        feature_envy_min_external_calls=cfg["feature_envy_min_external_calls"],
        god_class_min_fields=cfg["god_class_min_fields"],
        god_class_min_methods_for_field_branch=cfg["god_class_min_methods_for_field_branch"],
    )
    print(f"[thresholds] {thresholds}")

    print("[build] parsing TEST once (data/raw/test) ...")
    modules, parse_errors, file_count = parse_split("test")
    print(f"[build] test: {file_count} files, {len(modules)} parsed ok, {len(parse_errors)} parse errors")

    if not DOMINANT_DIR.exists() or not any(DOMINANT_DIR.rglob("*.pt")):
        raise SystemExit(
            f"[ABORT] {DOMINANT_DIR} missing -- expected the corrected_dominant structural "
            "set already built by scripts/experiment_fe_dominant_test_eval.py's step 1. "
            "Not rebuilding it a second time (rebuild TEST exactly once)."
        )
    n_dominant_existing = len(list(DOMINANT_DIR.rglob("*.pt")))
    print(f"[build] reusing existing corrected_dominant set: {n_dominant_existing} files (not rebuilt)")

    n_graphs, failures = 0, []
    for repo, relpath, mod in modules:
        try:
            result = build_hetero_graph_original(mod, thresholds)
        except Exception as e:  # noqa: BLE001
            failures.append((repo, relpath, f"{type(e).__name__}: {e}"))
            continue
        out_path = ORIGINAL_OUT / repo / (relpath.replace("\\", "/") + ".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result.data, out_path)
        n_graphs += 1
    print(f"[build] corrected_original: {n_graphs} graphs built, {len(failures)} failures")
    if failures:
        for f in failures[:10]:
            print("  ", f)

    # --- verification: corrected_original vs corrected_dominant must be
    # identical everywhere except column 4 of method/function node features ---
    print("\n[verify] comparing corrected_original vs corrected_dominant, file by file ...")
    orig_files = sorted(ORIGINAL_OUT.rglob("*.pt"))
    n_checked, n_col4_only_diff, n_unexpected_diff, n_missing = 0, 0, 0, 0
    unexpected = []
    fe_count_orig, fe_count_dom = 0, 0
    lm_count_orig, lm_count_dom = 0, 0
    gc_count_orig, gc_count_dom = 0, 0
    for op in orig_files:
        rel = op.relative_to(ORIGINAL_OUT)
        dp = DOMINANT_DIR / rel
        if not dp.exists():
            n_missing += 1
            continue
        d_orig = torch.load(op, weights_only=False)
        d_dom = torch.load(dp, weights_only=False)
        n_checked += 1

        fe_count_orig += int(d_orig["method"].y_feature_envy.sum().item())
        fe_count_dom += int(d_dom["method"].y_feature_envy.sum().item())
        lm_count_orig += int(d_orig["method"].y_long_method.sum().item()) + int(d_orig["function"].y_long_method.sum().item())
        lm_count_dom += int(d_dom["method"].y_long_method.sum().item()) + int(d_dom["function"].y_long_method.sum().item())
        gc_count_orig += int(d_orig["class"].y.sum().item())
        gc_count_dom += int(d_dom["class"].y.sum().item())

        file_ok = True
        for ntype in ("method", "function"):
            xo, xd = d_orig[ntype].x, d_dom[ntype].x
            if xo.shape[0] != xd.shape[0]:
                file_ok = False
                unexpected.append((str(rel), f"{ntype} node count differs: {xo.shape[0]} vs {xd.shape[0]}"))
                break
            if xo.shape[0] == 0:
                continue
            if not torch.equal(xo[:, :4], xd[:, :4]):
                file_ok = False
                unexpected.append((str(rel), f"{ntype} columns 0-3 differ (unexpected)"))
                break
            if not torch.equal(d_orig[ntype].y_long_method, d_dom[ntype].y_long_method):
                file_ok = False
                unexpected.append((str(rel), f"{ntype} y_long_method differs (unexpected)"))
                break
            if not torch.equal(d_orig[ntype].y_feature_envy, d_dom[ntype].y_feature_envy):
                file_ok = False
                unexpected.append((str(rel), f"{ntype} y_feature_envy differs (unexpected)"))
                break
        if file_ok and d_orig["class"].x.shape[0] > 0:
            if not torch.equal(d_orig["class"].x, d_dom["class"].x) or not torch.equal(d_orig["class"].y, d_dom["class"].y):
                file_ok = False
                unexpected.append((str(rel), "class features/labels differ (unexpected -- god_class doesn't use this feature at all)"))
        if not file_ok:
            n_unexpected_diff += 1
        else:
            n_col4_only_diff += 1

    print(f"[verify] files checked: {n_checked}, missing counterpart: {n_missing}")
    print(f"[verify] identical everywhere except (at most) column 4: {n_col4_only_diff}")
    print(f"[verify] UNEXPECTED differences beyond column 4: {n_unexpected_diff}")
    if unexpected:
        print("[verify] examples of unexpected differences:")
        for u in unexpected[:15]:
            print("   ", u)

    print(f"\n[labels] corrected_original: feature_envy={fe_count_orig} long_method={lm_count_orig} god_class={gc_count_orig}")
    print(f"[labels] corrected_dominant: feature_envy={fe_count_dom} long_method={lm_count_dom} god_class={gc_count_dom}")
    print("[labels] (these two rows are expected to match each other exactly -- same parser, only the")
    print(" input FEATURE differs, never the label. Labels are NOT expected to match the old, stale")
    print(" data/processed/graphs_hybrid/test counts -- that comparison is reported separately below.)")

    # For reference only: how do the corrected label counts compare to the
    # STALE existing TEST data's own label counts? Read-only, no file touched.
    from pathlib import Path as _P
    stale_dir = ROOT / "data" / "processed" / "graphs" / "test"
    if stale_dir.exists():
        fe_stale, lm_stale, gc_stale = 0, 0, 0
        for p in stale_dir.rglob("*.pt"):
            d = torch.load(p, weights_only=False)
            fe_stale += int(d["method"].y_feature_envy.sum().item()) if "method" in d.node_types else 0
            lm_stale += (int(d["method"].y_long_method.sum().item()) if "method" in d.node_types else 0) + \
                        (int(d["function"].y_long_method.sum().item()) if "function" in d.node_types else 0)
            gc_stale += int(d["class"].y.sum().item()) if "class" in d.node_types else 0
        print(f"\n[reference, read-only] STALE existing data/processed/graphs/test labels: "
              f"feature_envy={fe_stale} long_method={lm_stale} god_class={gc_stale}")
        print(f"[reference] delta (corrected - stale): "
              f"feature_envy={fe_count_orig - fe_stale} long_method={lm_count_orig - lm_stale} god_class={gc_count_orig - gc_stale}")

    if n_missing or n_unexpected_diff:
        print("\n[RESULT] VERIFICATION FAILED -- do not proceed to CodeBERT augmentation or evaluation.")
        raise SystemExit(1)
    print("\n[RESULT] VERIFICATION PASSED: corrected_original and corrected_dominant are identical "
          "in every feature except column 4 (method/function), across all files checked.")
    print("Stopping here per instruction. No CodeBERT augmentation run. No TEST inference run.")


if __name__ == "__main__":
    main()
