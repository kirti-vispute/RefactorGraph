# -*- coding: utf-8 -*-
"""FINAL, ONE-TIME TEST evaluation of the Experiment #1 seed=43 candidate
(models/experiment_fe_dominant/seed_43/). Explicitly authorized by the
user in this session, after the full TRAIN/VAL review
(docs/experiment_fe_dominant_candidate_review.md) recommended it.

This is the ONLY script in this experiment that reads data/raw/test.
Read-only against production: does not modify
data/processed/graphs/test, data/processed/graphs_hybrid/test (only
reads them, to patch a NEW, separate file), the deployed checkpoint,
production graph_builder.py, or backend/app/inference.py.

Steps (mirrors the TRAIN/VAL methodology exactly, same integrity
discipline):
  1. Build TEST structural graphs using the experiment's
     dominant_external_count feature (scratch_external_eval/graph_builder_dominant_ext.py)
     -> data/processed/graphs_experiment_fe_dominant/test (NEW, separate
     from production data/processed/graphs/test).
  2. Patch column 4 (method/function) into a copy of the EXISTING
     production TEST hybrid graphs (data/processed/graphs_hybrid/test,
     already CodeBERT-augmented once, per this project's one-time-TEST-
     touch discipline -- not re-embedded here) -> NEW directory
     data/processed/graphs_hybrid_experiment_fe_dominant/test. Every
     file verified byte-identical outside column 4 and all labels
     before being trusted, exactly as done for TRAIN/VAL.
  3. Evaluate models/experiment_fe_dominant/seed_43/model.pt on this
     TEST set, same batched run_eval methodology as
     scripts/phase16_final_test_eval.py, at the standard 0.5 threshold
     (matching how every other TEST number in this project has been
     reported -- this raw-metric evaluation is independent of the live
     0.65 Feature Envy serving threshold, which is a separate,
     unmodified, production-only concern).

Output: docs/experiment_fe_dominant_test_results.json,
printed report used verbatim in the final TEST report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scratch_external_eval.graph_builder_dominant_ext import build_hetero_graph
from ml.preprocessing.label_rules import LabelThresholds
from ml.models.gat_baseline import HeteroGAT
from scripts.build_dataset import parse_split
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS, apply_norm, run_eval
from torch_geometric.loader import DataLoader

STRUCT_TEST_DIR = ROOT / "data" / "processed" / "graphs_experiment_fe_dominant" / "test"
PROD_HYBRID_TEST_DIR = ROOT / "data" / "processed" / "graphs_hybrid" / "test"
OUT_HYBRID_TEST_DIR = ROOT / "data" / "processed" / "graphs_hybrid_experiment_fe_dominant" / "test"
CANDIDATE_DIR = ROOT / "models" / "experiment_fe_dominant" / "seed_43"
DOCS_DIR = ROOT / "docs"


def step1_build_structural():
    print("[step 1] parsing TEST (data/raw/test) -- this is the one authorized TEST touch ...")
    modules, parse_errors, file_count = parse_split("test")
    print(f"[step 1] test: {file_count} files, {len(modules)} parsed ok, {len(parse_errors)} parse errors")

    # Thresholds: load directly from the already-verified, currently-active
    # configs/label_thresholds.json (TRAIN-derived, unchanged by this
    # experiment -- already confirmed identical to a fresh TRAIN
    # recomputation in scripts/experiment_fe_dominant_rebuild_structural.py's
    # own integrity check) rather than recomputing, to avoid any
    # redundant TRAIN re-parse and use the single authoritative source.
    cfg = json.loads((ROOT / "configs" / "label_thresholds.json").read_text())
    thresholds = LabelThresholds(
        long_method_statements=cfg["long_method_statements"],
        god_class_method_count=cfg["god_class_method_count"],
        god_class_loc=cfg["god_class_loc"],
        feature_envy_min_external_calls=cfg["feature_envy_min_external_calls"],
        god_class_min_fields=cfg["god_class_min_fields"],
        god_class_min_methods_for_field_branch=cfg["god_class_min_methods_for_field_branch"],
    )
    print(f"[step 1] thresholds (from configs/label_thresholds.json, unchanged): {thresholds}")

    n_graphs, failures = 0, []
    for repo, relpath, mod in modules:
        try:
            result = build_hetero_graph(mod, thresholds)
        except Exception as e:  # noqa: BLE001
            failures.append((repo, relpath, f"{type(e).__name__}: {e}"))
            continue
        out_path = STRUCT_TEST_DIR / repo / (relpath.replace("\\", "/") + ".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result.data, out_path)
        n_graphs += 1
    print(f"[step 1] built {n_graphs} structural TEST graphs (experiment feature), {len(failures)} failures")
    if failures:
        for f in failures[:10]:
            print("  ", f)
    return thresholds


def step2_patch_hybrid():
    print("\n[step 2] patching TEST hybrid graphs (column 4 only, verified first) ...")
    struct_files = sorted(STRUCT_TEST_DIR.rglob("*.pt"))
    patched, mismatched, missing = 0, [], []
    for sp in struct_files:
        rel = sp.relative_to(STRUCT_TEST_DIR)
        hp = PROD_HYBRID_TEST_DIR / rel
        if not hp.exists():
            missing.append(str(rel))
            continue
        d_struct = torch.load(sp, weights_only=False)
        d_hybrid = torch.load(hp, weights_only=False)

        ok = True
        for ntype in ("method", "function"):
            xs, xh = d_struct[ntype].x, d_hybrid[ntype].x
            if xs.shape[0] != xh.shape[0]:
                ok = False
                break
            if xs.shape[0] == 0:
                continue
            if not torch.equal(xs[:, :4], xh[:, :4]):
                ok = False
                break
            if not torch.equal(d_struct[ntype].y_long_method, d_hybrid[ntype].y_long_method):
                ok = False
                break
            if not torch.equal(d_struct[ntype].y_feature_envy, d_hybrid[ntype].y_feature_envy):
                ok = False
                break
        if ok and d_struct["class"].x.shape[0] > 0:
            if not torch.equal(d_struct["class"].y, d_hybrid["class"].y):
                ok = False
        if not ok:
            mismatched.append(str(rel))
            continue

        for ntype in ("method", "function"):
            if d_hybrid[ntype].x.shape[0] == 0:
                continue
            new_x = d_hybrid[ntype].x.clone()
            new_x[:, 4] = d_struct[ntype].x[:, 4]
            d_hybrid[ntype].x = new_x

        out_path = OUT_HYBRID_TEST_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d_hybrid, out_path)
        patched += 1

    print(f"[step 2] patched={patched} missing={len(missing)} mismatched={len(mismatched)}")
    if missing:
        print("  missing:", missing[:10])
    if mismatched:
        print("  mismatched:", mismatched[:10])
    if missing or mismatched:
        raise SystemExit("[ABORT] TEST hybrid patch had missing/mismatched files -- refusing to evaluate on unverified data.")
    print("[step 2] OK: every TEST file verified byte-identical outside column 4 before patching.")


def macro_f1(metrics):
    f1s = [v["f1"] for v in metrics.values() if v is not None]
    return sum(f1s) / len(f1s) if f1s else 0.0


def step3_evaluate():
    print("\n[step 3] evaluating candidate (seed=43 experiment) on patched TEST set ...")
    best = json.loads((CANDIDATE_DIR / "best_params.json").read_text())
    state = torch.load(CANDIDATE_DIR / "model.pt", weights_only=True)
    raw_norm = torch.load(CANDIDATE_DIR / "norm_stats.pt", weights_only=False)
    norm_stats = {nt: (torch.tensor(m), torch.tensor(s)) for nt, (m, s) in raw_norm.items()}

    test_graphs = [torch.load(p, weights_only=False) for p in sorted(OUT_HYBRID_TEST_DIR.rglob("*.pt"))]
    print(f"[step 3] test graphs loaded: {len(test_graphs)}")
    apply_norm(test_graphs, norm_stats)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    model = HeteroGAT(
        hidden_dim=best["hidden_dim"], heads=best["heads"], dropout=best["dropout"],
        node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
    )
    model.load_state_dict(state)
    model.eval()

    metrics = run_eval(model, test_loader, torch.device("cpu"))
    result = {"metrics": metrics, "macro_f1": macro_f1(metrics), "n_test_graphs": len(test_graphs)}

    print(f"\n[RESULT] long_method: {metrics['long_method']}")
    print(f"[RESULT] feature_envy: {metrics['feature_envy']}")
    print(f"[RESULT] god_class: {metrics['god_class']}")
    print(f"[RESULT] macro_f1: {result['macro_f1']:.4f}")

    out_path = DOCS_DIR / "experiment_fe_dominant_test_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")
    return result


def main():
    step1_build_structural()
    step2_patch_hybrid()
    step3_evaluate()
    print("\nDone. TEST evaluated exactly once for this candidate.")
    print("Production files, deployed checkpoint, thresholds: untouched.")


if __name__ == "__main__":
    main()
