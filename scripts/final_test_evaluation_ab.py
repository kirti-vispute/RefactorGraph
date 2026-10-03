# -*- coding: utf-8 -*-
"""FINAL TEST EVALUATION -- AUTHORIZED (user-authorized, one-time).

Evaluation A: deployed checkpoint (models/hybrid_class_pool_tuned_fixed_data,
seed=43, God-Class-fix candidate promoted earlier this session) on
data/processed/graphs_hybrid_test_corrected_original (current parser +
external_access_count, CodeBERT-augmented this session).

Evaluation B: models/experiment_fe_dominant/seed_43 ONLY on
data/processed/graphs_hybrid_test_corrected_dominant (current parser +
dominant_external_count, patched from the same base as A).

Exact same methodology as scripts/phase16_final_test_eval.py
(evaluate_checkpoint): HeteroGAT(class_method_pool=True), each
checkpoint's OWN norm_stats (fit on its own TRAIN, never refit here),
DataLoader(batch_size=64, shuffle=False), run_eval from
scripts/train_hybrid_baseline.py (fixed 0.5 threshold for all 3 tasks --
the same threshold convention the historical 0.740/0.525/0.767/0.677
baseline was measured with; NOT the live-serving 0.65 FE threshold in
backend/app/inference.py, which is a separate, unrelated deployment
concern).

Run exactly once. No retry. No tuning. No deployment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.loader import DataLoader

from ml.models.gat_baseline import HeteroGAT
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS, apply_norm, run_eval

BATCH_SIZE = 64


def load_norm_stats(path: Path) -> dict:
    raw = torch.load(path, weights_only=False)
    return {nt: (torch.tensor(m), torch.tensor(s)) for nt, (m, s) in raw.items()}


def evaluate(name: str, model_dir: Path, graph_dir: Path) -> dict:
    params = json.loads((model_dir / "best_params.json").read_text(encoding="utf-8"))
    norm_stats = load_norm_stats(model_dir / "norm_stats.pt")

    graphs = [torch.load(p, weights_only=False) for p in sorted(graph_dir.rglob("*.pt"))]
    print(f"[{name}] n_graphs={len(graphs)}")
    apply_norm(graphs, norm_stats)
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    model = HeteroGAT(hidden_dim=params["hidden_dim"], heads=params["heads"], dropout=params["dropout"],
                       node_feature_dims=HYBRID_NODE_DIMS, edge_types=None,
                       class_method_pool=params["class_method_pool"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    state = torch.load(model_dir / "model.pt", weights_only=True)
    model.load_state_dict(state)
    print(f"[{name}] trainable_params={n_params} class_method_pool={params['class_method_pool']}")

    metrics = run_eval(model, loader, device)
    print(f"[{name}] " + " ".join(
        f"{task}_f1={v['f1']:.3f}" if v else f"{task}_f1=n/a" for task, v in metrics.items()
    ))
    return metrics, len(graphs)


def macro_f1(metrics: dict):
    f1s = [v["f1"] for v in metrics.values() if v is not None]
    return sum(f1s) / len(f1s) if f1s else None


def main():
    print("=" * 70)
    print("EVALUATION A: deployed checkpoint on corrected_original TEST")
    print("=" * 70)
    metrics_a, n_a = evaluate(
        "A",
        ROOT / "models" / "hybrid_class_pool_tuned_fixed_data",
        ROOT / "data" / "processed" / "graphs_hybrid_test_corrected_original",
    )

    print()
    print("=" * 70)
    print("EVALUATION B: seed_43 candidate on corrected_dominant TEST")
    print("=" * 70)
    metrics_b, n_b = evaluate(
        "B",
        ROOT / "models" / "experiment_fe_dominant" / "seed_43",
        ROOT / "data" / "processed" / "graphs_hybrid_test_corrected_dominant",
    )

    result = {
        "n_test_graphs_a": n_a,
        "n_test_graphs_b": n_b,
        "evaluation_a": metrics_a,
        "evaluation_b": metrics_b,
        "macro_f1_a": macro_f1(metrics_a),
        "macro_f1_b": macro_f1(metrics_b),
    }
    out_path = ROOT / "docs" / "final_test_evaluation_ab_raw.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[saved raw results] {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for task in ("long_method", "feature_envy", "god_class"):
        va = metrics_a[task]
        vb = metrics_b[task]
        print(f"{task}: A f1={va['f1']:.4f} (p={va['precision']:.4f} r={va['recall']:.4f} tp={va['tp']} fp={va['fp']} fn={va['fn']} tn={va['tn']})")
        print(" " * len(task) + f"  B f1={vb['f1']:.4f} (p={vb['precision']:.4f} r={vb['recall']:.4f} tp={vb['tp']} fp={vb['fp']} fn={vb['fn']} tn={vb['tn']})")
    print(f"macro-F1: A={macro_f1(metrics_a):.4f}  B={macro_f1(metrics_b):.4f}")


if __name__ == "__main__":
    main()
