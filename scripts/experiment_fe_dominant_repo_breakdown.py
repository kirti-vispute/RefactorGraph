# -*- coding: utf-8 -*-
"""Corrected per-repo Feature Envy breakdown for the 4-seed experiment
(authorized: docs/feature_envy_generalization_investigation.md section
10). Reuses the already-saved checkpoints from
scripts/experiment_fe_dominant_train_eval.py (models/experiment_fe_dominant/seed_*)
-- no retraining.

Replaces scripts/experiment_fe_dominant_train_eval.py's own per-repo
function, which used an UNBATCHED (per-graph) forward pass and was found,
before being reported, to diverge substantially from the batched
inference that produces the headline run_eval metrics (up to ~12 logit
units on some nodes, not the ~0.01 floating-point batching noise this
project has documented elsewhere -- a real discrepancy, not negligible
precision noise; root cause not fully isolated, but avoided entirely
here by using the SAME batch_size=64 batched forward pass as
run_eval/the headline numbers, so per-repo counts are guaranteed to sum
exactly to the already-reported aggregate tp/fp/fn/tn).
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
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS, apply_norm

EXPERIMENT_HYBRID_DIR = ROOT / "data" / "processed" / "graphs_hybrid_experiment_fe_dominant"
EXPERIMENT_MODELS_DIR = ROOT / "models" / "experiment_fe_dominant"
SEEDS = [42, 43, 44, 45]
BATCH_SIZE = 64


def load_val_with_repo():
    paths = sorted((EXPERIMENT_HYBRID_DIR / "val").rglob("*.pt"))
    graphs, repos = [], []
    for p in paths:
        rel = p.relative_to(EXPERIMENT_HYBRID_DIR / "val")
        repos.append(rel.parts[0])
        graphs.append(torch.load(p, weights_only=False))
    return graphs, repos


@torch.no_grad()
def batched_repo_breakdown(model, val_graphs, val_repos):
    """Same batch_size/shuffle as run_eval, so results are numerically
    identical to the already-reported aggregate metrics -- just also
    tallied per source repo via each batch's own graph-index vector."""
    loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    by_repo = {}
    graph_ptr = 0
    all_tp = all_fp = all_fn = all_tn = 0
    for batch in loader:
        edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
        h = model(batch.x_dict, edge_index_dict)
        if "method" not in h or batch["method"].x.shape[0] == 0:
            graph_ptr += batch.num_graphs
            continue
        logits = model.predict_feature_envy(h)
        probs = torch.sigmoid(logits)
        labels = batch["method"].y_feature_envy
        pred = (probs >= 0.5).float()
        node_graph_idx = batch["method"].batch  # which graph (0..num_graphs-1) within this batch, per node
        for i in range(labels.shape[0]):
            repo = val_repos[graph_ptr + int(node_graph_idx[i].item())]
            y_, p_ = int(labels[i].item()), int(pred[i].item())
            d = by_repo.setdefault(repo, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            if y_ == 1 and p_ == 1:
                d["tp"] += 1
                all_tp += 1
            elif y_ == 0 and p_ == 1:
                d["fp"] += 1
                all_fp += 1
            elif y_ == 1 and p_ == 0:
                d["fn"] += 1
                all_fn += 1
            else:
                d["tn"] += 1
                all_tn += 1
        graph_ptr += batch.num_graphs
    return by_repo, {"tp": all_tp, "fp": all_fp, "fn": all_fn, "tn": all_tn}


def main():
    val_graphs, val_repos = load_val_with_repo()
    results = {}
    for seed in SEEDS:
        seed_dir = EXPERIMENT_MODELS_DIR / f"seed_{seed}"
        best = json.loads((seed_dir / "best_params.json").read_text())
        state = torch.load(seed_dir / "model.pt", weights_only=True)
        raw_norm = torch.load(seed_dir / "norm_stats.pt", weights_only=False)
        norm_stats = {nt: (torch.tensor(m), torch.tensor(s)) for nt, (m, s) in raw_norm.items()}

        graphs_copy = [g.clone() for g in val_graphs]
        apply_norm(graphs_copy, norm_stats)

        model = HeteroGAT(
            hidden_dim=best["hidden_dim"], heads=best["heads"], dropout=best["dropout"],
            node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
        )
        model.load_state_dict(state)
        model.eval()

        by_repo, agg = batched_repo_breakdown(model, graphs_copy, val_repos)
        p = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) else 0.0
        r = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"[seed {seed}] aggregate (batched, cross-check vs run_eval): "
              f"tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} tn={agg['tn']} "
              f"precision={p:.4f} recall={r:.4f} f1={f1:.4f}")
        print(f"[seed {seed}] per-repo: {by_repo}")
        results[seed] = {"aggregate": agg, "by_repo": by_repo}

    out_path = ROOT / "docs" / "experiment_fe_dominant_repo_breakdown.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
