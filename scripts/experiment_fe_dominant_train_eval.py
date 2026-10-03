# -*- coding: utf-8 -*-
"""EXPERIMENT-ONLY training+eval (authorized: docs/feature_envy_generalization_investigation.md
section 10, user-approved scope in this session). Trains the 4-seed
sweep (42/43/44/45) on the experiment's dominant_external_count hybrid
graphs, using the EXACT current best_params.json hyperparameters
(models/hybrid_class_pool_tuned_fixed_data/best_params.json, itself
never modified) -- no hyperparameter search, no architecture change.

Reads ONLY data/processed/graphs_hybrid_experiment_fe_dominant/{train,val}
(this experiment's own output) -- never touches data/raw/test,
data/processed/graphs_hybrid/test, or any TEST artifact. Never loads,
modifies, or overwrites models/hybrid_class_pool_tuned_fixed_data (the
deployed checkpoint) or backend/app/inference.py.

Saves per-seed checkpoints under models/experiment_fe_dominant/seed_NN/
(new, separate directory -- not overwriting anything) and a single JSON
with every number the final report needs, including a per-repo VAL
feature_envy breakdown (sqlalchemy/sphinx/starlette) to check whether
any improvement is sqlalchemy-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.loader import DataLoader

import scripts.tune_hybrid_optuna as tho
from ml.models.gat_baseline import HeteroGAT
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS, apply_norm, compute_norm_stats, run_eval

FROZEN_DIR = ROOT / "models" / "hybrid_class_pool_tuned_fixed_data"  # deployed candidate -- READ ONLY
EXPERIMENT_HYBRID_DIR = ROOT / "data" / "processed" / "graphs_hybrid_experiment_fe_dominant"
EXPERIMENT_MODELS_DIR = ROOT / "models" / "experiment_fe_dominant"
DOCS_DIR = ROOT / "docs"
EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 32
SEEDS = [42, 43, 44, 45]


def load_split_with_repo(split: str):
    """Like scripts.train_hybrid_baseline.load_split, but from the
    experiment directory and preserving each graph's repo name (needed
    for the per-repo VAL breakdown -- not needed/used for TRAIN)."""
    paths = sorted((EXPERIMENT_HYBRID_DIR / split).rglob("*.pt"))
    graphs, repos = [], []
    for p in paths:
        rel = p.relative_to(EXPERIMENT_HYBRID_DIR / split)
        repos.append(rel.parts[0])
        graphs.append(torch.load(p, weights_only=False))
    return graphs, repos


def macro_f1(metrics: dict) -> float:
    f1s = [v["f1"] for v in metrics.values() if v is not None]
    return sum(f1s) / len(f1s) if f1s else 0.0


@torch.no_grad()
def feature_envy_by_repo(model, val_graphs, val_repos, norm_stats, device):
    """Per-repo feature_envy TP/FP/FN/TN at threshold 0.5 (this script
    evaluates the raw model, not the live-serving 0.65 threshold -- the
    deployed threshold is a separate, unmodified production concern)."""
    model.eval()
    by_repo = {}
    for g, repo in zip(val_graphs, val_repos):
        if g["method"].x.shape[0] == 0:
            continue
        edge_index_dict = {et: g[et].edge_index for et in g.edge_types}
        h = model(g.x_dict, edge_index_dict)
        logits = model.predict_feature_envy(h)
        probs = torch.sigmoid(logits)
        labels = g["method"].y_feature_envy
        pred = (probs >= 0.5).float()
        d = by_repo.setdefault(repo, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        for p_, y_ in zip(pred.tolist(), labels.tolist()):
            if y_ == 1 and p_ == 1:
                d["tp"] += 1
            elif y_ == 0 and p_ == 1:
                d["fp"] += 1
            elif y_ == 1 and p_ == 0:
                d["fn"] += 1
            else:
                d["tn"] += 1
    return by_repo


def main():
    best = json.loads((FROZEN_DIR / "best_params.json").read_text(encoding="utf-8"))
    print(f"[config] reusing frozen best_params.json unchanged: {best}")

    all_results = {}

    for seed in SEEDS:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        tho.SEED = seed
        torch.manual_seed(seed)

        train_graphs, _train_repos = load_split_with_repo("train")
        val_graphs, val_repos = load_split_with_repo("val")
        print(f"[load] train={len(train_graphs)} val={len(val_graphs)}")

        norm_stats = compute_norm_stats(train_graphs)
        apply_norm(train_graphs, norm_stats)
        apply_norm(val_graphs, norm_stats)

        loss_lm, loss_fe, loss_gc = tho.build_losses(train_graphs)
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

        best_score, best_epoch, best_state = tho.train_one(
            best["hidden_dim"], best["heads"], best["dropout"],
            best["lr"], best["weight_decay"],
            train_loader, val_loader, loss_lm, loss_fe, loss_gc,
            device=torch.device("cpu"), epochs=EPOCHS, patience=PATIENCE, trial=None,
            edge_types=None, class_method_pool=True,
            weight_lm=best.get("weight_lm", 1.0), weight_fe=best.get("weight_fe", 1.0), weight_gc=best.get("weight_gc", 1.0),
        )
        print(f"[train] done: best_epoch={best_epoch} best_val_macro_f1={best_score:.4f}")

        model = HeteroGAT(
            hidden_dim=best["hidden_dim"], heads=best["heads"], dropout=best["dropout"],
            node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
        )
        model.load_state_dict(best_state)
        model.eval()

        metrics = run_eval(model, val_loader, torch.device("cpu"))
        repo_breakdown = feature_envy_by_repo(model, val_graphs, val_repos, norm_stats, torch.device("cpu"))

        print(f"[seed {seed}] macro_f1={macro_f1(metrics):.4f} "
              f"long_method_f1={metrics['long_method']['f1']:.4f} "
              f"feature_envy_f1={metrics['feature_envy']['f1']:.4f} "
              f"god_class_f1={metrics['god_class']['f1']:.4f}")
        print(f"[seed {seed}] feature_envy by repo: {repo_breakdown}")

        seed_dir = EXPERIMENT_MODELS_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, seed_dir / "model.pt")
        torch.save({nt: (m.tolist(), s.tolist()) for nt, (m, s) in norm_stats.items()}, seed_dir / "norm_stats.pt")
        (seed_dir / "best_params.json").write_text(json.dumps({**best, "best_epoch": best_epoch, "seed": seed}, indent=2))
        (seed_dir / "final_val_metrics.json").write_text(json.dumps(metrics, indent=2))

        all_results[seed] = {
            "best_epoch": best_epoch,
            "macro_f1": macro_f1(metrics),
            "metrics": metrics,
            "feature_envy_by_repo": repo_breakdown,
        }

    out_path = DOCS_DIR / "experiment_fe_dominant_raw_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved raw results to {out_path}")
    print("Deployed checkpoint (models/hybrid_class_pool_tuned_fixed_data) was NOT modified.")
    print("Production ml/graph/graph_builder.py was NOT modified.")
    print("backend/app/inference.py was NOT modified.")
    print("TEST was never read.")


if __name__ == "__main__":
    main()
