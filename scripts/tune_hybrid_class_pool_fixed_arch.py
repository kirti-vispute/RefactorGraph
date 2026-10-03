# -*- coding: utf-8 -*-
"""Phase 13e: follow-up to the Phase 13c free Optuna search over Design B
(architecture + loss weights jointly). Motivation, from analyzing Phase
13c's models/hybrid_class_pool_tuned/optuna_trials.csv (13 completed
trials, per-task F1 logged as user attrs — Phase 13d):

Trial 9 (heads=4, hidden_dim=128, dropout=0.125 — essentially Phase 10's
own architecture) scored long_method_f1=0.804, the best of any completed
trial and close to Phase 10's 0.831. But its macro-F1 (0.630) lost to
smaller architectures like trial 17's winning heads=2/hidden_dim=64
(long_method_f1=0.789, macro=0.648) because those smaller configs traded
long_method capacity for larger feature_envy/god_class gains. The free
search's macro-F1 objective is indifferent to *which* task a config is
good at, so it converged on a config that is NOT the best available for
long_method specifically — the joint architecture+weight search spent
budget exploring capacity trade-offs the loss-weight axis alone can't
separate from.

This script isolates the two axes: architecture is pinned to Phase 10's
own config (heads=4, hidden_dim=128 via fixed_heads/fixed_hidden_dim_per_
head on build_objective, ml.tune_hybrid_class_pool_optuna.build_objective),
and only dropout/lr/weight_decay/weight_fe/weight_gc are searched. If
long_method's residual regression is fixable by loss reweighting at all,
it should show up here, at the capacity where Design B's untuned run
(models/hybrid_class_pool/final_val_metrics.json) already got
long_method_f1=0.791 close to Phase 10 with weight_fe=weight_gc=1.0.

Same discipline as every prior phase: TRAIN split for training, VAL split
for the search objective/pruning and final report, TEST split untouched
(never read here). No labels, graph construction, or split changed.

Output: models/hybrid_class_pool_fixed_arch/{model.pt, norm_stats.pt,
best_params.json, final_val_metrics.json, optuna_trials.csv},
docs/class_pool_fixed_arch_report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
import torch
from torch_geometric.loader import DataLoader

from ml.models.gat_baseline import HeteroGAT
from scripts.train_hybrid_baseline import (
    HYBRID_NODE_DIMS,
    apply_norm,
    compute_norm_stats,
    load_split,
    run_eval,
)
from scripts.tune_hybrid_class_pool_optuna import build_objective
from scripts.tune_hybrid_optuna import build_losses, train_one

ROOT = Path(__file__).resolve().parents[1]
TUNED_DIR = ROOT / "models" / "hybrid_tuned"
CLASS_POOL_DIR = ROOT / "models" / "hybrid_class_pool"
CLASS_POOL_TUNED_DIR = ROOT / "models" / "hybrid_class_pool_tuned"
MODELS_DIR = ROOT / "models" / "hybrid_class_pool_fixed_arch"
DOCS_DIR = ROOT / "docs"

EPOCHS = 150
PATIENCE = 15
SEARCH_EPOCHS = 60
SEARCH_PATIENCE = 10
BATCH_SIZE = 32
SEED = 42
N_TRIALS = 20

FIXED_HEADS = 4
FIXED_HIDDEN_DIM_PER_HEAD = 32  # hidden_dim = 128, matches models/hybrid_tuned/best_params.json


def main():
    device = torch.device("cpu")
    print("[load] reading hybrid graphs ...")
    train_graphs = load_split("train")
    val_graphs = load_split("val")
    print(f"[load] train={len(train_graphs)} val={len(val_graphs)}")

    norm_stats = compute_norm_stats(train_graphs)
    apply_norm(train_graphs, norm_stats)
    apply_norm(val_graphs, norm_stats)

    loss_lm, loss_fe, loss_gc = build_losses(train_graphs)

    objective = build_objective(
        train_graphs, val_graphs, loss_lm, loss_fe, loss_gc, device,
        fixed_heads=FIXED_HEADS, fixed_hidden_dim_per_head=FIXED_HIDDEN_DIM_PER_HEAD,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"[search done] best macro val f1={study.best_value:.4f} params={study.best_params}")
    print(f"[search done] best trial per-task f1: {study.best_trial.user_attrs}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(MODELS_DIR / "optuna_trials.csv", index=False)

    best = study.best_params
    hidden_dim = FIXED_HEADS * FIXED_HIDDEN_DIM_PER_HEAD
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    best_score, best_epoch, best_state = train_one(
        hidden_dim, FIXED_HEADS, best["dropout"], best["lr"], best["weight_decay"],
        train_loader, val_loader, loss_lm, loss_fe, loss_gc,
        device, EPOCHS, PATIENCE, trial=None, edge_types=None, class_method_pool=True,
        weight_lm=1.0, weight_fe=best["weight_fe"], weight_gc=best["weight_gc"],
    )

    model = HeteroGAT(hidden_dim=hidden_dim, heads=FIXED_HEADS, dropout=best["dropout"],
                       node_feature_dims=HYBRID_NODE_DIMS, edge_types=None,
                       class_method_pool=True).to(device)
    model.load_state_dict(best_state)
    final_val_metrics = run_eval(model, val_loader, device)

    torch.save(best_state, MODELS_DIR / "model.pt")
    torch.save({nt: (m.tolist(), s.tolist()) for nt, (m, s) in norm_stats.items()},
                MODELS_DIR / "norm_stats.pt")
    (MODELS_DIR / "best_params.json").write_text(json.dumps({
        "heads": FIXED_HEADS, "hidden_dim": hidden_dim, "dropout": best["dropout"],
        "lr": best["lr"], "weight_decay": best["weight_decay"],
        "weight_lm": 1.0, "weight_fe": best["weight_fe"], "weight_gc": best["weight_gc"],
        "class_method_pool": True, "best_epoch": best_epoch,
    }, indent=2), encoding="utf-8")
    (MODELS_DIR / "final_val_metrics.json").write_text(
        json.dumps(final_val_metrics, indent=2), encoding="utf-8"
    )

    def _load(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    tuned_metrics = _load(TUNED_DIR / "final_val_metrics.json")
    pool_metrics = _load(CLASS_POOL_DIR / "final_val_metrics.json")
    pool_tuned_metrics = _load(CLASS_POOL_TUNED_DIR / "final_val_metrics.json")

    def macro_f1(metrics):
        if not metrics:
            return None
        f1s = [v["f1"] for v in metrics.values() if v is not None]
        return sum(f1s) / len(f1s) if f1s else None

    lines = ["# Class-Method-Pool Fixed-Architecture Tuning Report (Phase 13e, Design B)\n"]
    lines.append(
        "Follow-up to docs/class_pool_tuning_report.md (Phase 13c). Analyzing that search's "
        "per-task-logged trials (models/hybrid_class_pool_tuned/optuna_trials.csv, added in "
        "Phase 13d) showed the best single trial for long_method (heads=4, hidden_dim=128, "
        "essentially Phase 10's own architecture) scored long_method_f1=0.804, close to Phase "
        "10's 0.831 — but lost on macro-F1 to smaller architectures that traded long_method "
        "capacity for feature_envy/god_class gains. This run pins the architecture to Phase "
        f"10's own (heads={FIXED_HEADS}, hidden_dim={hidden_dim}) and searches only "
        "dropout/lr/weight_decay/weight_fe/weight_gc, to isolate whether loss reweighting alone "
        "(without shrinking capacity) can close long_method's gap while keeping the "
        "feature_envy/god_class gains from Phase 13c.\n"
    )
    lines.append(
        f"TPE sampler + median pruner, seed={SEED}, {N_TRIALS} trials, search budget "
        f"max_epochs={SEARCH_EPOCHS}/patience={SEARCH_PATIENCE}. Architecture fixed: "
        f"heads={FIXED_HEADS}, hidden_dim={hidden_dim}, class_method_pool=True. Search space: "
        f"dropout in [0.0, 0.5], lr log-uniform [1e-4, 1e-2], weight_decay log-uniform "
        f"[1e-6, 1e-2], weight_fe in [0.5, 4.0], weight_gc in [0.5, 4.0] (weight_lm fixed at "
        f"1.0). Objective: macro-F1. Winning config retrained with the full budget "
        f"(max_epochs={EPOCHS}, patience={PATIENCE}) for the numbers below.\n"
        f"\nOnly TRAIN and VAL splits are touched. TEST split remains untouched.\n"
        f"\nBest params: dropout={best['dropout']:.3f}, lr={best['lr']:.2e}, "
        f"weight_decay={best['weight_decay']:.2e}, weight_fe={best['weight_fe']:.3f}, "
        f"weight_gc={best['weight_gc']:.3f}, best_epoch={best_epoch}.\n"
    )

    if tuned_metrics and pool_metrics and pool_tuned_metrics:
        lines.append(
            "\n## Comparison: Phase 10 vs untuned Design B vs Design B tuned (free arch, "
            "Phase 13c) vs Design B tuned (fixed arch, Phase 13e) — same val split\n"
        )
        header = "| task | Phase 10 | Design B untuned | Design B tuned (free arch) | Design B tuned (fixed arch) |"
        lines.append(header)
        lines.append("|" + "---|" * (header.count("|") - 1))
        for task in ("long_method", "feature_envy", "god_class"):
            old, pool, pool_tuned, fixed = (
                tuned_metrics.get(task), pool_metrics.get(task),
                pool_tuned_metrics.get(task), final_val_metrics.get(task),
            )
            if not (old and pool and pool_tuned and fixed):
                lines.append(f"| {task} | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {task} | {old['f1']:.3f} | {pool['f1']:.3f} | {pool_tuned['f1']:.3f} | {fixed['f1']:.3f} |"
            )
        macro_old = macro_f1(tuned_metrics)
        macro_pool = macro_f1(pool_metrics)
        macro_pool_tuned = macro_f1(pool_tuned_metrics)
        macro_fixed = macro_f1(final_val_metrics)
        lines.append(
            f"| **macro-F1** | {macro_old:.3f} | {macro_pool:.3f} | {macro_pool_tuned:.3f} | {macro_fixed:.3f} |"
        )
    else:
        lines.append("\n(comparison baselines not found — no comparison table.)\n")

    for task, v in final_val_metrics.items():
        if v is None:
            lines.append(f"\n## {task} (fixed-arch tuned Design B raw)\n\n- no eligible nodes in val split\n")
            continue
        lines.append(f"\n## {task} (fixed-arch tuned Design B raw)\n")
        lines.append(
            f"- precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} "
            f"roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f} "
            f"(n={v['n']}, positives={v['n_positive']}, tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']})"
        )

    (DOCS_DIR / "class_pool_fixed_arch_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved model/history to {MODELS_DIR}, report to {DOCS_DIR / 'class_pool_fixed_arch_report.md'}")


if __name__ == "__main__":
    main()
