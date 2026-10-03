# -*- coding: utf-8 -*-
"""Phase 13c: Optuna search scoped to Design B (HeteroGAT(class_method_pool=
True), see ml/models/gat_baseline.py and docs/class_pool_report.md).
Architecture is fixed to Design B for every trial — this phase does not
re-compare architectures, it tunes one already selected per the controlled
Phase 13b experiment (Design B beat Design A on every task on the same val
split, same Phase 10 hyperparameters).

Motivation (from the Phase 13b comparison): Design B's forward pass is
proven to leave the method/long_method/feature_envy pathway byte-for-byte
identical to Phase 10 (tests/test_gat_baseline.py::
test_class_pool_does_not_alter_method_pathway) — yet long_method and
feature_envy still regressed versus Phase 10 under Design B. That residual
regression isn't a forward-pass leak; it's shared-encoder/multi-task
gradient interference: `loss = loss_lm + loss_fe + loss_gc` is one joint sum
backpropagated through one optimizer, so god_class's gradient still updates
the same shared method-encoder/conv weights that long_method/feature_envy
depend on. Loss weighting is the direct lever for that: down-weighting
god_class's contribution (or up-weighting the starved feature_envy task)
changes how much of the shared backbone's gradient budget each task claims,
without touching the forward-pass architecture again.

Search space extends Phase 10/13b's (heads, hidden_dim_per_head, dropout,
lr, weight_decay) with weight_fe and weight_gc — multipliers on
loss_fe/loss_gc in the joint sum. weight_lm is left fixed at 1.0: only the
two multipliers relative to loss_lm matter (scaling all three equally is a
no-op on gradient direction), so fixing one anchors the search instead of
wasting trials on a redundant degree of freedom. TPE sampler (seed=42) +
median pruner on the per-epoch macro-F1 (mean of long_method/feature_envy/
god_class val F1), same objective metric as Phase 10.

Only TRAIN and VAL splits are touched (VAL for the search objective/pruning
and the final report). TEST split is never read — data/processed/
graphs_hybrid has no test/ directory and this script never looks for one.
Labels, graph construction, and the train/val/test split are untouched.

After the search: retrains the winning trial's full config (architecture +
weight_fe/weight_gc) with the full Phase 9/10 budget (max_epochs=150,
patience=15) for a clean final report, then evaluates on VAL and compares
per-task F1 + macro-F1 against Phase 10 hybrid (models/hybrid_tuned) and
untuned Design B (models/hybrid_class_pool, weight_fe=weight_gc=1.0).

Output: models/hybrid_class_pool_tuned/{model.pt, norm_stats.pt,
best_params.json, final_val_metrics.json, optuna_trials.csv},
docs/class_pool_tuning_report.md.
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
from scripts.tune_hybrid_optuna import build_losses, train_one

ROOT = Path(__file__).resolve().parents[1]
TUNED_DIR = ROOT / "models" / "hybrid_tuned"
CLASS_POOL_DIR = ROOT / "models" / "hybrid_class_pool"
MODELS_DIR = ROOT / "models" / "hybrid_class_pool_tuned"
DOCS_DIR = ROOT / "docs"

EPOCHS = 150
PATIENCE = 15
SEARCH_EPOCHS = 60
SEARCH_PATIENCE = 10
BATCH_SIZE = 32
SEED = 42
N_TRIALS = 30


def build_objective(train_graphs, val_graphs, loss_lm, loss_fe, loss_gc, device,
                     search_epochs: int = SEARCH_EPOCHS, search_patience: int = SEARCH_PATIENCE,
                     batch_size: int = BATCH_SIZE, fixed_heads: int = None,
                     fixed_hidden_dim_per_head: int = None):
    """Factory so the objective can be unit-tested against tiny synthetic
    graphs/short budgets without touching disk or running a real search.

    fixed_heads/fixed_hidden_dim_per_head (Phase 13e): when given, pin the
    architecture instead of sampling it, so a search can isolate the
    dropout/lr/weight_decay/weight_fe/weight_gc effect at one fixed
    capacity — added after the Phase 13c free search showed long_method's
    best per-trial F1 (trial 9, 0.804) landed at heads=4/hidden_dim=128
    (Phase 10's own architecture), while the macro-F1-winning trial used a
    smaller heads=2/hidden_dim=64 that traded long_method capacity for
    feature_envy/god_class gains. Default (None) preserves the original
    free-search behavior byte-for-byte."""

    def objective(trial):
        heads = fixed_heads if fixed_heads is not None else trial.suggest_categorical("heads", [1, 2, 4])
        hidden_per_head = (fixed_hidden_dim_per_head if fixed_hidden_dim_per_head is not None
                            else trial.suggest_categorical("hidden_dim_per_head", [8, 16, 32]))
        hidden_dim = heads * hidden_per_head
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        weight_fe = trial.suggest_float("weight_fe", 0.5, 4.0)
        weight_gc = trial.suggest_float("weight_gc", 0.5, 4.0)

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

        best_score, _, best_state = train_one(
            hidden_dim, heads, dropout, lr, weight_decay,
            train_loader, val_loader, loss_lm, loss_fe, loss_gc,
            device, search_epochs, search_patience, trial=trial,
            edge_types=None, class_method_pool=True,
            weight_lm=1.0, weight_fe=weight_fe, weight_gc=weight_gc,
        )

        # Phase 13d: log per-task F1 at this trial's best epoch as user
        # attributes — trials_dataframe() only carries the scalar objective
        # (macro-F1) by default, which hides whether a trial's macro-F1 came
        # from balanced gains or from one task's gain masking another's
        # loss. Reconstructing from best_state and re-running run_eval is a
        # single extra val-loader pass (cheap relative to the training
        # loop) and is exact for this trial's best epoch, since best_state
        # is the literal state_dict saved at that epoch inside train_one.
        if best_state is not None:
            eval_model = HeteroGAT(hidden_dim=hidden_dim, heads=heads, dropout=dropout,
                                    node_feature_dims=HYBRID_NODE_DIMS, edge_types=None,
                                    class_method_pool=True).to(device)
            eval_model.load_state_dict(best_state)
            metrics = run_eval(eval_model, val_loader, device)
            for task, v in metrics.items():
                trial.set_user_attr(f"{task}_f1", v["f1"] if v is not None else None)
            trial.set_user_attr("macro_f1", best_score)

        return best_score

    return objective


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

    objective = build_objective(train_graphs, val_graphs, loss_lm, loss_fe, loss_gc, device)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"[search done] best macro val f1={study.best_value:.4f} params={study.best_params}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(MODELS_DIR / "optuna_trials.csv", index=False)

    best = study.best_params
    heads = best["heads"]
    hidden_dim = heads * best["hidden_dim_per_head"]
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    best_score, best_epoch, best_state = train_one(
        hidden_dim, heads, best["dropout"], best["lr"], best["weight_decay"],
        train_loader, val_loader, loss_lm, loss_fe, loss_gc,
        device, EPOCHS, PATIENCE, trial=None, edge_types=None, class_method_pool=True,
        weight_lm=1.0, weight_fe=best["weight_fe"], weight_gc=best["weight_gc"],
    )

    model = HeteroGAT(hidden_dim=hidden_dim, heads=heads, dropout=best["dropout"],
                       node_feature_dims=HYBRID_NODE_DIMS, edge_types=None,
                       class_method_pool=True).to(device)
    model.load_state_dict(best_state)
    final_val_metrics = run_eval(model, val_loader, device)

    torch.save(best_state, MODELS_DIR / "model.pt")
    torch.save({nt: (m.tolist(), s.tolist()) for nt, (m, s) in norm_stats.items()},
                MODELS_DIR / "norm_stats.pt")
    (MODELS_DIR / "best_params.json").write_text(json.dumps({
        "heads": heads, "hidden_dim": hidden_dim, "dropout": best["dropout"],
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

    def macro_f1(metrics):
        if not metrics:
            return None
        f1s = [v["f1"] for v in metrics.values() if v is not None]
        return sum(f1s) / len(f1s) if f1s else None

    lines = ["# Class-Method-Pool Optuna Tuning Report (Phase 13c, Design B)\n"]
    lines.append(
        f"TPE sampler + median pruner, seed={SEED}, {N_TRIALS} trials, search budget "
        f"max_epochs={SEARCH_EPOCHS}/patience={SEARCH_PATIENCE} per trial. Architecture fixed "
        f"to Design B (class_method_pool=True) for every trial — see docs/class_pool_report.md "
        f"for why Design B was selected over Design A. Search space: heads in {{1,2,4}}, "
        f"hidden_dim_per_head in {{8,16,32}}, dropout in [0.0, 0.5], lr log-uniform "
        f"[1e-4, 1e-2], weight_decay log-uniform [1e-6, 1e-2], weight_fe in [0.5, 4.0], "
        f"weight_gc in [0.5, 4.0] (weight_lm fixed at 1.0 — only the two ratios relative to "
        f"loss_lm matter for gradient direction). Objective: macro-F1 (mean val F1 across the "
        f"3 tasks). Winning config retrained with the full budget (max_epochs={EPOCHS}, "
        f"patience={PATIENCE}) for the numbers below.\n"
        f"\nOnly TRAIN and VAL splits are touched. TEST split remains untouched.\n"
        f"\nBest params: heads={heads}, hidden_dim={hidden_dim}, dropout={best['dropout']:.3f}, "
        f"lr={best['lr']:.2e}, weight_decay={best['weight_decay']:.2e}, "
        f"weight_fe={best['weight_fe']:.3f}, weight_gc={best['weight_gc']:.3f}, "
        f"best_epoch={best_epoch}.\n"
    )

    if tuned_metrics is not None and pool_metrics is not None:
        lines.append("\n## Comparison: Phase 10 hybrid vs untuned Design B vs tuned Design B (same val split)\n")
        header = "| task | Phase 10 F1 | Design B (untuned) F1 | Design B (tuned) F1 | untuned->tuned |"
        lines.append(header)
        lines.append("|" + "---|" * (header.count("|") - 1))
        for task in ("long_method", "feature_envy", "god_class"):
            old = tuned_metrics.get(task)
            pool = pool_metrics.get(task)
            tuned = final_val_metrics.get(task)
            if old is None or pool is None or tuned is None:
                lines.append(f"| {task} | n/a | n/a | n/a | n/a |")
                continue
            delta = tuned["f1"] - pool["f1"]
            lines.append(
                f"| {task} | {old['f1']:.3f} | {pool['f1']:.3f} | {tuned['f1']:.3f} | {delta:+.3f} |"
            )
        macro_old, macro_pool, macro_tuned = macro_f1(tuned_metrics), macro_f1(pool_metrics), macro_f1(final_val_metrics)
        lines.append(
            f"| **macro-F1** | {macro_old:.3f} | {macro_pool:.3f} | {macro_tuned:.3f} | "
            f"{(macro_tuned - macro_pool):+.3f} |"
        )
    else:
        lines.append("\n(comparison baselines not found — no comparison table.)\n")

    for task, v in final_val_metrics.items():
        if v is None:
            lines.append(f"\n## {task} (tuned Design B raw)\n\n- no eligible nodes in val split\n")
            continue
        lines.append(f"\n## {task} (tuned Design B raw)\n")
        lines.append(
            f"- precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} "
            f"roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f} "
            f"(n={v['n']}, positives={v['n_positive']}, tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']})"
        )

    (DOCS_DIR / "class_pool_tuning_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved model/history to {MODELS_DIR}, report to {DOCS_DIR / 'class_pool_tuning_report.md'}")


if __name__ == "__main__":
    main()
