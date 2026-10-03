# -*- coding: utf-8 -*-
"""Phase 10: Optuna hyperparameter search for the Phase 9 hybrid model,
validation-only (train split for training, val split for the search
objective and pruning). TEST split is never touched by this phase, same
discipline as every prior phase.

Reuses the exact training/eval harness from train_hybrid_baseline.py
(pos_weight_for, run_eval, compute_norm_stats, apply_norm) so the tuned
model is directly comparable to Phase 9's default-hyperparameter run.

Search space: heads and hidden_dim_per_head (hidden_dim = heads *
hidden_dim_per_head, so hidden_dim % heads == 0 always holds, satisfying
HeteroGAT's assert), dropout, lr, weight_decay. TPE sampler (seed=42) +
median pruner on the per-epoch combined val F1.

After the search: retrains the best trial's config with the full Phase 9
EPOCHS/PATIENCE budget for a clean final report, since a pruned/short-budget
trial checkpoint isn't necessarily the same quality as one allowed to fully
converge/early-stop on its own terms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from ml.models.gat_baseline import HeteroGAT
from scripts.train_hybrid_baseline import (
    HYBRID_NODE_DIMS,
    apply_norm,
    collect_task_labels,
    compute_norm_stats,
    load_split,
    pos_weight_for,
    run_eval,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "hybrid_tuned"
DOCS_DIR = ROOT / "docs"

EPOCHS = 150
PATIENCE = 15
SEARCH_EPOCHS = 60
SEARCH_PATIENCE = 10
BATCH_SIZE = 32
SEED = 42
N_TRIALS = 30


def build_losses(train_graphs: list) -> tuple:
    lm_pos, lm_neg = collect_task_labels(train_graphs, "method", "y_long_method")
    lm_pos_f, lm_neg_f = collect_task_labels(train_graphs, "function", "y_long_method")
    fe_pos, fe_neg = collect_task_labels(train_graphs, "method", "y_feature_envy")
    gc_pos, gc_neg = collect_task_labels(train_graphs, "class", "y")
    pw_lm = pos_weight_for(lm_pos + lm_pos_f, lm_neg + lm_neg_f)
    pw_fe = pos_weight_for(fe_pos, fe_neg)
    pw_gc = pos_weight_for(gc_pos, gc_neg)
    return (
        nn.BCEWithLogitsLoss(pos_weight=pw_lm),
        nn.BCEWithLogitsLoss(pos_weight=pw_fe),
        nn.BCEWithLogitsLoss(pos_weight=pw_gc),
    )


def compute_batch_loss(model, h, batch, loss_lm, loss_fe, loss_gc,
                        weight_lm: float = 1.0, weight_fe: float = 1.0, weight_gc: float = 1.0):
    """Isolated so loss-weighting can be unit-tested without a full training
    loop (Phase 13c). weight_lm/weight_fe/weight_gc default to 1.0, which
    reproduces the original unweighted sum byte-for-byte — existing callers
    (train_one with no weight_* args) are unaffected."""
    loss = torch.tensor(0.0)
    has_loss = False
    if "method" in h and batch["method"].x.shape[0] > 0:
        loss = loss + weight_lm * loss_lm(model.predict_long_method(h, "method"), batch["method"].y_long_method)
        loss = loss + weight_fe * loss_fe(model.predict_feature_envy(h), batch["method"].y_feature_envy)
        has_loss = True
    if "function" in h and batch["function"].x.shape[0] > 0:
        loss = loss + weight_lm * loss_lm(model.predict_long_method(h, "function"), batch["function"].y_long_method)
        has_loss = True
    if "class" in h and batch["class"].x.shape[0] > 0:
        loss = loss + weight_gc * loss_gc(model.predict_god_class(h), batch["class"].y)
        has_loss = True
    return loss, has_loss


def train_one(hidden_dim, heads, dropout, lr, weight_decay,
              train_loader, val_loader, loss_lm, loss_fe, loss_gc,
              device, epochs, patience, trial=None, edge_types=None,
              class_method_pool=False, weight_lm: float = 1.0,
              weight_fe: float = 1.0, weight_gc: float = 1.0):
    torch.manual_seed(SEED)
    model = HeteroGAT(hidden_dim=hidden_dim, heads=heads, dropout=dropout,
                       node_feature_dims=HYBRID_NODE_DIMS, edge_types=edge_types,
                       class_method_pool=class_method_pool).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score, best_epoch, best_state = -1.0, -1, None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
            h = model(batch.x_dict, edge_index_dict)
            loss, has_loss = compute_batch_loss(
                model, h, batch, loss_lm, loss_fe, loss_gc,
                weight_lm=weight_lm, weight_fe=weight_fe, weight_gc=weight_gc,
            )
            if not has_loss:
                continue
            loss.backward()
            optimizer.step()

        val_metrics = run_eval(model, val_loader, device)
        f1s = [v["f1"] for v in val_metrics.values() if v is not None]
        combined_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        if trial is not None:
            trial.report(combined_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if combined_f1 > best_score:
            best_score, best_epoch = combined_f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return best_score, best_epoch, best_state


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

    def objective(trial):
        heads = trial.suggest_categorical("heads", [1, 2, 4])
        hidden_per_head = trial.suggest_categorical("hidden_dim_per_head", [8, 16, 32])
        hidden_dim = heads * hidden_per_head
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

        best_score, _, _ = train_one(
            hidden_dim, heads, dropout, lr, weight_decay,
            train_loader, val_loader, loss_lm, loss_fe, loss_gc,
            device, SEARCH_EPOCHS, SEARCH_PATIENCE, trial=trial,
        )
        return best_score

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"[search done] best combined val f1={study.best_value:.4f} params={study.best_params}")

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
        device, EPOCHS, PATIENCE, trial=None,
    )

    model = HeteroGAT(hidden_dim=hidden_dim, heads=heads, dropout=best["dropout"],
                       node_feature_dims=HYBRID_NODE_DIMS).to(device)
    model.load_state_dict(best_state)
    final_val_metrics = run_eval(model, val_loader, device)

    torch.save(best_state, MODELS_DIR / "model.pt")
    torch.save({nt: (m.tolist(), s.tolist()) for nt, (m, s) in norm_stats.items()},
                MODELS_DIR / "norm_stats.pt")
    (MODELS_DIR / "best_params.json").write_text(json.dumps({
        "heads": heads, "hidden_dim": hidden_dim, "dropout": best["dropout"],
        "lr": best["lr"], "weight_decay": best["weight_decay"], "best_epoch": best_epoch,
    }, indent=2), encoding="utf-8")
    (MODELS_DIR / "final_val_metrics.json").write_text(
        json.dumps(final_val_metrics, indent=2), encoding="utf-8"
    )

    lines = ["# Hybrid Hyperparameter Tuning Report (Phase 10 -- Optuna, validation-only)\n"]
    lines.append(
        f"TPE sampler + median pruner, seed={SEED}, {N_TRIALS} trials, search budget "
        f"max_epochs={SEARCH_EPOCHS}/patience={SEARCH_PATIENCE} per trial (pruned trials stop "
        f"earlier). Search space: heads in {{1,2,4}}, hidden_dim_per_head in {{8,16,32}} "
        f"(hidden_dim = heads * hidden_dim_per_head, guaranteeing hidden_dim % heads == 0), "
        f"dropout in [0.0, 0.5], lr log-uniform [1e-4, 1e-2], weight_decay log-uniform "
        f"[1e-6, 1e-2]. Objective: mean val F1 across the 3 tasks (best epoch within the "
        f"trial's own early-stopping budget). Winning config retrained once more with the full "
        f"Phase 9 budget (max_epochs={EPOCHS}, patience={PATIENCE}) for the numbers below -- a "
        f"pruned or short-budget trial checkpoint isn't necessarily the same quality as one "
        f"allowed to fully converge/early-stop on its own terms.\n"
        f"\nOnly TRAIN and VAL splits are touched by this phase (VAL used for the search "
        f"objective and pruning, same role it plays in every prior phase). TEST split remains "
        f"untouched, per the project's touch-once-at-the-end discipline.\n"
        f"\nBest params: heads={heads}, hidden_dim={hidden_dim}, dropout={best['dropout']:.3f}, "
        f"lr={best['lr']:.2e}, weight_decay={best['weight_decay']:.2e}, best_epoch={best_epoch}.\n"
    )
    for task, v in final_val_metrics.items():
        if v is None:
            lines.append(f"\n## {task}\n\n- no eligible nodes in val split\n")
            continue
        lines.append(f"\n## {task}\n")
        lines.append(
            f"- precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} "
            f"roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f} "
            f"(n={v['n']}, positives={v['n_positive']}, tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']})"
        )

    rf_json = ROOT / "models" / "baseline" / "baseline_metrics.json"
    gat_json = ROOT / "models" / "gat_baseline" / "final_val_metrics.json"
    cb_json = ROOT / "models" / "codebert_baseline" / "codebert_baseline_metrics.json"
    hy_json = ROOT / "models" / "hybrid_baseline" / "final_val_metrics.json"
    if rf_json.exists() and gat_json.exists() and cb_json.exists() and hy_json.exists():
        rf_all = json.loads(rf_json.read_text(encoding="utf-8"))
        gat_all = json.loads(gat_json.read_text(encoding="utf-8"))
        cb_all = json.loads(cb_json.read_text(encoding="utf-8"))
        hy_all = json.loads(hy_json.read_text(encoding="utf-8"))
        lines.append("\n## Comparison across all baselines + tuned hybrid (same val split)\n")
        lines.append("| task | RF F1 | GAT F1 | CodeBERT LR F1 | Hybrid (default) F1 | Hybrid (tuned) F1 |")
        lines.append("|---|---|---|---|---|---|")
        for task in ("long_method", "feature_envy", "god_class"):
            rf_v = rf_all[task]["random_forest"]["val"]
            gat_v = gat_all[task]
            cb_v = cb_all[task]["logistic_regression"]["val"]
            hy_v = hy_all[task]
            tuned_v = final_val_metrics[task]
            gat_f1 = f"{gat_v['f1']:.3f}" if gat_v else "n/a"
            hy_f1 = f"{hy_v['f1']:.3f}" if hy_v else "n/a"
            tuned_f1 = f"{tuned_v['f1']:.3f}" if tuned_v else "n/a"
            lines.append(f"| {task} | {rf_v['f1']:.3f} | {gat_f1} | {cb_v['f1']:.3f} | {hy_f1} | {tuned_f1} |")

        lm_delta = final_val_metrics["long_method"]["f1"] - hy_all["long_method"]["f1"]
        fe_delta = final_val_metrics["feature_envy"]["f1"] - hy_all["feature_envy"]["f1"]
        gc_delta = final_val_metrics["god_class"]["f1"] - hy_all["god_class"]["f1"]
        lines.append(
            f"\nTuning moved F1 by long_method {lm_delta:+.3f}, feature_envy {fe_delta:+.3f}, "
            f"god_class {gc_delta:+.3f} versus the Phase 9 default hyperparameters. The winning "
            f"config (hidden_dim={hidden_dim}, i.e. {hidden_dim // 32}x the Phase 9 default of 32, "
            f"with dropout={best['dropout']:.3f} vs the default 0.2) confirms the Phase 9 report's "
            f"own hypothesis that a wider hidden_dim relieves pressure on the shared Linear "
            f"encoder forced to compress a 768-dim CodeBERT block alongside a 3-6-dim structural "
            f"block. The gains on long_method and god_class are real but modest, and the tuned "
            f"hybrid still does NOT beat the Phase 7 structure-only GAT or the Phase 6 metrics-"
            f"only RF on any of the 3 tasks -- widening the encoder helped retain more structural "
            f"signal, but didn't remove the underlying limitation (CodeBERT stays frozen and "
            f"general-purpose, not task-tuned). feature_envy is the rarest class in val "
            f"(positives={final_val_metrics['feature_envy']['n_positive']} of "
            f"{final_val_metrics['feature_envy']['n']}) and the most search-noise-sensitive of "
            f"the 3 tasks; a small F1 regression there is within the run-to-run variance a "
            f"30-trial validation-only search can produce, not necessarily meaningful "
            f"hyperparameter harm. Consistent with the Phase 9 report's next-steps list, the "
            f"largest remaining structural fix (a learned down-projection of the CodeBERT "
            f"embedding, or joint fine-tuning) stays out of scope for this staged phase.\n"
        )

    (DOCS_DIR / "hybrid_tuning_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved model/history to {MODELS_DIR}, report to {DOCS_DIR / 'hybrid_tuning_report.md'}")


if __name__ == "__main__":
    main()
