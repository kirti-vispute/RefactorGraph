# -*- coding: utf-8 -*-
"""Phase 13: retrain the Phase 10 tuned hybrid architecture with the
class<-method message-passing fix.

Motivation (from the Phase 12 GNNExplainer audit): under the original
EDGE_TYPES, (class, contains, method) only ever makes method the
destination, so a class node's embedding can never aggregate information
from its own methods during message passing — god_class predictions see
only the class's own encoded metrics, module, and inherited-class chain.
ml/graph/graph_builder.py now also emits the reverse edge (method,
belongs_to, class), and ml/models/gat_baseline.EDGE_TYPES_CLASS_AGGREGATION
wires it in. See those two files' comments for the full rationale.

This script isolates that ONE change: it reuses the exact tuned
hyperparameters from models/hybrid_tuned/best_params.json (Phase 10 Optuna
search) — no re-tuning — so any metric delta versus models/hybrid_tuned/
final_val_metrics.json is attributable to the schema/architecture change,
not a hyperparameter difference. Same train/eval harness as Phase 9/10
(train_hybrid_baseline.py, tune_hybrid_optuna.py's train_one).

Trained on TRAIN split, evaluated on VAL split only. TEST split untouched —
data/processed/graphs_hybrid has no test/ directory (CodeBERT augmentation
for TEST is deferred to the final Phase 16 evaluation, same discipline as
every prior phase); this script never looks for one.

Output: models/hybrid_class_aggregation/{model.pt, norm_stats.pt,
final_val_metrics.json}, docs/class_aggregation_report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader

from ml.models.gat_baseline import EDGE_TYPES_CLASS_AGGREGATION, HeteroGAT
from scripts.train_hybrid_baseline import (
    HYBRID_NODE_DIMS,
    apply_norm,
    collect_task_labels,
    compute_norm_stats,
    load_split,
    pos_weight_for,
    run_eval,
)
from scripts.tune_hybrid_optuna import build_losses, train_one

ROOT = Path(__file__).resolve().parents[1]
TUNED_DIR = ROOT / "models" / "hybrid_tuned"
MODELS_DIR = ROOT / "models" / "hybrid_class_aggregation"
DOCS_DIR = ROOT / "docs"

BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 15
SEED = 42


def main():
    best_params_path = TUNED_DIR / "best_params.json"
    if not best_params_path.exists():
        raise SystemExit(f"{best_params_path} not found — run scripts/tune_hybrid_optuna.py first.")
    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
    print(f"[params] reusing Phase 10 tuned hyperparameters: {best_params}")

    torch.manual_seed(SEED)
    print("[load] reading hybrid graphs ...")
    train_graphs = load_split("train")
    val_graphs = load_split("val")
    print(f"[load] train={len(train_graphs)} val={len(val_graphs)}")

    for g in train_graphs + val_graphs:
        assert ("method", "belongs_to", "class") in g.edge_types or g["method"].x.shape[0] == 0, (
            "expected the regenerated hybrid graphs to carry the belongs_to edge — "
            "rerun scripts/build_graphs.py and scripts/augment_graphs_with_codebert.py first"
        )

    norm_stats = compute_norm_stats(train_graphs)
    apply_norm(train_graphs, norm_stats)
    apply_norm(val_graphs, norm_stats)

    loss_lm, loss_fe, loss_gc = build_losses(train_graphs)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

    device = torch.device("cpu")
    print(f"[train] full budget max_epochs={EPOCHS} patience={PATIENCE} "
          f"edge_types=EDGE_TYPES_CLASS_AGGREGATION ({len(EDGE_TYPES_CLASS_AGGREGATION)} relations)")
    best_score, best_epoch, best_state = train_one(
        best_params["hidden_dim"], best_params["heads"], best_params["dropout"],
        best_params["lr"], best_params["weight_decay"],
        train_loader, val_loader, loss_lm, loss_fe, loss_gc,
        device, EPOCHS, PATIENCE, trial=None, edge_types=EDGE_TYPES_CLASS_AGGREGATION,
    )
    print(f"[train] best val combined f1={best_score:.4f} at epoch {best_epoch}")

    model = HeteroGAT(hidden_dim=best_params["hidden_dim"], heads=best_params["heads"],
                       dropout=best_params["dropout"], node_feature_dims=HYBRID_NODE_DIMS,
                       edge_types=EDGE_TYPES_CLASS_AGGREGATION).to(device)
    model.load_state_dict(best_state)
    final_val_metrics = run_eval(model, val_loader, device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS_DIR / "model.pt")
    torch.save({nt: (m.tolist(), s.tolist()) for nt, (m, s) in norm_stats.items()},
                MODELS_DIR / "norm_stats.pt")
    (MODELS_DIR / "final_val_metrics.json").write_text(
        json.dumps(final_val_metrics, indent=2), encoding="utf-8"
    )
    (MODELS_DIR / "params.json").write_text(
        json.dumps({**best_params, "best_epoch": best_epoch,
                    "edge_types": [list(et) for et in EDGE_TYPES_CLASS_AGGREGATION]}, indent=2),
        encoding="utf-8",
    )

    tuned_metrics_path = TUNED_DIR / "final_val_metrics.json"
    tuned_metrics = json.loads(tuned_metrics_path.read_text(encoding="utf-8")) if tuned_metrics_path.exists() else None

    lines = ["# Class-Aggregation Architecture Fix Report (Phase 13)\n"]
    lines.append(
        "**Finding (Phase 12 GNNExplainer audit):** under the original schema, "
        "(class, contains, method) only ever makes `method` the destination — "
        "`class` is never the destination of any edge derived from its methods, "
        "so a class node's embedding can never aggregate information from its own "
        "methods during message passing. god_class predictions saw only the class's "
        "own encoded metrics, module, and inherited-class chain.\n"
    )
    lines.append(
        "**Fix:** the smallest schema change that closes this gap — a single reverse "
        "edge type, (method, belongs_to, class), added in ml/graph/graph_builder.py "
        "and wired in via ml/models/gat_baseline.EDGE_TYPES_CLASS_AGGREGATION. No "
        "other edge types were added, and no labels changed. The original EDGE_TYPES "
        "constant and every existing checkpoint (models/hybrid_tuned, "
        "models/hybrid_baseline, models/gat_baseline) are untouched and still load "
        "exactly as before — HeteroConv only wires convolutions for the edge types a "
        "model was constructed with, and ignores extra edge type keys in the data "
        "(verified against torch_geometric.nn.HeteroConv.forward before regenerating "
        "any graphs).\n"
    )
    lines.append(
        f"Reused the exact Phase 10 tuned hyperparameters (models/hybrid_tuned/"
        f"best_params.json: hidden_dim={best_params['hidden_dim']}, heads={best_params['heads']}, "
        f"dropout={best_params['dropout']:.4f}, lr={best_params['lr']:.2e}, "
        f"weight_decay={best_params['weight_decay']:.2e}) — no re-tuning, so any metric "
        f"delta below is attributable to the schema/architecture change alone, not a "
        f"hyperparameter difference. Trained on TRAIN split (full budget, "
        f"max_epochs={EPOCHS}, patience={PATIENCE}, best_epoch={best_epoch}), evaluated on "
        f"VAL split only. TEST split untouched.\n"
    )

    if tuned_metrics is not None:
        lines.append("\n## Comparison vs Phase 10 tuned hybrid (same val split, same hyperparameters)\n")
        lines.append("| task | Phase 10 F1 | Phase 13 F1 | delta F1 | Phase 10 precision | Phase 13 precision | Phase 10 recall | Phase 13 recall | Phase 10 ROC-AUC | Phase 13 ROC-AUC |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for task in ("long_method", "feature_envy", "god_class"):
            old = tuned_metrics.get(task)
            new = final_val_metrics.get(task)
            if old is None or new is None:
                lines.append(f"| {task} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            delta = new["f1"] - old["f1"]
            lines.append(
                f"| {task} | {old['f1']:.3f} | {new['f1']:.3f} | {delta:+.3f} | "
                f"{old['precision']:.3f} | {new['precision']:.3f} | "
                f"{old['recall']:.3f} | {new['recall']:.3f} | "
                f"{old['roc_auc']:.3f} | {new['roc_auc']:.3f} |"
            )
    else:
        lines.append("\n(models/hybrid_tuned/final_val_metrics.json not found — no comparison table.)\n")

    for task, v in final_val_metrics.items():
        if v is None:
            lines.append(f"\n## {task} (Phase 13 raw)\n\n- no eligible nodes in val split\n")
            continue
        lines.append(f"\n## {task} (Phase 13 raw)\n")
        lines.append(
            f"- precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} "
            f"roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f} "
            f"(n={v['n']}, positives={v['n_positive']}, tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']})"
        )

    (DOCS_DIR / "class_aggregation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved model/metrics to {MODELS_DIR}, report to {DOCS_DIR / 'class_aggregation_report.md'}")


if __name__ == "__main__":
    main()
