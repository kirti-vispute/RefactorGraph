# -*- coding: utf-8 -*-
"""Robustness-audit fix retrain (docs/robustness_audit_phase_a.md).

Retrains Design B (class_method_pool=True) with the EXACT same architecture
and hyperparameters as the currently-frozen best config
(models/hybrid_class_pool_tuned/best_params.json), on the graphs rebuilt by
scripts/rebuild_train_val_data.py + scripts/augment_graphs_with_codebert.py
after the parser/graph fixes (nested function/class representation, scoped
call resolution, external_access_count double-count fix).

Architecture/hyperparameters are held IDENTICAL to the frozen checkpoint on
purpose -- this isolates ONE variable (data/label/graph correctness) rather
than conflating a data change with an architecture or hyperparameter change
in the same experiment (per the project's "every architectural change needs
a controlled comparison" rule). A fresh hyperparameter search on the
corrected data is legitimate future work, not done here.

Because the VAL set's own labels and node population changed (corrected
Feature Envy metric, previously-invisible nested-function/class nodes now
present), a raw macro-F1 comparison against the frozen model's OLD
final_val_metrics.json is not a clean apples-to-apples comparison by
itself -- the evaluation set itself changed. This script therefore reports
THREE numbers per task, to triangulate honestly:

  A) OLD model (frozen weights) on OLD val    -- already recorded in
     models/hybrid_class_pool_tuned/final_val_metrics.json
  B) OLD model (frozen weights) on NEW val    -- a cheap diagnostic re-eval:
     holds the model constant, isolates what changed about val itself
  C) NEW model (retrained on new data) on NEW val -- the real result of
     this fix + retrain

Output written to a NEW directory (models/hybrid_class_pool_tuned_fixed_data/)
-- the frozen models/hybrid_class_pool_tuned/ checkpoint is never touched or
overwritten (it is not under git; the master prompt requires it stay frozen
until a deliberate final decision is made).

Only TRAIN and VAL are read. TEST is never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
FROZEN_DIR = ROOT / "models" / "hybrid_class_pool_tuned"
MODELS_DIR = ROOT / "models" / "hybrid_class_pool_tuned_fixed_data"
DOCS_DIR = ROOT / "docs"

EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 32
SEED = 42


def macro_f1(metrics: dict) -> float:
    f1s = [v["f1"] for v in metrics.values() if v is not None]
    return sum(f1s) / len(f1s) if f1s else 0.0


def main():
    device = torch.device("cpu")
    torch.manual_seed(SEED)

    best = json.loads((FROZEN_DIR / "best_params.json").read_text(encoding="utf-8"))
    print(f"[config] reusing frozen best_params.json unchanged: {best}")

    print("[load] reading rebuilt hybrid graphs (train/val only) ...")
    train_graphs = load_split("train")
    val_graphs = load_split("val")
    print(f"[load] train={len(train_graphs)} val={len(val_graphs)}")

    norm_stats = compute_norm_stats(train_graphs)
    apply_norm(train_graphs, norm_stats)
    apply_norm(val_graphs, norm_stats)

    loss_lm, loss_fe, loss_gc = build_losses(train_graphs)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

    # --- (B) frozen weights, evaluated on the NEW (corrected) val set ---
    frozen_params = json.loads((FROZEN_DIR / "best_params.json").read_text(encoding="utf-8"))
    frozen_model = HeteroGAT(
        hidden_dim=frozen_params["hidden_dim"], heads=frozen_params["heads"], dropout=frozen_params["dropout"],
        node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
    ).to(device)
    frozen_model.load_state_dict(torch.load(FROZEN_DIR / "model.pt", weights_only=True))
    frozen_on_new_val = run_eval(frozen_model, val_loader, device)
    print(f"[diagnostic] frozen model on NEW val: macro-F1={macro_f1(frozen_on_new_val):.4f}")

    # --- (C) retrain with identical hyperparameters on the fixed data ---
    print("[train] retraining with frozen best_params.json on fixed data ...")
    best_score, best_epoch, best_state = train_one(
        best["hidden_dim"], best["heads"], best["dropout"],
        best["lr"], best["weight_decay"],
        train_loader, val_loader, loss_lm, loss_fe, loss_gc,
        device, EPOCHS, PATIENCE, trial=None, edge_types=None, class_method_pool=True,
        weight_lm=best.get("weight_lm", 1.0), weight_fe=best.get("weight_fe", 1.0), weight_gc=best.get("weight_gc", 1.0),
    )
    print(f"[train] done: best_epoch={best_epoch} best_val_macro_f1={best_score:.4f}")

    new_model = HeteroGAT(
        hidden_dim=best["hidden_dim"], heads=best["heads"], dropout=best["dropout"],
        node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
    ).to(device)
    new_model.load_state_dict(best_state)
    new_on_new_val = run_eval(new_model, val_loader, device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS_DIR / "model.pt")
    torch.save({nt: (m.tolist(), s.tolist()) for nt, (m, s) in norm_stats.items()}, MODELS_DIR / "norm_stats.pt")
    (MODELS_DIR / "best_params.json").write_text(json.dumps({**best, "best_epoch": best_epoch}, indent=2), encoding="utf-8")
    (MODELS_DIR / "final_val_metrics.json").write_text(json.dumps(new_on_new_val, indent=2), encoding="utf-8")

    old_on_old_val = json.loads((FROZEN_DIR / "final_val_metrics.json").read_text(encoding="utf-8"))

    lines = ["# Post-Fix Retrain Comparison (docs/robustness_audit_phase_a.md)\n"]
    lines.append(
        "Same architecture + hyperparameters as the frozen "
        "models/hybrid_class_pool_tuned checkpoint, retrained on the graphs "
        "rebuilt after the parser/graph fixes. TEST untouched throughout.\n"
    )
    lines.append(
        "\n## A) frozen model on OLD val  vs  B) frozen model on NEW val  vs  C) retrained model on NEW val\n"
    )
    header = "| task | A: old model / old val | B: old model / new val | C: new model / new val |"
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    for task in ("long_method", "feature_envy", "god_class"):
        a = old_on_old_val.get(task)
        b = frozen_on_new_val.get(task)
        c = new_on_new_val.get(task)
        row = [task]
        for v in (a, b, c):
            row.append(f"{v['f1']:.3f}" if v else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append(
        f"| **macro-F1** | {macro_f1(old_on_old_val):.3f} | {macro_f1(frozen_on_new_val):.3f} | {macro_f1(new_on_new_val):.3f} |"
    )
    lines.append(
        "\nA vs B isolates what changed about VAL itself (labels/graph "
        "population) with the model held constant. B vs C isolates whether "
        "retraining on the corrected data helps, on the SAME (new) "
        "evaluation set -- this is the fair, single-variable comparison. "
        "A vs C is NOT a controlled comparison (both the model and the "
        "evaluation set changed) and should not be read as \"the fix "
        "improved/hurt performance by X\" on its own.\n"
    )
    for task, v in new_on_new_val.items():
        if v is None:
            lines.append(f"\n## {task} (new model, new val, raw)\n\n- no eligible nodes\n")
            continue
        lines.append(f"\n## {task} (new model, new val, raw)\n")
        lines.append(
            f"- precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} "
            f"roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f} "
            f"(n={v['n']}, positives={v['n_positive']}, tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']})"
        )

    (DOCS_DIR / "post_fix_retrain_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved model/history to {MODELS_DIR}, report to {DOCS_DIR / 'post_fix_retrain_comparison.md'}")
    print("Frozen checkpoint at", FROZEN_DIR, "was NOT modified.")


if __name__ == "__main__":
    main()
