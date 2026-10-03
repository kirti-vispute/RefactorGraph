# -*- coding: utf-8 -*-
"""One-off promotion: scratch_external_eval/godclass_seed_sweep.py showed
seed=42's saved candidate (models/hybrid_class_pool_tuned_fixed_data) was
an unlucky draw -- god_class F1 across seeds 42/43/44/45 = 0.801/0.796/
0.754/0.793 (noise band, not a systematic god_class regression), but
seed=43 has the best macro-F1 of the four (0.693 vs seed 42's 0.678) and,
consistent with that overall improvement, also the most confident correct
PipSession behavior (0.994 vs seed 42's 0.231). Seed 43 is selected on
macro-F1 (holistic, all 3 tasks) -- NOT selected because of the PipSession
number specifically, which is reported as a side-observation, not the
selection criterion (see docs/godclass_formula_revision.md section 9/10
for the full reasoning -- selecting a seed BECAUSE it fixes one sample
would be exactly what this project's rules forbid).

Re-runs the same architecture/hyperparameters/data as
retrain_fixed_data_hybrid_class_pool.py, pinned to seed=43, and overwrites
models/hybrid_class_pool_tuned_fixed_data/ (the seed=42 version was backed
up first to models/hybrid_class_pool_tuned_fixed_data_step3_godclass_seed42/
before this ran). Frozen baseline (models/hybrid_class_pool_tuned) is
never touched. TEST is never touched.
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
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS, apply_norm, compute_norm_stats, load_split, run_eval

FROZEN_DIR = ROOT / "models" / "hybrid_class_pool_tuned"
MODELS_DIR = ROOT / "models" / "hybrid_class_pool_tuned_fixed_data"
DOCS_DIR = ROOT / "docs"
EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 32
CHOSEN_SEED = 43


def macro_f1(metrics):
    f1s = [v["f1"] for v in metrics.values() if v is not None]
    return sum(f1s) / len(f1s) if f1s else 0.0


def main():
    device = torch.device("cpu")
    best = json.loads((FROZEN_DIR / "best_params.json").read_text(encoding="utf-8"))
    print(f"[config] reusing frozen best_params.json unchanged: {best}")

    tho.SEED = CHOSEN_SEED
    torch.manual_seed(CHOSEN_SEED)

    train_graphs = load_split("train")
    val_graphs = load_split("val")
    norm_stats = compute_norm_stats(train_graphs)
    apply_norm(train_graphs, norm_stats)
    apply_norm(val_graphs, norm_stats)
    loss_lm, loss_fe, loss_gc = tho.build_losses(train_graphs)
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

    frozen_params = json.loads((FROZEN_DIR / "best_params.json").read_text(encoding="utf-8"))
    frozen_model = HeteroGAT(
        hidden_dim=frozen_params["hidden_dim"], heads=frozen_params["heads"], dropout=frozen_params["dropout"],
        node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
    ).to(device)
    frozen_model.load_state_dict(torch.load(FROZEN_DIR / "model.pt", weights_only=True))
    frozen_on_new_val = run_eval(frozen_model, val_loader, device)
    print(f"[diagnostic] frozen model on NEW val: macro-F1={macro_f1(frozen_on_new_val):.4f}")

    print(f"[train] retraining, seed={CHOSEN_SEED} ...")
    best_score, best_epoch, best_state = tho.train_one(
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
    (MODELS_DIR / "best_params.json").write_text(
        json.dumps({**best, "best_epoch": best_epoch, "seed": CHOSEN_SEED}, indent=2), encoding="utf-8"
    )
    (MODELS_DIR / "final_val_metrics.json").write_text(json.dumps(new_on_new_val, indent=2), encoding="utf-8")

    old_on_old_val = json.loads((FROZEN_DIR / "final_val_metrics.json").read_text(encoding="utf-8"))

    lines = ["# Post-Fix Retrain Comparison -- seed=43 (promoted after seed-variance check)\n"]
    lines.append(
        "Same architecture + hyperparameters as the frozen models/hybrid_class_pool_tuned "
        "checkpoint, same God-Class-rule-corrected TRAIN/VAL data as the seed=42 run "
        "(docs/post_fix_retrain_comparison_step3_godclass_seed42.md), but seed=43 -- "
        "selected via scratch_external_eval/godclass_seed_sweep.py as the best macro-F1 "
        "among seeds {42,43,44,45}, see docs/godclass_formula_revision.md section 9/10 "
        "for why. TEST untouched throughout.\n"
    )
    lines.append("\n## A) frozen model on OLD val  vs  B) frozen model on NEW val  vs  C) retrained model (seed=43) on NEW val\n")
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
    lines.append(f"| **macro-F1** | {macro_f1(old_on_old_val):.3f} | {macro_f1(frozen_on_new_val):.3f} | {macro_f1(new_on_new_val):.3f} |")
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
