# -*- coding: utf-8 -*-
"""Phase 13b: Design B for the god_class receptive-field fix — a dedicated
class<-method pooling path for the god_class head only, instead of adding an
edge type to the shared 2-layer GAT backbone (Design A,
scripts/train_hybrid_class_aggregation.py).

Why Design A alone isn't enough: EDGE_TYPES_CLASS_AGGREGATION closed the
god_class receptive-field gap but regressed long_method AND feature_envy too
on the same val split, same hyperparameters (see docs/class_aggregation_
report.md) — not just god_class, which is the one task that edge was meant
to help. Root cause (ml/models/gat_baseline.py, see the comment above
CLASS_METHOD_POOL): (class, contains, method) is method's ONLY incoming edge
type, so at conv2 every method reads its own class's embedding back — and
once belongs_to lets that class embedding absorb every sibling method's
conv1 output, conv2 folds that sibling-aggregate signal into each method's
own embedding, which head_long_method and head_feature_envy also read. The
three tasks were never supposed to share that coupling.

Design B (HeteroGAT(class_method_pool=True)) avoids the coupling entirely:
the shared backbone stays on the plain EDGE_TYPES (no belongs_to edge is
wired into conv1/conv2 at all), and after conv2 the existing (class,
contains, method) edge_index — already present in every graph, no schema
change needed — is used to mean-pool method embeddings into a per-class
vector fed ONLY to head_god_class (concatenated with class's own conv2
embedding). method/function embeddings are structurally untouched, so
long_method/feature_envy are guaranteed byte-for-byte identical to Phase 10
(see tests/test_gat_baseline.py::test_class_pool_does_not_alter_method_
pathway) — the interference this script measures on long_method/feature_envy
should be ~0 by construction; this run's job is to confirm that empirically
and to see whether god_class alone benefits versus Phase 10 and versus
Design A.

Same reuse-Phase-10-hyperparameters discipline as Design A: hidden_dim,
heads, dropout, lr, weight_decay all taken from models/hybrid_tuned/
best_params.json, no re-tuning, so any metric delta is attributable to the
architecture change alone. Trained on TRAIN split, evaluated on VAL split
only. TEST split untouched — data/processed/graphs_hybrid has no test/
directory; this script never looks for one.

Output: models/hybrid_class_pool/{model.pt, norm_stats.pt,
final_val_metrics.json, params.json}, docs/class_pool_report.md.
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
TUNED_DIR = ROOT / "models" / "hybrid_tuned"
CLASS_AGG_DIR = ROOT / "models" / "hybrid_class_aggregation"
MODELS_DIR = ROOT / "models" / "hybrid_class_pool"
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

    norm_stats = compute_norm_stats(train_graphs)
    apply_norm(train_graphs, norm_stats)
    apply_norm(val_graphs, norm_stats)

    loss_lm, loss_fe, loss_gc = build_losses(train_graphs)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

    device = torch.device("cpu")
    print(f"[train] full budget max_epochs={EPOCHS} patience={PATIENCE} "
          f"edge_types=EDGE_TYPES (backbone unchanged) class_method_pool=True")
    best_score, best_epoch, best_state = train_one(
        best_params["hidden_dim"], best_params["heads"], best_params["dropout"],
        best_params["lr"], best_params["weight_decay"],
        train_loader, val_loader, loss_lm, loss_fe, loss_gc,
        device, EPOCHS, PATIENCE, trial=None, edge_types=None, class_method_pool=True,
    )
    print(f"[train] best val combined f1={best_score:.4f} at epoch {best_epoch}")

    model = HeteroGAT(hidden_dim=best_params["hidden_dim"], heads=best_params["heads"],
                       dropout=best_params["dropout"], node_feature_dims=HYBRID_NODE_DIMS,
                       edge_types=None, class_method_pool=True).to(device)
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
        json.dumps({**best_params, "best_epoch": best_epoch, "class_method_pool": True}, indent=2),
        encoding="utf-8",
    )

    def _load(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    tuned_metrics = _load(TUNED_DIR / "final_val_metrics.json")
    class_agg_metrics = _load(CLASS_AGG_DIR / "final_val_metrics.json")

    lines = ["# Class-Method-Pool Architecture Fix Report (Phase 13b, Design B)\n"]
    lines.append(
        "Design A (EDGE_TYPES_CLASS_AGGREGATION, docs/class_aggregation_report.md) fixed "
        "god_class's receptive field but regressed long_method and feature_envy too. Root "
        "cause: (class, contains, method) is method's only incoming edge, so once belongs_to "
        "lets class absorb its methods' embeddings, conv2 folds that sibling-aggregate signal "
        "back into every method's own embedding — which all three heads read.\n"
    )
    lines.append(
        "**Design B:** class_method_pool=True keeps the shared backbone on plain EDGE_TYPES "
        "(no belongs_to edge wired into conv1/conv2 at all) and instead mean-pools conv2's "
        "method embeddings into a per-class vector, using the existing (class, contains, "
        "method) edge_index, fed only to head_god_class (concatenated with class's own conv2 "
        "embedding). method/function embeddings are structurally untouched by this — see "
        "tests/test_gat_baseline.py::test_class_pool_does_not_alter_method_pathway, which "
        "asserts h_dict['method'] and the long_method/feature_envy head outputs are "
        "byte-for-byte identical whether or not pooling is enabled.\n"
    )
    lines.append(
        f"Reused the exact Phase 10 tuned hyperparameters (models/hybrid_tuned/"
        f"best_params.json: hidden_dim={best_params['hidden_dim']}, heads={best_params['heads']}, "
        f"dropout={best_params['dropout']:.4f}, lr={best_params['lr']:.2e}, "
        f"weight_decay={best_params['weight_decay']:.2e}) — no re-tuning. Trained on TRAIN "
        f"split (max_epochs={EPOCHS}, patience={PATIENCE}, best_epoch={best_epoch}), evaluated "
        f"on VAL split only. TEST split untouched.\n"
    )

    if tuned_metrics is not None:
        lines.append("\n## Comparison: Phase 10 tuned vs Design A (reverse edge) vs Design B (pooling)\n")
        header = "| task | Phase 10 F1 | Design A F1 | Design B F1 | Phase10->A | Phase10->B | Phase 10 prec | Design B prec | Phase 10 recall | Design B recall | Phase 10 ROC-AUC | Design B ROC-AUC |"
        lines.append(header)
        lines.append("|" + "---|" * (header.count("|") - 1))
        for task in ("long_method", "feature_envy", "god_class"):
            old = tuned_metrics.get(task)
            a = (class_agg_metrics or {}).get(task)
            b = final_val_metrics.get(task)
            if old is None or b is None:
                lines.append(f"| {task} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            a_f1 = f"{a['f1']:.3f}" if a else "n/a"
            d_a = f"{(a['f1'] - old['f1']):+.3f}" if a else "n/a"
            d_b = f"{(b['f1'] - old['f1']):+.3f}"
            lines.append(
                f"| {task} | {old['f1']:.3f} | {a_f1} | {b['f1']:.3f} | {d_a} | {d_b} | "
                f"{old['precision']:.3f} | {b['precision']:.3f} | "
                f"{old['recall']:.3f} | {b['recall']:.3f} | "
                f"{old['roc_auc']:.3f} | {b['roc_auc']:.3f} |"
            )
    else:
        lines.append("\n(models/hybrid_tuned/final_val_metrics.json not found — no comparison table.)\n")

    for task, v in final_val_metrics.items():
        if v is None:
            lines.append(f"\n## {task} (Design B raw)\n\n- no eligible nodes in val split\n")
            continue
        lines.append(f"\n## {task} (Design B raw)\n")
        lines.append(
            f"- precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} "
            f"roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f} "
            f"(n={v['n']}, positives={v['n_positive']}, tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']})"
        )

    (DOCS_DIR / "class_pool_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved model/metrics to {MODELS_DIR}, report to {DOCS_DIR / 'class_pool_report.md'}")


if __name__ == "__main__":
    main()
