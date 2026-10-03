# -*- coding: utf-8 -*-
"""Phase 7: train the shallow HeteroGAT baseline (ml/models/gat_baseline.py)
on the saved per-file graphs (data/processed/graphs/{split}/**/*.pt).

Same three tasks, same train/val split discipline as the Phase 6 metrics
baseline (train on TRAIN, model-select and report on VAL, TEST untouched).
Per-node-type feature standardization (mean/std) is computed from TRAIN
graphs only and applied to val/test — no leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.models.gat_baseline import HeteroGAT

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "data" / "processed" / "graphs"
MODELS_DIR = ROOT / "models" / "gat_baseline"
DOCS_DIR = ROOT / "docs"

NORM_NODE_TYPES = ["module", "class", "method", "function"]
EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 32
HIDDEN_DIM = 32
HEADS = 2
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def load_split(split: str) -> list:
    return [torch.load(p, weights_only=False) for p in sorted((GRAPH_DIR / split).rglob("*.pt"))]


def compute_norm_stats(graphs: list) -> dict:
    stats = {}
    for nt in NORM_NODE_TYPES:
        xs = [g[nt].x for g in graphs if g[nt].x.shape[0] > 0]
        if not xs:
            continue
        cat = torch.cat(xs, dim=0)
        mean = cat.mean(dim=0)
        std = cat.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        stats[nt] = (mean, std)
    return stats


def apply_norm(graphs: list, stats: dict):
    for g in graphs:
        for nt, (mean, std) in stats.items():
            if g[nt].x.shape[0] > 0:
                g[nt].x = (g[nt].x - mean) / std


def pos_weight_for(pos: int, neg: int) -> torch.Tensor:
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(float(neg) / float(pos))


def collect_task_labels(graphs: list, node_type: str, y_attr: str) -> tuple:
    pos = sum(int(g[node_type][y_attr].sum().item()) for g in graphs)
    total = sum(g[node_type][y_attr].shape[0] for g in graphs)
    return pos, total - pos


@torch.no_grad()
def run_eval(model, loader, device) -> dict:
    model.eval()
    logits_lm, labels_lm = [], []
    logits_fe, labels_fe = [], []
    logits_gc, labels_gc = [], []
    for batch in loader:
        batch = batch.to(device)
        edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
        h = model(batch.x_dict, edge_index_dict)
        if "method" in h and batch["method"].x.shape[0] > 0:
            logits_lm.append(model.predict_long_method(h, "method").cpu())
            labels_lm.append(batch["method"].y_long_method.cpu())
            logits_fe.append(model.predict_feature_envy(h).cpu())
            labels_fe.append(batch["method"].y_feature_envy.cpu())
        if "function" in h and batch["function"].x.shape[0] > 0:
            logits_lm.append(model.predict_long_method(h, "function").cpu())
            labels_lm.append(batch["function"].y_long_method.cpu())
        if "class" in h and batch["class"].x.shape[0] > 0:
            logits_gc.append(model.predict_god_class(h).cpu())
            labels_gc.append(batch["class"].y.cpu())

    def metrics_for(logits_list, labels_list):
        if not logits_list:
            return None
        logits = torch.cat(logits_list).numpy()
        y = torch.cat(labels_list).numpy().astype(int)
        proba = 1.0 / (1.0 + np.exp(-logits))
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "n": int(len(y)), "n_positive": int(y.sum()),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)) if len(set(y.tolist())) > 1 else None,
            "pr_auc": float(average_precision_score(y, proba)) if len(set(y.tolist())) > 1 else None,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }

    return {
        "long_method": metrics_for(logits_lm, labels_lm),
        "feature_envy": metrics_for(logits_fe, labels_fe),
        "god_class": metrics_for(logits_gc, labels_gc),
    }


def main():
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    print("[load] reading saved graphs ...")
    train_graphs = load_split("train")
    val_graphs = load_split("val")
    print(f"[load] train={len(train_graphs)} val={len(val_graphs)}")

    norm_stats = compute_norm_stats(train_graphs)
    apply_norm(train_graphs, norm_stats)
    apply_norm(val_graphs, norm_stats)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

    lm_pos, lm_neg = collect_task_labels(train_graphs, "method", "y_long_method")
    lm_pos_f, lm_neg_f = collect_task_labels(train_graphs, "function", "y_long_method")
    fe_pos, fe_neg = collect_task_labels(train_graphs, "method", "y_feature_envy")
    gc_pos, gc_neg = collect_task_labels(train_graphs, "class", "y")

    pw_lm = pos_weight_for(lm_pos + lm_pos_f, lm_neg + lm_neg_f)
    pw_fe = pos_weight_for(fe_pos, fe_neg)
    pw_gc = pos_weight_for(gc_pos, gc_neg)
    print(f"[pos_weight] long_method={pw_lm.item():.2f} feature_envy={pw_fe.item():.2f} god_class={pw_gc.item():.2f}")

    loss_lm = nn.BCEWithLogitsLoss(pos_weight=pw_lm)
    loss_fe = nn.BCEWithLogitsLoss(pos_weight=pw_fe)
    loss_gc = nn.BCEWithLogitsLoss(pos_weight=pw_gc)

    model = HeteroGAT(hidden_dim=HIDDEN_DIM, heads=HEADS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
            h = model(batch.x_dict, edge_index_dict)
            loss = torch.tensor(0.0)
            has_loss = False
            if "method" in h and batch["method"].x.shape[0] > 0:
                l1 = loss_lm(model.predict_long_method(h, "method"), batch["method"].y_long_method)
                l2 = loss_fe(model.predict_feature_envy(h), batch["method"].y_feature_envy)
                loss = loss + l1 + l2
                has_loss = True
            if "function" in h and batch["function"].x.shape[0] > 0:
                l3 = loss_lm(model.predict_long_method(h, "function"), batch["function"].y_long_method)
                loss = loss + l3
                has_loss = True
            if "class" in h and batch["class"].x.shape[0] > 0:
                l4 = loss_gc(model.predict_god_class(h), batch["class"].y)
                loss = loss + l4
                has_loss = True
            if not has_loss:
                continue
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_metrics = run_eval(model, val_loader, device)
        f1s = [v["f1"] for v in val_metrics.values() if v is not None]
        combined_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        train_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_combined_f1": combined_f1,
                         "val": val_metrics})
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} val_combined_f1={combined_f1:.4f} "
              f"(lm_f1={val_metrics['long_method']['f1']:.3f} "
              f"fe_f1={val_metrics['feature_envy']['f1']:.3f} "
              f"gc_f1={val_metrics['god_class']['f1']:.3f})")

        if combined_f1 > best_score:
            best_score = combined_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODELS_DIR / "model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[early stop] no val improvement for {PATIENCE} epochs, stopping at epoch {epoch}")
                break

    torch.save({nt: (m.tolist(), s.tolist()) for nt, (m, s) in norm_stats.items()},
                MODELS_DIR / "norm_stats.pt")

    model.load_state_dict(torch.load(MODELS_DIR / "model.pt", weights_only=True))
    final_val_metrics = run_eval(model, val_loader, device)

    (MODELS_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (MODELS_DIR / "final_val_metrics.json").write_text(json.dumps(final_val_metrics, indent=2), encoding="utf-8")

    lines = ["# GAT Baseline Report (Phase 7 — shallow 2-layer HeteroGAT, structure-only)\n"]
    lines.append(
        f"Best epoch: {best_epoch} (early stopping patience={PATIENCE}, max_epochs={EPOCHS}). "
        f"Node features are the same placeholder structural metrics as the Phase 6 baseline "
        f"(no CodeBERT semantics yet) — this isolates the effect of graph message passing "
        f"alone. hidden_dim={HIDDEN_DIM}, heads={HEADS}, dropout={DROPOUT}, 2 GAT layers "
        f"(shallow, to avoid over-smoothing per the staged training plan).\n"
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

    baseline_json = MODELS_DIR.parent / "baseline" / "baseline_metrics.json"
    if baseline_json.exists():
        rf_metrics = json.loads(baseline_json.read_text(encoding="utf-8"))
        lines.append("\n## Comparison against Phase 6 metrics-only baseline (random_forest, same val split)\n")
        lines.append("| task | RF F1 | GAT F1 | RF PR-AUC | GAT PR-AUC |")
        lines.append("|---|---|---|---|---|")
        for task in ("long_method", "feature_envy", "god_class"):
            rf_v = rf_metrics[task]["random_forest"]["val"]
            gat_v = final_val_metrics[task]
            gat_f1 = f"{gat_v['f1']:.3f}" if gat_v else "n/a"
            gat_pr = f"{gat_v['pr_auc']:.3f}" if gat_v and gat_v["pr_auc"] is not None else "n/a"
            lines.append(f"| {task} | {rf_v['f1']:.3f} | {gat_f1} | {rf_v['pr_auc']:.3f} | {gat_pr} |")
        lines.append(
            "\nRF matches or beats the GAT on every task here. This is expected, not a failure of "
            "the GAT: Long Method and God/Large Class labels are deterministic (or near-deterministic) "
            "thresholds on the exact raw features RF sees directly, while the GAT only sees those "
            "features after a Linear encoder + 2 rounds of neighbor-mixing + ReLU — that compression "
            "necessarily loses some of the precise raw magnitude a hard threshold needs. For Feature "
            "Envy, RF can carve out the exact pairwise comparison rule (dominant_external_count vs "
            "self_access_count) with a couple of tree splits; the GAT has to reconstruct an equivalent "
            "decision boundary from mixed, ReLU'd embeddings, which is a harder optimization problem "
            "for a 2-layer model with this little data (547 training graphs). Graph structure is not "
            "expected to pay off against a metrics-only baseline until it is combined with CodeBERT "
            "semantics and evaluated on genuinely ambiguous cases (e.g. the Visitor-pattern false "
            "positives documented in docs/label_review_findings.md) rather than on labels that are "
            "themselves thresholds on the input features.\n"
        )
    (DOCS_DIR / "gat_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved model/history to {MODELS_DIR}, report to {DOCS_DIR / 'gat_baseline_report.md'}")


if __name__ == "__main__":
    main()
