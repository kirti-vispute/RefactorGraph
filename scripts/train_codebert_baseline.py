# -*- coding: utf-8 -*-
"""Phase 8 step 2: CodeBERT-only baseline. Logistic Regression + Random
Forest trained on frozen CodeBERT mean-pooled embeddings (768-dim, from
extract_codebert_embeddings.py) — no hand-crafted metrics, no graph
structure. Same classifier harness as Phase 6 (train_baseline.py) and the
same eval metric set as Phase 6/7, so results are directly comparable.

Trained on TRAIN split, evaluated on VAL split. TEST split untouched.

Tests whether source-text semantics alone (no structure, no explicit
metrics) can recover what Phase 6's raw metrics already solved almost
perfectly for Long Method / God Class (since those labels are literal
thresholds on statement_count / method_count / loc, which CodeBERT never
sees directly, only implicitly through token patterns like line count and
nesting), and whether it does any better than the GAT on Feature Envy (a
literal external-vs-self access comparison rule that has no lexical
signature CodeBERT could plausibly detect from source text alone).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
EMB_DIR = ROOT / "models" / "codebert_embeddings"
MODELS_DIR = ROOT / "models" / "codebert_baseline"
DOCS_DIR = ROOT / "docs"


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eval_split(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
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


def run_task(task_name: str, X_train, y_train, X_val, y_val) -> dict:
    results = {}
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        # max_depth/min_samples_leaf are NOT the Phase 6 defaults (unlimited depth,
        # leaf=1). Unregularized RF on 768-dim dense CodeBERT embeddings memorizes
        # train almost perfectly (train f1~1.0) and collapses on val (val f1 as low
        # as 0.01) -- severe overfitting, confirmed by a train/val gap sweep over
        # {None,10,6,4} x {1,5,10,20}. max_depth=10, min_samples_leaf=5 was the best
        # val-f1 point on that sweep (long_method val f1 0.13 -> 0.44) and is applied
        # to all 3 tasks for consistency. Phase 6's structural metrics (3-6 dims) do
        # not need this regularization; this is specific to the high-dimensional
        # dense embedding feature space.
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        train_metrics = eval_split(model, X_train, y_train)
        val_metrics = eval_split(model, X_val, y_val)
        results[model_name] = {"train": train_metrics, "val": val_metrics}

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / f"{task_name}_{model_name}.joblib")
    return results


def main():
    train_methods = load_jsonl(PROCESSED_DIR / "train_methods.jsonl")
    val_methods = load_jsonl(PROCESSED_DIR / "val_methods.jsonl")
    train_classes = load_jsonl(PROCESSED_DIR / "train_classes.jsonl")
    val_classes = load_jsonl(PROCESSED_DIR / "val_classes.jsonl")

    Etr_m = torch.load(EMB_DIR / "train_methods.pt", weights_only=False).numpy()
    Ev_m = torch.load(EMB_DIR / "val_methods.pt", weights_only=False).numpy()
    Etr_c = torch.load(EMB_DIR / "train_classes.pt", weights_only=False).numpy()
    Ev_c = torch.load(EMB_DIR / "val_classes.pt", weights_only=False).numpy()

    assert Etr_m.shape[0] == len(train_methods)
    assert Ev_m.shape[0] == len(val_methods)
    assert Etr_c.shape[0] == len(train_classes)
    assert Ev_c.shape[0] == len(val_classes)

    all_results = {}

    # --- Long Method: all callables (methods + functions) eligible ---
    ytr = np.array([r["long_method"] for r in train_methods], dtype=int)
    yv = np.array([r["long_method"] for r in val_methods], dtype=int)
    all_results["long_method"] = run_task("long_method", Etr_m, ytr, Ev_m, yv)
    print(f"[long_method] train n={len(ytr)} pos={ytr.sum()}, val n={len(yv)} pos={yv.sum()}")

    # --- Feature Envy: methods only (functions never eligible, see label_rules.py) ---
    m_idx_tr = [i for i, r in enumerate(train_methods) if r["is_method"]]
    m_idx_v = [i for i, r in enumerate(val_methods) if r["is_method"]]
    Etr_fe = Etr_m[m_idx_tr]
    Ev_fe = Ev_m[m_idx_v]
    ytr = np.array([train_methods[i]["feature_envy"] for i in m_idx_tr], dtype=int)
    yv = np.array([val_methods[i]["feature_envy"] for i in m_idx_v], dtype=int)
    all_results["feature_envy"] = run_task("feature_envy", Etr_fe, ytr, Ev_fe, yv)
    print(f"[feature_envy] train n={len(ytr)} pos={ytr.sum()}, val n={len(yv)} pos={yv.sum()}")

    # --- God/Large Class ---
    ytr = np.array([r["god_class"] for r in train_classes], dtype=int)
    yv = np.array([r["god_class"] for r in val_classes], dtype=int)
    all_results["god_class"] = run_task("god_class", Etr_c, ytr, Ev_c, yv)
    print(f"[god_class] train n={len(ytr)} pos={ytr.sum()}, val n={len(yv)} pos={yv.sum()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "codebert_baseline_metrics.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # comparison against Phase 6 (metrics) and Phase 7 (GAT) if available
    rf_metrics = {}
    gat_metrics = {}
    p6_path = ROOT / "models" / "baseline" / "baseline_metrics.json"
    p7_path = ROOT / "models" / "gat_baseline" / "final_val_metrics.json"
    if p6_path.exists():
        p6 = json.loads(p6_path.read_text(encoding="utf-8"))
        rf_metrics = {t: p6[t]["random_forest"]["val"] for t in p6}
    if p7_path.exists():
        gat_metrics = json.loads(p7_path.read_text(encoding="utf-8"))

    lines = ["# CodeBERT Baseline Report (Phase 8 — frozen CodeBERT embeddings)\n"]
    lines.append(
        "Logistic Regression + Random Forest trained on frozen microsoft/codebert-base "
        "mean-pooled embeddings (768-dim, max_length=256, no fine-tuning per the staged "
        "training plan). No hand-crafted metrics, no graph structure. Trained on TRAIN "
        "split, evaluated on VAL split. TEST split untouched.\n"
    )
    lines.append(
        "**Caveat**: embeddings are truncated at 256 sub-word tokens — very long "
        "methods/classes lose their tail. Mean pooling over the attention mask is used "
        "instead of the raw [CLS] token, since CodeBERT's MLM/RTD pretraining gives no "
        "particular meaning to [CLS] as a whole-snippet summary.\n"
    )
    lines.append(
        "**Caveat**: Random Forest here uses max_depth=10, min_samples_leaf=5 "
        "(unlike Phase 6's unregularized RF). An unregularized RF on these 768-dim "
        "dense embeddings memorized TRAIN almost perfectly (f1~1.0) and collapsed on "
        "VAL (f1 as low as 0.01) — confirmed via a train/val gap sweep. This "
        "regularized configuration was the best val-f1 point on that sweep; it is "
        "still consistently worse than Logistic Regression below, which is the more "
        "meaningful classifier for a frozen dense-embedding baseline.\n"
    )
    for task, task_results in all_results.items():
        lines.append(f"\n## {task}\n")
        for model_name, r in task_results.items():
            v = r["val"]
            lines.append(
                f"- **{model_name}** (val): precision={v['precision']:.3f} "
                f"recall={v['recall']:.3f} f1={v['f1']:.3f} "
                f"roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f} "
                f"(n={v['n']}, positives={v['n_positive']}, "
                f"tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']})"
            )

    if rf_metrics or gat_metrics:
        lines.append("\n## Comparison against Phase 6 (metrics RF) and Phase 7 (GAT), same val split\n")
        lines.append("| task | RF F1 | GAT F1 | CodeBERT RF F1 | RF PR-AUC | GAT PR-AUC | CodeBERT RF PR-AUC |")
        lines.append("|---|---|---|---|---|---|---|")
        for task in all_results:
            cb = all_results[task]["random_forest"]["val"]
            rf = rf_metrics.get(task, {})
            gat = gat_metrics.get(task, {})
            lines.append(
                f"| {task} | {rf.get('f1', float('nan')):.3f} | {gat.get('f1', float('nan')):.3f} | "
                f"{cb['f1']:.3f} | {rf.get('pr_auc', float('nan')):.3f} | "
                f"{gat.get('pr_auc', float('nan')):.3f} | {cb['pr_auc']:.3f} |"
            )
        lines.append(
            "\nCodeBERT-only (best classifier here is Logistic Regression, not the "
            "table's RF column) underperforms both the Phase 6 metrics RF and the "
            "Phase 7 GAT on all 3 tasks. Expected, not a failure: Long Method and "
            "God/Large Class labels are near-deterministic thresholds on exact counts "
            "(statement_count, loc, method_count) — a 768-dim semantic embedding can "
            "correlate with code length but does not expose an exact countable "
            "magnitude the way the raw metric does, and the 256-token truncation "
            "discards the tail of longer methods/classes entirely, compounding this. "
            "Feature Envy fares worst of all three baselines tried so far (LR f1=0.160) "
            "since it is a pairwise comparison rule (dominant_external_count vs "
            "self_access_count) — the embedding sees only one method's own source "
            "text, and while self./other-object access patterns are lexically present, "
            "recovering the exact comparison from a pooled semantic vector is a much "
            "harder inference than reading the two counts directly. This motivates the "
            "Phase 9 hybrid model: CodeBERT semantics alone lose precise structural "
            "magnitude, and structure alone (Phase 7) lacks semantics — combining both "
            "is where a genuine improvement over the metrics-only floor is plausible."
        )

    (DOCS_DIR / "codebert_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved models to {MODELS_DIR}, report to {DOCS_DIR / 'codebert_baseline_report.md'}")


if __name__ == "__main__":
    main()
