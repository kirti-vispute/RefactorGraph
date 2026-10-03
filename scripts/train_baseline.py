# -*- coding: utf-8 -*-
"""Phase 6: metrics-only ML baseline (REQUIRED comparison point per project
spec section 11) — Logistic Regression + Random Forest trained on the same
hand-crafted structural metrics already stored in data/processed/*.jsonl.
No source text, no graph structure, no CodeBERT: this is the floor every
later model (GAT-only, CodeBERT-only, hybrid) must be compared against.

Trained on TRAIN split only, evaluated on VAL split only. TEST split is not
touched here — per the project's test-set discipline it is reserved for the
final cross-model comparison after every model (including this baseline) is
finalized, so it is scored exactly once.

Honest caveat (documented, not hidden): Long Method and God/Large Class
labels are DETERMINISTIC thresholds on a subset of these same features
(statement_count; method_count AND loc), so a model given those features is
expected to score very close to ceiling — this is a sanity check that the
label pipeline and features are consistent with each other, not evidence the
model has learned to detect real code smells. Feature Envy's label is a
genuine two-feature comparison rule (dominant_external_count >= 3 AND >
self_access_count), so it is the more informative baseline of the three.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
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
MODELS_DIR = ROOT / "models" / "baseline"
DOCS_DIR = ROOT / "docs"

METHOD_FEATURES = [
    "loc", "statement_count", "self_access_count",
    "external_access_count", "dominant_external_count", "is_method",
]
CLASS_FEATURES = ["loc", "method_count", "field_count"]


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_matrix(records: list, feature_names: list) -> np.ndarray:
    rows = []
    for r in records:
        row = []
        for name in feature_names:
            v = r[name]
            row.append(float(v) if not isinstance(v, bool) else float(v))
        rows.append(row)
    return np.array(rows, dtype=float)


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


def run_task(task_name: str, X_train, y_train, X_val, y_val, feature_names: list) -> dict:
    results = {}
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        train_metrics = eval_split(model, X_train, y_train)
        val_metrics = eval_split(model, X_val, y_val)
        results[model_name] = {"train": train_metrics, "val": val_metrics}

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / f"{task_name}_{model_name}.joblib")

        if model_name == "random_forest":
            importances = dict(zip(feature_names, model.feature_importances_.tolist()))
            results[model_name]["feature_importances"] = importances
    return results


def main():
    train_methods = load_jsonl(PROCESSED_DIR / "train_methods.jsonl")
    val_methods = load_jsonl(PROCESSED_DIR / "val_methods.jsonl")
    train_classes = load_jsonl(PROCESSED_DIR / "train_classes.jsonl")
    val_classes = load_jsonl(PROCESSED_DIR / "val_classes.jsonl")

    for r in train_methods + val_methods:
        r["is_method"] = 1.0 if r["is_method"] else 0.0

    all_results = {}

    # --- Long Method: all callables (methods + functions) eligible ---
    Xtr = to_matrix(train_methods, METHOD_FEATURES)
    ytr = np.array([r["long_method"] for r in train_methods], dtype=int)
    Xv = to_matrix(val_methods, METHOD_FEATURES)
    yv = np.array([r["long_method"] for r in val_methods], dtype=int)
    all_results["long_method"] = run_task("long_method", Xtr, ytr, Xv, yv, METHOD_FEATURES)
    print(f"[long_method] train n={len(ytr)} pos={ytr.sum()}, val n={len(yv)} pos={yv.sum()}")

    # --- Feature Envy: methods only (functions are never eligible, see label_rules.py) ---
    train_m_only = [r for r in train_methods if r["is_method"]]
    val_m_only = [r for r in val_methods if r["is_method"]]
    Xtr = to_matrix(train_m_only, METHOD_FEATURES)
    ytr = np.array([r["feature_envy"] for r in train_m_only], dtype=int)
    Xv = to_matrix(val_m_only, METHOD_FEATURES)
    yv = np.array([r["feature_envy"] for r in val_m_only], dtype=int)
    all_results["feature_envy"] = run_task("feature_envy", Xtr, ytr, Xv, yv, METHOD_FEATURES)
    print(f"[feature_envy] train n={len(ytr)} pos={ytr.sum()}, val n={len(yv)} pos={yv.sum()}")

    # --- God/Large Class ---
    Xtr = to_matrix(train_classes, CLASS_FEATURES)
    ytr = np.array([r["god_class"] for r in train_classes], dtype=int)
    Xv = to_matrix(val_classes, CLASS_FEATURES)
    yv = np.array([r["god_class"] for r in val_classes], dtype=int)
    all_results["god_class"] = run_task("god_class", Xtr, ytr, Xv, yv, CLASS_FEATURES)
    print(f"[god_class] train n={len(ytr)} pos={ytr.sum()}, val n={len(yv)} pos={yv.sum()}")

    (MODELS_DIR / "baseline_metrics.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    lines = ["# Baseline Model Report (Phase 6 — metrics-only ML)\n"]
    lines.append(
        "Logistic Regression + Random Forest trained on hand-crafted structural "
        "metrics only (no source text, no graph structure). Trained on TRAIN "
        "split, evaluated on VAL split. TEST split untouched — reserved for the "
        "final cross-model comparison.\n"
    )
    lines.append(
        "**Caveat**: Long Method and God/Large Class labels are deterministic "
        "thresholds on a subset of these exact features, so near-ceiling scores "
        "on those two tasks are expected and mainly confirm label/feature "
        "consistency, not smell-detection ability in a general sense. Feature "
        "Envy's label is a genuine two-feature comparison rule, making it the "
        "more informative of the three baselines here.\n"
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
    (DOCS_DIR / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved models to {MODELS_DIR}, report to {DOCS_DIR / 'baseline_report.md'}")


if __name__ == "__main__":
    main()
