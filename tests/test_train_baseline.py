# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.train_baseline import eval_split, to_matrix
from sklearn.linear_model import LogisticRegression


def test_to_matrix_preserves_order_and_values():
    records = [
        {"loc": 10, "statement_count": 5, "is_method": 1.0},
        {"loc": 20, "statement_count": 15, "is_method": 0.0},
    ]
    X = to_matrix(records, ["loc", "statement_count", "is_method"])
    assert X.shape == (2, 3)
    assert X[0].tolist() == [10.0, 5.0, 1.0]
    assert X[1].tolist() == [20.0, 15.0, 0.0]


def test_eval_split_perfect_classifier_reports_ones():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    metrics = eval_split(model, X, y)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 2


def test_eval_split_handles_single_class_val_without_crashing():
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0, 1])
    model = LogisticRegression().fit(X_train, y_train)
    X_val = np.array([[0.0], [0.0]])
    y_val = np.array([0, 0])
    metrics = eval_split(model, X_val, y_val)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["n"] == 2
