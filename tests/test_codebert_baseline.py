# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.extract_codebert_embeddings import mean_pool
from scripts.train_codebert_baseline import eval_split


def test_mean_pool_ignores_padding_positions():
    # 2 sequences, 3 tokens each, hidden dim 2. Second sequence has 1 pad token.
    last_hidden = torch.tensor([
        [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],
        [[2.0, 2.0], [4.0, 4.0], [999.0, 999.0]],  # last position is padding
    ])
    attention_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0],
    ])
    pooled = mean_pool(last_hidden, attention_mask)
    assert torch.allclose(pooled[0], torch.tensor([3.0, 3.0]))  # mean(1,3,5)
    assert torch.allclose(pooled[1], torch.tensor([3.0, 3.0]))  # mean(2,4), pad excluded


class _StubModel:
    """Predicts exactly y (as probability 1.0/0.0) — sanity-checks eval_split
    on frozen-embedding-shaped inputs without loading a real classifier."""

    def __init__(self, y):
        self._y = y

    def predict(self, X):
        return self._y

    def predict_proba(self, X):
        p1 = self._y.astype(float)
        return np.stack([1 - p1, p1], axis=1)


def test_eval_split_perfect_classifier_reports_ones():
    y = np.array([0, 1, 1, 0, 1])
    model = _StubModel(y)
    X = np.zeros((5, 768))
    m = eval_split(model, X, y)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 3 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_eval_split_handles_single_class_val_without_crashing():
    y = np.array([0, 0, 0])
    model = _StubModel(y)
    X = np.zeros((3, 768))
    m = eval_split(model, X, y)
    assert m["roc_auc"] is None
    assert m["pr_auc"] is None
    assert m["precision"] == 0.0
