# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.graph.graph_builder import build_hetero_graph
from ml.models.gat_baseline import HeteroGAT
from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds
from torch_geometric.loader import DataLoader as PyGDataLoader

from scripts.error_analysis import apply_norm, confusion_counts, predict_batch, predict_file, render_examples
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS

LOW_THRESHOLDS = LabelThresholds(long_method_statements=1000, god_class_method_count=1000, god_class_loc=1000)

SRC = '''
class Course:
    def get_marks(self):
        return 1

class Student:
    def __init__(self, name):
        self.name = name

    def calculate_result(self, course):
        course.get_marks()
        return self.name

def helper():
    return 1
'''


def _fake_hybrid_graph():
    m = parse_source(SRC, path="school.py")
    data = build_hetero_graph(m, LOW_THRESHOLDS).data
    for nt in ("class", "method", "function"):
        k = data[nt].x.shape[0]
        data[nt].x = torch.cat([data[nt].x, torch.zeros((k, 768))], dim=1)
    return data


def test_predict_file_returns_one_prob_per_node():
    data = _fake_hybrid_graph()
    n_method = data["method"].x.shape[0]
    n_function = data["function"].x.shape[0]
    n_class = data["class"].x.shape[0]

    model = HeteroGAT(hidden_dim=8, heads=2, dropout=0.0, node_feature_dims=HYBRID_NODE_DIMS)
    model.eval()

    norm_stats = {
        nt: (torch.zeros(HYBRID_NODE_DIMS[nt]), torch.ones(HYBRID_NODE_DIMS[nt]))
        for nt in ("class", "method", "function")
    }
    apply_norm(data, norm_stats)

    preds = predict_file(model, data)
    assert len(preds["method_lm"]) == n_method
    assert len(preds["method_fe"]) == n_method
    assert len(preds["function_lm"]) == n_function
    assert len(preds["class_gc"]) == n_class
    assert all(0.0 <= p <= 1.0 for p in preds["method_lm"])


def test_predict_batch_splits_back_to_per_graph_lists():
    data_a = _fake_hybrid_graph()
    data_b = _fake_hybrid_graph()
    n_method = data_a["method"].x.shape[0]
    n_function = data_a["function"].x.shape[0]
    n_class = data_a["class"].x.shape[0]

    model = HeteroGAT(hidden_dim=8, heads=2, dropout=0.0, node_feature_dims=HYBRID_NODE_DIMS)
    model.eval()
    norm_stats = {
        nt: (torch.zeros(HYBRID_NODE_DIMS[nt]), torch.ones(HYBRID_NODE_DIMS[nt]))
        for nt in ("class", "method", "function")
    }
    apply_norm(data_a, norm_stats)
    apply_norm(data_b, norm_stats)

    # single-graph predictions, for comparison against the batched split
    single_a = predict_file(model, data_a)
    single_b = predict_file(model, data_b)

    loader = PyGDataLoader([data_a, data_b], batch_size=2, shuffle=False)
    batch = next(iter(loader))
    preds = predict_batch(model, batch)

    assert len(preds["method_lm"]) == 2
    assert len(preds["method_lm"][0]) == n_method
    assert len(preds["method_lm"][1]) == n_method
    assert len(preds["function_lm"][0]) == n_function
    assert len(preds["class_gc"][0]) == n_class

    # identical graphs (data_a and data_b built from the same source) should
    # get identical per-graph predictions from the batched pass, and closely
    # match (not necessarily bit-identical, per the module docstring) the
    # single-graph pass.
    assert preds["method_lm"][0] == preds["method_lm"][1]
    for a, b in zip(preds["method_lm"][0], single_a["method_lm"]):
        assert abs(a - b) < 1e-3


def test_confusion_counts_basic():
    recs = [
        {"label": 1, "prob": 0.9},  # tp
        {"label": 1, "prob": 0.1},  # fn
        {"label": 0, "prob": 0.8},  # fp
        {"label": 0, "prob": 0.2},  # tn
        {"label": 0, "prob": 0.2},  # tn
    ]
    cc = confusion_counts(recs)
    assert cc == {
        "n": 5, "positives": 2, "tp": 1, "fp": 1, "fn": 1, "tn": 2,
        "precision": 0.5, "recall": 0.5, "f1": 0.5,
    }


def test_render_examples_picks_hardest_by_confidence():
    recs = [
        {"label": 0, "prob": 0.9, "id": "hardest_fp"},
        {"label": 0, "prob": 0.1, "id": "easy_tn"},
        {"label": 1, "prob": 0.05, "id": "hardest_fn"},
        {"label": 1, "prob": 0.95, "id": "easy_tp"},
    ]
    hardest_fp = render_examples(recs, label=0, top_k=1, reverse=True)
    hardest_fn = render_examples(recs, label=1, top_k=1, reverse=False)
    assert hardest_fp[0]["id"] == "hardest_fp"
    assert hardest_fn[0]["id"] == "hardest_fn"
