# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from ml.graph.graph_builder import build_hetero_graph
from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds
from scripts.tune_hybrid_optuna import HYBRID_NODE_DIMS, train_one

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


def _fake_hybrid_graphs(n=4):
    graphs = []
    for _ in range(n):
        m = parse_source(SRC, path="school.py")
        data = build_hetero_graph(m, LOW_THRESHOLDS).data
        for nt in ("class", "method", "function"):
            k = data[nt].x.shape[0]
            data[nt].x = torch.cat([data[nt].x, torch.zeros((k, 768))], dim=1)
        graphs.append(data)
    return graphs


def test_train_one_runs_and_respects_hidden_dim_heads_divisibility():
    graphs = _fake_hybrid_graphs()
    train_loader = DataLoader(graphs, batch_size=2, shuffle=False)
    val_loader = DataLoader(graphs, batch_size=2, shuffle=False)
    loss_lm = nn.BCEWithLogitsLoss()
    loss_fe = nn.BCEWithLogitsLoss()
    loss_gc = nn.BCEWithLogitsLoss()

    best_score, best_epoch, best_state = train_one(
        hidden_dim=8, heads=2, dropout=0.0, lr=1e-3, weight_decay=0.0,
        train_loader=train_loader, val_loader=val_loader,
        loss_lm=loss_lm, loss_fe=loss_fe, loss_gc=loss_gc,
        device=torch.device("cpu"), epochs=2, patience=2, trial=None,
    )
    assert best_epoch in (1, 2)
    assert best_state is not None
    assert 0.0 <= best_score <= 1.0


def test_hybrid_node_dims_match_train_hybrid_baseline():
    assert HYBRID_NODE_DIMS["class"] == 3 + 768
    assert HYBRID_NODE_DIMS["method"] == 5 + 768
    assert HYBRID_NODE_DIMS["function"] == 5 + 768
