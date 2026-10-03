# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.graph.graph_builder import build_hetero_graph
from ml.models.gat_baseline import HeteroGAT
from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds
from scripts.augment_graphs_with_codebert import _snippet
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS

LOW_THRESHOLDS = LabelThresholds(long_method_statements=1000, god_class_method_count=1000, god_class_loc=1000)

SRC = '''
import os

class Course:
    def get_marks(self):
        return 1

class Student:
    def __init__(self, name):
        self.name = name

    def calculate_result(self, course):
        course.get_marks()
        course.get_marks()
        course.get_marks()
        return self.name

def helper():
    return 1
'''


def _build_graph_with_fake_embeddings():
    """Mirrors what augment_graphs_with_codebert.py does structurally: take a
    Phase 7 structure-only graph and concatenate a (fake, zero) 768-dim
    embedding onto class/method/function node features, to sanity-check the
    hybrid model's forward pass without loading the real CodeBERT model."""
    m = parse_source(SRC, path="school.py")
    data = build_hetero_graph(m, LOW_THRESHOLDS).data
    for nt in ("class", "method", "function"):
        n = data[nt].x.shape[0]
        data[nt].x = torch.cat([data[nt].x, torch.zeros((n, 768))], dim=1)
    return data


def test_snippet_extracts_inclusive_line_range():
    lines = ["def f():", "    x = 1", "    return x", "trailing"]
    assert _snippet(lines, 1, 3) == "def f():\n    x = 1\n    return x"


def test_snippet_returns_empty_string_for_missing_file():
    assert _snippet([], 1, 3) == ""


def test_hybrid_gat_forward_pass_with_concatenated_embeddings():
    data = _build_graph_with_fake_embeddings()
    model = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0, node_feature_dims=HYBRID_NODE_DIMS)
    model.eval()
    with torch.no_grad():
        h = model(data.x_dict, data.edge_index_dict)
    lm = model.predict_long_method(h, "method")
    fe = model.predict_feature_envy(h)
    gc = model.predict_god_class(h)
    assert lm.shape[0] == data["method"].x.shape[0]
    assert fe.shape[0] == data["method"].x.shape[0]
    assert gc.shape[0] == data["class"].x.shape[0]
    assert torch.isfinite(lm).all()
    assert torch.isfinite(fe).all()
    assert torch.isfinite(gc).all()


def test_hybrid_node_dims_match_structural_plus_768():
    assert HYBRID_NODE_DIMS["class"] == 3 + 768
    assert HYBRID_NODE_DIMS["method"] == 5 + 768
    assert HYBRID_NODE_DIMS["function"] == 5 + 768
    assert HYBRID_NODE_DIMS["module"] == 1
