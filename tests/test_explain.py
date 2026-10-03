# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.graph.graph_builder import build_hetero_graph
from ml.models.gat_baseline import EDGE_TYPES, HeteroGAT
from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds
from torch_geometric.explain import Explainer, GNNExplainer

from scripts.explain import (
    TaskWrapper,
    build_name_index,
    find_node_index,
    prune_to_receptive_field,
    receptive_field,
)
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


def _fake_hybrid_graph_and_mod():
    mod = parse_source(SRC, path="school.py")
    data = build_hetero_graph(mod, LOW_THRESHOLDS).data
    for nt in ("class", "method", "function"):
        k = data[nt].x.shape[0]
        data[nt].x = torch.cat([data[nt].x, torch.zeros((k, 768))], dim=1)
    return mod, data


def test_receptive_field_excludes_unreachable_function():
    node_types, edge_types = receptive_field("method", EDGE_TYPES, hops=2)
    assert "function" not in node_types
    assert ("module", "contains", "function") not in edge_types
    assert "method" in node_types and "class" in node_types and "module" in node_types


def test_receptive_field_for_class_excludes_method():
    node_types, edge_types = receptive_field("class", EDGE_TYPES, hops=2)
    assert "method" not in node_types
    assert "module" in node_types


def test_prune_to_receptive_field_drops_disconnected_types():
    _, data = _fake_hybrid_graph_and_mod()
    node_types, edge_types = receptive_field("method", EDGE_TYPES, hops=2)
    x_sub, ei_sub = prune_to_receptive_field(data, node_types, edge_types)
    assert "function" not in x_sub
    for et in ei_sub:
        assert et[0] in x_sub and et[2] in x_sub


def test_find_node_index_matches_qualname_and_lineno():
    mod, _ = _fake_hybrid_graph_and_mod()
    # global method order: Course.get_marks=0, Student.__init__=1, Student.calculate_result=2
    idx = find_node_index(mod, "method", "Student.calculate_result", mod.classes[1].methods[1].lineno)
    assert idx == 2
    idx0 = find_node_index(mod, "method", "Course.get_marks", mod.classes[0].methods[0].lineno)
    assert idx0 == 0


def test_build_name_index_matches_construction_order():
    mod, _ = _fake_hybrid_graph_and_mod()
    names = build_name_index(mod, "school.py")
    assert names["class"] == ["Course", "Student"]
    assert names["method"] == ["Course.get_marks", "Student.__init__", "Student.calculate_result"]
    assert names["function"] == ["helper"]
    assert names["parameter"] == ["Student.__init__.name", "Student.calculate_result.course"]


def test_gnn_explainer_end_to_end_on_pruned_graph():
    mod, data = _fake_hybrid_graph_and_mod()
    model = HeteroGAT(hidden_dim=8, heads=2, dropout=0.0, node_feature_dims=HYBRID_NODE_DIMS)
    model.eval()
    wrapper = TaskWrapper(model, "long_method", "method")

    node_types, edge_types = receptive_field("method", EDGE_TYPES, hops=2)
    x_sub, ei_sub = prune_to_receptive_field(data, node_types, edge_types)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=5),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )
    idx = find_node_index(mod, "method", "Student.calculate_result", mod.classes[1].methods[1].lineno)
    explanation = explainer(x_sub, ei_sub, index=idx)

    assert explanation["method"].node_mask.shape == x_sub["method"].shape
    for et in explanation.edge_types:
        assert explanation[et].edge_mask.shape[0] == ei_sub[et].shape[1]
