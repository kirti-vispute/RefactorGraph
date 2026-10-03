# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader

from ml.graph.graph_builder import build_hetero_graph
from ml.models.gat_baseline import EDGE_TYPES_CLASS_AGGREGATION, HeteroGAT
from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds

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


def _build_graph():
    m = parse_source(SRC, path="school.py")
    return build_hetero_graph(m, LOW_THRESHOLDS).data


def test_forward_pass_produces_finite_logits_for_all_tasks():
    data = _build_graph()
    model = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0)
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


def test_module_embedding_survives_despite_never_being_a_destination():
    """module is never the dst of any edge type, so HeteroConv alone would
    drop it after layer 1 — the h_dict.update() pattern in HeteroGAT.forward
    must preserve it instead of losing it as a source for layer 2 edges like
    (module, contains, class)."""
    data = _build_graph()
    model = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        h = model(data.x_dict, data.edge_index_dict)
    assert "module" in h
    assert h["module"].shape == (1, 16)


def _class_output_grad_wrt_method_input(edge_types):
    data = _build_graph()
    data["method"].x = data["method"].x.clone().requires_grad_(True)
    model = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0, edge_types=edge_types)
    model.eval()
    h = model(data.x_dict, data.edge_index_dict)
    gc = model.predict_god_class(h)
    gc.sum().backward()
    return data["method"].x.grad


def test_default_edge_types_class_cannot_see_method_gradients():
    """Documents the Phase 13 finding: under the original schema, (class,
    contains, method) only ever makes method the destination, so class is
    never reachable FROM method — god_class predictions get zero gradient
    w.r.t. method's own input features."""
    grad = _class_output_grad_wrt_method_input(edge_types=None)
    # method.x is never on the path from god_class's output back to the
    # graph's leaves under the original schema, so autograd never visits it
    # and .grad stays None (not a zero tensor) — itself the proof of no path.
    assert grad is None


def test_class_aggregation_edge_types_let_class_see_method_gradients():
    """With EDGE_TYPES_CLASS_AGGREGATION (adds the method->belongs_to->class
    reverse edge), god_class predictions must get nonzero gradient w.r.t.
    method's own input features — direct evidence class now aggregates
    information from its method children."""
    grad = _class_output_grad_wrt_method_input(edge_types=EDGE_TYPES_CLASS_AGGREGATION)
    assert grad is not None
    assert torch.any(grad != 0)


def _class_output_grad_wrt_method_input_pool():
    data = _build_graph()
    data["method"].x = data["method"].x.clone().requires_grad_(True)
    model = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0, class_method_pool=True)
    model.eval()
    h = model(data.x_dict, data.edge_index_dict)
    gc = model.predict_god_class(h)
    gc.sum().backward()
    return data["method"].x.grad


def test_class_pool_lets_god_class_see_method_gradients():
    """class_method_pool=True (Design B) reuses the existing (class,
    contains, method) edge_index to mean-pool method embeddings into the
    god_class head's input, without adding any edge type to the backbone —
    god_class predictions must still get nonzero gradient w.r.t. method's
    own input features."""
    grad = _class_output_grad_wrt_method_input_pool()
    assert grad is not None
    assert torch.any(grad != 0)


def test_class_pool_does_not_alter_method_pathway():
    """Design B's whole point is that long_method/feature_envy — which read
    h_dict["method"] — must be byte-for-byte unaffected by whether the
    god_class head also pools method embeddings, since pooling happens after
    conv2 and only feeds a separate concatenated input to head_god_class."""
    data = _build_graph()
    torch.manual_seed(0)
    model_no_pool = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0, class_method_pool=False)
    torch.manual_seed(0)
    model_pool = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0, class_method_pool=True)
    model_no_pool.eval()
    model_pool.eval()
    with torch.no_grad():
        h1 = model_no_pool(data.x_dict, data.edge_index_dict)
        h2 = model_pool(data.x_dict, data.edge_index_dict)
        assert torch.allclose(h1["method"], h2["method"])
        assert torch.allclose(h1["class"], h2["class"])
        lm1 = model_no_pool.predict_long_method(h1, "method")
        lm2 = model_pool.predict_long_method(h2, "method")
        fe1 = model_no_pool.predict_feature_envy(h1)
        fe2 = model_pool.predict_feature_envy(h2)
        assert torch.allclose(lm1, lm2)
        assert torch.allclose(fe1, fe2)


def test_batching_two_graphs_works():
    data = _build_graph()
    loader = DataLoader([data, data], batch_size=2)
    model = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0)
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        h = model(batch.x_dict, batch.edge_index_dict)
    assert h["method"].shape[0] == 2 * data["method"].x.shape[0]
    assert h["class"].shape[0] == 2 * data["class"].x.shape[0]
