# -*- coding: utf-8 -*-
"""Phase 13c targeted tests: loss-weight pass-through in
scripts/tune_hybrid_optuna.compute_batch_loss, and objective wiring in
scripts/tune_hybrid_class_pool_optuna.build_objective. Both run against a
tiny synthetic graph (same fixture pattern as tests/test_gat_baseline.py) so
they stay fast — no real graphs_hybrid data or disk I/O is touched."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
import torch

from ml.graph.graph_builder import build_hetero_graph
from ml.models.gat_baseline import HeteroGAT
from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds
from scripts.tune_hybrid_class_pool_optuna import build_objective
from scripts.tune_hybrid_optuna import compute_batch_loss

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


def _build_hybrid_graph():
    """train_one (used by build_objective) hardcodes HYBRID_NODE_DIMS
    (class/method/function padded to +768 for CodeBERT), same fixture
    pattern as tests/test_tune_hybrid_optuna.py::_fake_hybrid_graphs — the
    plain structural-only graph from _build_graph is the wrong width for
    build_objective's model construction."""
    data = _build_graph()
    for nt in ("class", "method", "function"):
        k = data[nt].x.shape[0]
        data[nt].x = torch.cat([data[nt].x, torch.zeros((k, 768))], dim=1)
    return data


def _sum_loss(pred, target):
    """Dummy loss: sum of predictions, independent of target — isolates the
    weight multiplier's effect from any real BCE numerics."""
    return pred.sum()


def test_compute_batch_loss_weight_pass_through():
    """weight_lm/weight_fe/weight_gc must scale their own term only, and the
    total must equal the manually-weighted sum of the 4 per-task
    components (method long_method, feature_envy, function long_method,
    god_class) — not e.g. all scaled by the same factor or dropped."""
    data = _build_graph()
    model = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        h = model(data.x_dict, data.edge_index_dict)

        lm_method = _sum_loss(model.predict_long_method(h, "method"), data["method"].y_long_method)
        fe = _sum_loss(model.predict_feature_envy(h), data["method"].y_feature_envy)
        lm_function = _sum_loss(model.predict_long_method(h, "function"), data["function"].y_long_method)
        gc = _sum_loss(model.predict_god_class(h), data["class"].y)

        weight_lm, weight_fe, weight_gc = 2.0, 3.0, 0.5
        expected = weight_lm * lm_method + weight_fe * fe + weight_lm * lm_function + weight_gc * gc

        loss, has_loss = compute_batch_loss(
            model, h, data, _sum_loss, _sum_loss, _sum_loss,
            weight_lm=weight_lm, weight_fe=weight_fe, weight_gc=weight_gc,
        )
        assert has_loss
        assert torch.allclose(loss, expected)


def test_compute_batch_loss_default_weights_match_unweighted_sum():
    """No weight_* args (Phase 10's original call site) must reproduce the
    plain unweighted sum byte-for-byte — backward compatibility for the
    existing tune_hybrid_optuna.py objective()/main()."""
    data = _build_graph()
    model = HeteroGAT(hidden_dim=16, heads=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        h = model(data.x_dict, data.edge_index_dict)
        loss_default, _ = compute_batch_loss(model, h, data, _sum_loss, _sum_loss, _sum_loss)
        loss_explicit, _ = compute_batch_loss(
            model, h, data, _sum_loss, _sum_loss, _sum_loss,
            weight_lm=1.0, weight_fe=1.0, weight_gc=1.0,
        )
        assert torch.allclose(loss_default, loss_explicit)


def test_build_objective_wires_loss_weight_params():
    """build_objective's trial must suggest weight_fe/weight_gc alongside
    the existing architecture params, and actually route them into
    train_one (class_method_pool=True) — a 1-trial study run end to end on
    a tiny graph/short budget is the cheapest way to prove the wiring
    reaches Optuna's search space, not just that the function is callable."""
    data = _build_hybrid_graph()
    train_graphs = [data]
    val_graphs = [data]

    objective = build_objective(
        train_graphs, val_graphs, _sum_loss, _sum_loss, _sum_loss,
        torch.device("cpu"), search_epochs=1, search_patience=1, batch_size=1,
    )

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=1)

    assert "weight_fe" in study.best_params
    assert "weight_gc" in study.best_params
    assert 0.5 <= study.best_params["weight_fe"] <= 4.0
    assert 0.5 <= study.best_params["weight_gc"] <= 4.0
    assert isinstance(study.best_value, float)


def test_build_objective_logs_per_task_f1_user_attrs():
    """Phase 13d: each trial must record long_method_f1/feature_envy_f1/
    god_class_f1/macro_f1 as user attrs, so optuna_trials.csv carries the
    per-task breakdown needed to audit whether a weight_fe/weight_gc choice
    trades one task off against another — not just the scalar macro-F1
    objective, which can hide that trade-off."""
    data = _build_hybrid_graph()
    train_graphs = [data]
    val_graphs = [data]

    objective = build_objective(
        train_graphs, val_graphs, _sum_loss, _sum_loss, _sum_loss,
        torch.device("cpu"), search_epochs=1, search_patience=1, batch_size=1,
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=1)

    attrs = study.best_trial.user_attrs
    for key in ("long_method_f1", "feature_envy_f1", "god_class_f1", "macro_f1"):
        assert key in attrs
    assert attrs["macro_f1"] == study.best_value


def test_build_objective_pins_fixed_architecture():
    """Phase 13e: fixed_heads/fixed_hidden_dim_per_head must remove those
    two params from the trial's search space entirely (not just bias
    toward one value) — study.best_params must NOT contain 'heads' or
    'hidden_dim_per_head' when fixed, since train_one is called with the
    literal fixed_heads/fixed_hidden_dim_per_head*fixed_heads values, never
    a trial.suggest_categorical result."""
    data = _build_hybrid_graph()
    train_graphs = [data]
    val_graphs = [data]

    objective = build_objective(
        train_graphs, val_graphs, _sum_loss, _sum_loss, _sum_loss,
        torch.device("cpu"), search_epochs=1, search_patience=1, batch_size=1,
        fixed_heads=4, fixed_hidden_dim_per_head=32,
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=1)

    assert "heads" not in study.best_params
    assert "hidden_dim_per_head" not in study.best_params
    assert "dropout" in study.best_params
    assert "weight_fe" in study.best_params
    assert "weight_gc" in study.best_params
