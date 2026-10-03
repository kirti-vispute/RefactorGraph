# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.graph.graph_builder import build_hetero_graph
from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds

LOW_THRESHOLDS = LabelThresholds(long_method_statements=1000, god_class_method_count=1000, god_class_loc=1000)


def test_empty_module_produces_empty_graph():
    m = parse_source("", path="empty.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    assert result.stats["n_class"] == 0
    assert result.stats["n_method"] == 0
    assert result.data["module"].x.shape == (1, 1)


def test_node_counts_match_source():
    src = '''
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
    m = parse_source(src, path="school.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    assert result.stats["n_class"] == 2
    assert result.stats["n_method"] == 3  # get_marks, __init__, calculate_result
    assert result.stats["n_function"] == 1  # helper
    assert result.data["class"].x.shape[0] == 2
    assert result.data["method"].x.shape[0] == 3
    assert result.data["function"].x.shape[0] == 1


def test_self_attribute_use_edge_exists():
    src = '''
class Point:
    def __init__(self, x):
        self.x = x

    def get_x(self):
        return self.x
'''
    m = parse_source(src, path="point.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    key = ("method", "uses", "attribute")
    assert key in result.data.edge_types
    assert result.data[key].edge_index.shape[1] >= 1


def test_parameter_access_edge_for_feature_envy_style_method():
    src = '''
class Course:
    pass

class Student:
    def calculate_result(self, course):
        course.get_marks()
        course.get_credits()
        return self.name
'''
    m = parse_source(src, path="fe.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    key = ("method", "accesses", "parameter")
    assert key in result.data.edge_types
    assert result.data[key].edge_index.shape[1] >= 1


def test_calls_edge_resolved_within_module():
    src = '''
class Foo:
    def a(self):
        return self.b()

    def b(self):
        return 1
'''
    m = parse_source(src, path="calls.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    key = ("method", "calls", "method")
    assert key in result.data.edge_types
    edge_index = result.data[key].edge_index
    assert edge_index.shape[1] == 1
    # a is index 0, b is index 1 (declaration order)
    assert edge_index[0, 0].item() == 0
    assert edge_index[1, 0].item() == 1


def test_inherits_edge_resolved_within_module():
    src = '''
class Base:
    pass

class Child(Base):
    pass
'''
    m = parse_source(src, path="inherit.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    key = ("class", "inherits", "class")
    assert key in result.data.edge_types
    edge_index = result.data[key].edge_index
    assert edge_index.shape[1] == 1
    assert edge_index[0, 0].item() == 1  # Child
    assert edge_index[1, 0].item() == 0  # Base


def test_unresolved_base_class_produces_no_inherits_edge():
    src = '''
class Child(SomeExternalBase):
    pass
'''
    m = parse_source(src, path="extbase.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    assert ("class", "inherits", "class") not in result.data.edge_types


def test_belongs_to_edge_is_reverse_of_contains_method():
    src = '''
class Course:
    def get_marks(self):
        return 1

class Student:
    def __init__(self, name):
        self.name = name

    def calculate_result(self, course):
        return self.name
'''
    m = parse_source(src, path="school.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    contains_key = ("class", "contains", "method")
    belongs_key = ("method", "belongs_to", "class")
    assert belongs_key in result.data.edge_types
    contains_edges = result.data[contains_key].edge_index
    belongs_edges = result.data[belongs_key].edge_index
    assert belongs_edges.shape[1] == contains_edges.shape[1] == 3
    # belongs_to(method, class) pairs must be exactly contains(class, method) reversed
    contains_pairs = set(zip(contains_edges[0].tolist(), contains_edges[1].tolist()))
    belongs_pairs = set(zip(belongs_edges[1].tolist(), belongs_edges[0].tolist()))
    assert contains_pairs == belongs_pairs


def test_labels_not_present_in_feature_vectors():
    """Structural check that the label booleans never leak into node
    features — feature width for method/function/class nodes is fixed and
    does not grow when labels flip."""
    src = '''
class Big:
    def m1(self):
        pass
    def m2(self):
        pass
'''
    m = parse_source(src, path="nolabelleak.py")
    result_low = build_hetero_graph(m, LOW_THRESHOLDS)
    tight_thresholds = LabelThresholds(long_method_statements=0, god_class_method_count=0, god_class_loc=0)
    result_high = build_hetero_graph(m, tight_thresholds)
    assert result_low.data["method"].x.shape == result_high.data["method"].x.shape
    assert torch.equal(result_low.data["method"].x, result_high.data["method"].x)
    assert not torch.equal(result_low.data["method"].y_long_method, result_high.data["method"].y_long_method)


def test_nested_closure_becomes_a_real_function_node_with_contains_edge():
    """Regression for docs/robustness_audit_phase_a.md: a closure defined
    inside a method was previously silently absent from the graph entirely
    (0 function nodes for it, no edge, no error). It must now be a real
    `function` node reached by a (method, contains, function) edge from its
    enclosing method."""
    src = '''
class Outer:
    def method_with_closure(self):
        def helper(x):
            return x + 1
        return helper(1)
'''
    m = parse_source(src, path="closure.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    assert result.stats["n_function"] == 1
    key = ("method", "contains", "function")
    assert key in result.data.edge_types
    edge_index = result.data[key].edge_index
    assert edge_index.shape[1] == 1
    assert edge_index[0, 0].item() == 0  # method_with_closure is method index 0
    assert edge_index[1, 0].item() == 0  # helper is function index 0


def test_call_to_own_nested_closure_resolves_to_a_calls_edge():
    src = '''
def top_level():
    def helper():
        return 1
    return helper()
'''
    m = parse_source(src, path="closurecall.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    key = ("function", "calls", "function")
    assert key in result.data.edge_types
    edge_index = result.data[key].edge_index
    assert edge_index.shape[1] == 1


def test_same_named_closures_do_not_cross_resolve_calls():
    """Regression: a bare-name call to a nested closure must only resolve
    within its own defining scope -- two different top-level functions each
    defining their own local "helper" must not have their calls cross-wired
    to the wrong one."""
    src = '''
def top_a():
    def helper():
        return 1
    return helper()

def top_b():
    def helper():
        return 2
    return helper()
'''
    m = parse_source(src, path="noncollide.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    key = ("function", "calls", "function")
    edge_index = result.data[key].edge_index
    # exactly 2 calls edges (each top-level function calling its OWN
    # helper), not 4 (which would mean cross-resolution to the wrong one)
    assert edge_index.shape[1] == 2
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # function node order (declaration/visit order): helper(top_a)=0,
    # top_a=1, helper(top_b)=2, top_b=3
    assert (1, 0) in pairs  # top_a -> its own helper
    assert (3, 2) in pairs  # top_b -> its own helper
    assert (1, 2) not in pairs  # top_a must NOT resolve to top_b's helper
    assert (3, 0) not in pairs  # top_b must NOT resolve to top_a's helper


def test_nested_class_gets_contains_edge_from_true_parent_not_module():
    """Regression: a class nested inside another class (e.g. `class Meta:`)
    previously got a (module, contains, class) edge, misrepresenting the
    module as its direct container instead of the real enclosing class."""
    src = '''
class Outer:
    class Inner:
        def inner_method(self):
            return 1
'''
    m = parse_source(src, path="nested_class.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    outer_idx = next(i for i, c in enumerate(m.classes) if c.name == "Outer")
    inner_idx = next(i for i, c in enumerate(m.classes) if c.name == "Inner")

    class_contains_class = result.data[("class", "contains", "class")].edge_index
    pairs = set(zip(class_contains_class[0].tolist(), class_contains_class[1].tolist()))
    assert (outer_idx, inner_idx) in pairs

    if ("module", "contains", "class") in result.data.edge_types:
        module_contains_class = result.data[("module", "contains", "class")].edge_index
        module_dsts = set(module_contains_class[1].tolist())
        assert inner_idx not in module_dsts  # Inner must not ALSO be wired from the module
        assert outer_idx in module_dsts  # Outer, the true top-level class, still is


def test_same_name_meta_inner_classes_do_not_collide():
    """Regression: class_index was previously keyed by simple name only --
    two unrelated `class Meta:` inner classes (a ubiquitous Django/DRF
    pattern) would collide in that index."""
    src = '''
class Article:
    class Meta:
        ordering = ["-created"]

class Author:
    class Meta:
        ordering = ["name"]
'''
    m = parse_source(src, path="meta_collision.py")
    result = build_hetero_graph(m, LOW_THRESHOLDS)
    assert result.stats["n_class"] == 4  # Article, Article.Meta, Author, Author.Meta
    class_contains_class = result.data[("class", "contains", "class")].edge_index
    assert class_contains_class.shape[1] == 2  # each Meta belongs to its OWN outer class only
