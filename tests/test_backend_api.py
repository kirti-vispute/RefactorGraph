# -*- coding: utf-8 -*-
"""Backend smoke tests: /health and /analyze against the real saved Design B
tuned checkpoint (models/hybrid_class_pool_tuned) -- no mocking of the model,
since the whole point is to catch preprocessing pipeline drift between
training and serving (shape mismatches, node-order mismatches, etc.)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from backend.app.main import app

SRC = '''
class Course:
    def get_marks(self):
        return 1

class Student:
    def __init__(self, name):
        self.name = name

    def calculate_result(self, course):
        course.get_marks()
        course.get_marks()
        return self.name

def helper():
    return 1
'''


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


def test_analyze_returns_predictions_for_every_node():
    with TestClient(app) as client:
        resp = client.post("/analyze", json={"source": SRC, "filename": "school.py"})
        assert resp.status_code == 200
        body = resp.json()

        lm_names = {p["name"] for p in body["long_method"]}
        assert lm_names == {"Course.get_marks", "Student.__init__", "Student.calculate_result", "helper"}

        fe_names = {p["name"] for p in body["feature_envy"]}
        assert fe_names == {"Course.get_marks", "Student.__init__", "Student.calculate_result"}

        gc_names = {p["name"] for p in body["god_class"]}
        assert gc_names == {"Course", "Student"}

        for task in ("long_method", "feature_envy", "god_class"):
            for p in body[task]:
                assert 0.0 <= p["probability"] <= 1.0
                assert p["predicted"] == (p["probability"] >= 0.5)
                assert p["file"] == "school.py"
                assert p["node_id"]
                assert p["explanation"]


def test_analyze_summary_counts_match_source():
    with TestClient(app) as client:
        resp = client.post("/analyze", json={"source": SRC, "filename": "school.py"})
        body = resp.json()
        summary = body["summary"]
        assert summary["filename"] == "school.py"
        assert summary["n_classes"] == 2
        assert summary["n_methods"] == 3
        assert summary["n_functions"] == 1
        assert summary["n_nodes"] > 0
        assert summary["n_edges"] > 0


def test_analyze_class_name_attribution():
    with TestClient(app) as client:
        resp = client.post("/analyze", json={"source": SRC, "filename": "school.py"})
        body = resp.json()

        by_name = {p["name"]: p for p in body["long_method"]}
        assert by_name["Course.get_marks"]["class_name"] == "Course"
        assert by_name["Student.calculate_result"]["class_name"] == "Student"
        assert by_name["helper"]["class_name"] is None

        gc_by_name = {p["name"]: p for p in body["god_class"]}
        assert gc_by_name["Course"]["class_name"] == "Course"


def test_analyze_graph_only_contains_real_node_and_edge_types():
    with TestClient(app) as client:
        resp = client.post("/analyze", json={"source": SRC, "filename": "school.py"})
        body = resp.json()
        graph = body["graph"]

        node_types = {n["type"] for n in graph["nodes"]}
        # "variable" is deliberately excluded from the graph schema (see
        # ml/graph/graph_builder.py) -- must never appear.
        assert "variable" not in node_types
        assert node_types <= {"module", "class", "method", "function", "attribute", "parameter", "import"}

        node_ids = {n["id"] for n in graph["nodes"]}
        for edge in graph["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids

        edge_types = {e["type"] for e in graph["edges"]}
        # only edges this exact source can actually produce -- SRC's only
        # method-to-method call is course.get_marks() where `course` is a
        # parameter, not `self`, which graph_builder deliberately does NOT
        # resolve to a "calls" edge (unresolvable external receiver, see
        # ml/graph/graph_builder.py docstring), so "calls" is correctly
        # absent here; see test_analyze_graph_includes_calls_edge_for_self_call.
        assert edge_types <= {"contains", "belongs_to", "calls", "uses", "accesses", "inherits", "imports"}
        assert "belongs_to" in edge_types
        assert "accesses" in edge_types or "uses" in edge_types

        # every prediction's node_id must resolve to a real graph node
        for task in ("long_method", "feature_envy", "god_class"):
            for p in body[task]:
                assert p["node_id"] in node_ids


def test_analyze_graph_includes_calls_edge_for_self_call():
    src = '''
class Greeter:
    def greet(self):
        return self.shout()

    def shout(self):
        return "hi"
'''
    with TestClient(app) as client:
        resp = client.post("/analyze", json={"source": src, "filename": "greeter.py"})
        body = resp.json()
        edge_types = {e["type"] for e in body["graph"]["edges"]}
        assert "calls" in edge_types


def test_analyze_rejects_unparseable_source():
    with TestClient(app) as client:
        resp = client.post("/analyze", json={"source": "def broken(:\n", "filename": "bad.py"})
        assert resp.status_code == 400
