# -*- coding: utf-8 -*-
"""Verifies ml/graph/node_naming.py's node ordering/records against the same
fixture and expectations already used by
tests/test_explain.py::test_build_name_index_matches_construction_order --
this is a standalone copy of that logic (see module docstring), so it needs
its own equivalent check."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.graph.node_naming import build_name_index, build_node_records
from ml.preprocessing.ast_parser import parse_source

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


def test_build_name_index_matches_construction_order():
    mod = parse_source(SRC, path="school.py")
    names = build_name_index(mod, "school.py")
    assert names["module"] == ["school.py"]
    assert names["class"] == ["Course", "Student"]
    assert names["method"] == ["Course.get_marks", "Student.__init__", "Student.calculate_result"]
    assert names["function"] == ["helper"]
    assert names["parameter"] == ["Student.__init__.name", "Student.calculate_result.course"]
    assert names["attribute"] == ["Student.name"]
    assert names["import"] == []


def test_build_node_records_carries_real_line_numbers_only_where_available():
    mod = parse_source(SRC, path="school.py")
    records = build_node_records(mod, "school.py")

    course = records["class"][0]
    assert course["name"] == "Course"
    assert course["lineno"] == mod.classes[0].lineno
    assert course["end_lineno"] == mod.classes[0].end_lineno

    calc = records["method"][2]
    assert calc["name"] == "Student.calculate_result"
    assert calc["lineno"] == mod.classes[1].methods[1].lineno

    # module/attribute/parameter/import have no source-text unit of their
    # own -- must never be fabricated.
    for p in records["parameter"]:
        assert p["lineno"] is None and p["end_lineno"] is None
    for a in records["attribute"]:
        assert a["lineno"] is None and a["end_lineno"] is None
