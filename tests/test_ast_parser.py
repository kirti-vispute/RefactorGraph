# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocessing.ast_parser import parse_source


def test_empty_file():
    m = parse_source("", path="empty.py")
    assert m.ok
    assert m.classes == []
    assert m.functions == []
    assert m.loc == 0


def test_syntax_error():
    m = parse_source("def f(:\n  pass", path="bad.py")
    assert not m.ok
    assert "SyntaxError" in m.parse_error


def test_unsupported_syntax_py2_print():
    m = parse_source("print 'hello'", path="py2.py")
    assert not m.ok


def test_nested_classes_and_functions():
    src = '''
class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        def helper():
            return 1
        return helper()
'''
    m = parse_source(src, path="nested.py")
    assert m.ok
    names = {c.name for c in m.classes}
    assert names == {"Outer", "Inner"}
    outer = next(c for c in m.classes if c.name == "Outer")
    method_names = {f.name for f in outer.methods}
    assert "outer_method" in method_names
    outer_method = next(f for f in outer.methods if f.name == "outer_method")
    assert [nf.name for nf in outer_method.nested_functions] == ["helper"]


def test_decorators_and_inheritance():
    src = '''
class Base:
    pass

class Child(Base):
    @staticmethod
    def util():
        pass

    @property
    def value(self):
        return self._value
'''
    m = parse_source(src, path="deco.py")
    child = next(c for c in m.classes if c.name == "Child")
    assert child.bases == ["Base"]
    util = next(f for f in child.methods if f.name == "util")
    assert util.decorators == ["staticmethod"]
    value = next(f for f in child.methods if f.name == "value")
    assert value.decorators == ["property"]


def test_calls_and_attribute_access_module_vs_self():
    src = '''
import click

class Greeter:
    def greet(self, other):
        click.echo("hi")
        self.name = "x"
        return other.name
'''
    m = parse_source(src, path="calls.py")
    assert "click" in m.imported_names()
    greeter = next(c for c in m.classes if c.name == "Greeter")
    greet = next(f for f in greeter.methods if f.name == "greet")
    receivers = {c.receiver for c in greet.calls}
    assert "click" in receivers
    attr_receivers = {a.receiver for a in greet.attr_accesses}
    assert {"self", "other"}.issubset(attr_receivers)
    assert "name" in greeter.class_attrs


def test_self_attribute_tracked_as_class_attr():
    src = '''
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
'''
    m = parse_source(src, path="point.py")
    point = next(c for c in m.classes if c.name == "Point")
    assert point.class_attrs == {"x", "y"}


def test_nested_closure_is_reachable_and_correctly_scoped():
    """Regression for docs/robustness_audit_phase_a.md finding: a nested
    function/closure was silently dropped from mod.functions entirely
    (never reachable as a graph node) even though it was correctly recorded
    on its parent's nested_functions list."""
    src = '''
class Outer:
    def method_with_closure(self):
        def helper(x):
            return x + 1
        return helper(1)
'''
    m = parse_source(src, path="closure.py")
    names = [(f.name, f.qualname, f.parent_qualname) for f in m.functions]
    assert names == [("helper", "Outer.method_with_closure.<locals>.helper", "Outer.method_with_closure")]


def test_same_named_closures_in_different_scopes_get_distinct_qualnames():
    """Two different methods each defining their own local "helper" must
    not collide -- distinct scopes need distinct qualified identities."""
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
    m = parse_source(src, path="collide.py")
    closures = [f for f in m.functions if f.name == "helper"]
    assert len(closures) == 2
    assert {f.qualname for f in closures} == {"top_a.<locals>.helper", "top_b.<locals>.helper"}
    assert {f.parent_qualname for f in closures} == {"top_a", "top_b"}


def test_nested_class_gets_correct_parent_qualname():
    """Regression: a class nested inside another class (e.g. `class Meta:`)
    was previously flattened with no record of its true parent."""
    src = '''
class Outer:
    class Inner:
        def inner_method(self):
            return 1
'''
    m = parse_source(src, path="nested_class.py")
    outer = next(c for c in m.classes if c.name == "Outer")
    inner = next(c for c in m.classes if c.name == "Inner")
    assert outer.parent_qualname is None
    assert outer.qualname == "Outer"
    assert inner.parent_qualname == "Outer"
    assert inner.qualname == "Outer.Inner"


def test_method_of_class_defined_inside_a_function_is_still_a_method():
    """Regression: is_method previously required the ENTIRE stack to have
    no function anywhere on it, not just "no function between here and the
    nearest class" -- a real instance method of a class returned by a
    factory function was misclassified as a bare function."""
    src = '''
def factory():
    class Dynamic:
        def real_method(self):
            return 1
    return Dynamic
'''
    m = parse_source(src, path="factory.py")
    dynamic = next(c for c in m.classes if c.name == "Dynamic")
    assert dynamic.parent_qualname == "factory"
    method = dynamic.methods[0]
    assert method.is_method is True
    assert method.class_name == "Dynamic"
    assert method.qualname == "Dynamic.real_method"


def test_missing_source_file(tmp_path):
    from ml.preprocessing.ast_parser import parse_file

    missing = tmp_path / "does_not_exist.py"
    m = parse_file(missing)
    assert not m.ok


def test_struct_hash_ignores_formatting_and_names_differ():
    a = parse_source("def f():\n    x = 1\n    return x\n", path="a.py")
    b = parse_source("def f():\n\n    x = 1\n    return x   \n", path="b.py")
    c = parse_source("def f():\n    y = 2\n    return y\n", path="c.py")
    assert a.functions[0].struct_hash == b.functions[0].struct_hash
    assert a.functions[0].struct_hash != c.functions[0].struct_hash


def test_very_large_file_parses():
    src = "\n".join(f"def f{i}():\n    return {i}" for i in range(2000))
    m = parse_source(src, path="large.py")
    assert m.ok
    assert len(m.functions) == 2000
