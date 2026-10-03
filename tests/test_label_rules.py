# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds, is_feature_envy, is_god_class, is_long_method


def test_module_call_not_flagged_as_feature_envy():
    """Regression test for the PyExamine bug: calling an imported module's
    functions repeatedly must NOT be flagged as Feature Envy."""
    src = '''
import click

class CLI:
    def cli(self):
        click.echo("a")
        click.echo("b")
        click.echo("c")
        click.echo("d")
'''
    m = parse_source(src, path="cli.py")
    cli_cls = next(c for c in m.classes if c.name == "CLI")
    fn = cli_cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)


def test_real_feature_envy_flagged():
    src = '''
class Course:
    pass

class Student:
    def calculate_result(self, course):
        course.get_marks()
        course.get_credits()
        course.get_teacher()
        course.get_syllabus()
        return self.name
'''
    m = parse_source(src, path="fe.py")
    student = next(c for c in m.classes if c.name == "Student")
    fn = student.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert is_feature_envy(fn, m, thresholds)


def test_long_method_threshold():
    body = "\n".join(f"    x{i} = {i}" for i in range(50))
    src = f"def big():\n{body}\n"
    m = parse_source(src, path="long.py")
    fn = m.functions[0]
    thresholds = LabelThresholds(long_method_statements=30, god_class_method_count=999, god_class_loc=999)
    assert is_long_method(fn, m, thresholds)
    thresholds_high = LabelThresholds(long_method_statements=1000, god_class_method_count=999, god_class_loc=999)
    assert not is_long_method(fn, m, thresholds_high)


def test_god_class_threshold():
    methods = "\n".join(f"    def m{i}(self):\n        pass\n" for i in range(15))
    src = f"class Big:\n{methods}\n"
    m = parse_source(src, path="god.py")
    cls = m.classes[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=10, god_class_loc=1)
    assert is_god_class(cls, thresholds)
    thresholds_high = LabelThresholds(long_method_statements=999, god_class_method_count=50, god_class_loc=1)
    assert not is_god_class(cls, thresholds_high)


def test_god_class_field_branch_catches_dataprocessor_shaped_class():
    """Regression test for the God Class formula revision: a class with
    real encapsulated state (field_count>=1) and enough loc, but fewer
    methods than the size-only floor (method_count>=10), must still be
    flagged -- this is the exact DataProcessor (7 methods/211 loc/15
    fields) and pip.PipSession (7/244/3) shape that the old AND-only rule
    structurally could never classify as God Class regardless of size."""
    filler = "\n".join(f"        x{i} = {i}" for i in range(200))
    src = f'''
class Big:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3

    def m1(self):
{filler}

    def m2(self):
        pass

    def m3(self):
        pass
'''
    m = parse_source(src, path="fieldbranch.py")
    cls = m.classes[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=10, god_class_loc=192)
    assert cls.methods.__len__() < 10
    assert is_god_class(cls, thresholds)


def test_god_class_field_branch_excludes_zero_field_algorithm_class():
    """Regression test: a large, few-method class with NO instance state
    (a cohesive algorithm/mixin, e.g. Black's StringSplitter -- 9/28 TRAIN
    classes in this exact shape) must stay excluded even though it clears
    the loc floor -- field_count>=1 is what distinguishes a state-heavy
    blob from a complex-but-cohesive algorithm."""
    filler = "\n".join(f"        x{i} = {i}" for i in range(200))
    src = f'''
class Algorithm:
    def run(self):
{filler}

    def helper(self):
        pass
'''
    m = parse_source(src, path="nofield.py")
    cls = m.classes[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=10, god_class_loc=192)
    assert cls.methods.__len__() < 10
    assert not is_god_class(cls, thresholds)


def test_god_class_field_branch_requires_loc_floor():
    """Regression test: field_count alone (without also clearing the loc
    floor) must not trigger the field branch -- a small class with a
    handful of attributes is not a God Class just because it has state."""
    src = '''
class Small:
    def __init__(self):
        self.a = 1
        self.b = 2

    def m(self):
        pass
'''
    m = parse_source(src, path="smallfields.py")
    cls = m.classes[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=10, god_class_loc=192)
    assert not is_god_class(cls, thresholds)


def test_god_class_field_branch_requires_minimum_methods():
    """Regression test: a near-zero-method giant class with lots of state
    is a Data Class (a different antipattern), not a God Class -- the
    field branch requires god_class_min_methods_for_field_branch to avoid
    mislabeling it."""
    filler = "\n".join(f"        self.x{i} = {i}" for i in range(200))
    src = f'''
class HugeDataBag:
    def __init__(self):
{filler}
'''
    m = parse_source(src, path="dataclass.py")
    cls = m.classes[0]
    thresholds = LabelThresholds(
        long_method_statements=999, god_class_method_count=10, god_class_loc=192,
        god_class_min_fields=1, god_class_min_methods_for_field_branch=3,
    )
    assert len(cls.methods) < 3
    assert cls.class_attrs.__len__() >= 1
    assert not is_god_class(cls, thresholds)


def test_local_variable_not_flagged_as_feature_envy():
    """Regression test for the manual-review finding: a locally-created
    variable (list/dict built inside the method) is not a collaborator
    object, so calling methods on it must not count as Feature Envy."""
    src = '''
class Thing:
    def build(self):
        result = {}
        result["a"] = 1
        result["b"] = 2
        result.update({"c": 3})
        result.update({"d": 4})
        return result
'''
    m = parse_source(src, path="local.py")
    cls = m.classes[0]
    fn = cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)


def test_loop_variable_over_self_collection_not_flagged():
    """Regression test: iterating self's own collection and touching the
    loop variable's attributes is normal, not Feature Envy."""
    src = '''
class Router:
    def find(self):
        for rule in self.rules:
            rule.match()
            rule.match()
            rule.match()
            rule.match()
        return None
'''
    m = parse_source(src, path="loop.py")
    cls = m.classes[0]
    fn = cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)


def test_statement_count_immune_to_long_docstring():
    """Regression test for the manual-review finding: raw LOC counted a
    100-line docstring as if it were 100 lines of logic. statement_count
    must not inflate on documentation."""
    src = '''
def documented():
    """
    ''' + "\n    ".join(f"line {i} of docs" for i in range(80)) + '''
    """
    return 1
'''
    m = parse_source(src, path="doc.py")
    fn = m.functions[0]
    assert fn.loc > 50
    assert fn.statement_count <= 2


def test_own_class_name_reference_not_flagged_as_feature_envy():
    """Regression test: `ClassName.foo()` inside a method of ClassName is the
    class referring to its own class-level state (a common alternative to
    self/cls), not envy of another class."""
    src = '''
class Subprocess:
    def set_exit_callback(self, callback):
        self._exit_callback = callback
        Subprocess.initialize()
        Subprocess._waiting[self.pid] = self
        Subprocess._try_cleanup_process(self.pid)
'''
    m = parse_source(src, path="ownclass.py")
    cls = m.classes[0]
    fn = cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)


def test_generic_dict_methods_not_flagged_as_feature_envy():
    """Regression test: calling generic dict/kwargs methods (.get/.setdefault)
    on a parameter is normal parameter parsing, not Feature Envy — the
    receiver isn't a domain object with behavior being misappropriated."""
    src = '''
class Provider:
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("default", self.default)
        kwargs.setdefault("ensure_ascii", self.ensure_ascii)
        kwargs.setdefault("sort_keys", self.sort_keys)
        return kwargs
'''
    m = parse_source(src, path="genericdict.py")
    cls = m.classes[0]
    fn = cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)


def test_generic_dict_attribute_access_not_double_counted():
    """Regression test: visit_Attribute fires for `x.get` in addition to
    visit_Call for `x.get(...)` — the generic-method filter must apply to
    both, not just the call side, or it silently leaks through."""
    src = '''
class Provider:
    def build(self, event_hooks):
        self._event_hooks = {
            "request": list(event_hooks.get("request", [])),
            "response": list(event_hooks.get("response", [])),
        }
'''
    m = parse_source(src, path="doublecount.py")
    cls = m.classes[0]
    fn = cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)


def test_classmethod_factory_not_flagged_as_feature_envy():
    """Regression test: an alternate-constructor classmethod (`from_x`)
    exists specifically to read an external object and build cls/self from
    it — that is its contract, not misplaced logic."""
    src = '''
class Extension:
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool("ENABLED"):
            raise NotConfigured
        o = cls(crawler.stats)
        crawler.signals.connect(o.closed)
        return o
'''
    m = parse_source(src, path="classmethod.py")
    cls = m.classes[0]
    fn = cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)


def test_except_variable_not_flagged_as_feature_envy():
    """Regression test: `except OSError as e:` binds e as a local name —
    accessing e.args on a caught exception is normal error handling, not
    envy of another class."""
    src = '''
class Pidfile:
    def validate(self):
        try:
            os.kill(self.wpid, 0)
        except OSError as e:
            if e.args[0] == 1:
                return self.wpid
            if e.args[0] == 2:
                return
            raise
'''
    m = parse_source(src, path="except.py")
    cls = m.classes[0]
    fn = cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)


def test_init_not_flagged_as_feature_envy():
    src = '''
class Wrapper:
    def __init__(self, other):
        other.setup()
        other.configure()
        other.validate()
        other.start()
'''
    m = parse_source(src, path="init.py")
    cls = m.classes[0]
    fn = cls.methods[0]
    thresholds = LabelThresholds(long_method_statements=999, god_class_method_count=999, god_class_loc=999)
    assert not is_feature_envy(fn, m, thresholds)
