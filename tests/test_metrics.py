# -*- coding: utf-8 -*-
"""Regression tests for ml/preprocessing/metrics.py, added during the
robustness audit (docs/robustness_audit_phase_a.md) after finding
`external_access_count`/`dominant_external_count` silently double-counted
every call-based interaction (`x.method()` fired both visit_Call and
visit_Attribute for the same syntactic expression)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocessing.ast_parser import parse_source
from ml.preprocessing.label_rules import LabelThresholds, is_feature_envy
from ml.preprocessing.metrics import function_metrics


def test_external_call_counted_once_not_twice():
    src = '''
class Outer:
    def demo(self, service):
        service.validate()
        service.process()
        service.finish()
        return 1
'''
    mod = parse_source(src, path="demo.py")
    fn = mod.classes[0].methods[0]
    fm = function_metrics(fn, mod)
    # 3 distinct calls on `service` -- must count as 3, not 6 (the old bug:
    # each `x.method()` fired both visit_Call and visit_Attribute for the
    # exact same syntactic location).
    assert fm.external_access_count == 3
    assert fm.dominant_external_count == 3
    assert fm.dominant_external_receiver == "service"


def test_pure_attribute_read_still_counts_once():
    src = '''
class Outer:
    def demo(self, service):
        return service.status
'''
    mod = parse_source(src, path="demo.py")
    fn = mod.classes[0].methods[0]
    fm = function_metrics(fn, mod)
    assert fm.external_access_count == 1


def test_self_call_counted_once_not_twice():
    src = '''
class Outer:
    def helper(self):
        return 1

    def demo(self):
        self.helper()
        self.helper()
        return 1
'''
    mod = parse_source(src, path="demo.py")
    demo = [m for m in mod.classes[0].methods if m.name == "demo"][0]
    fm = function_metrics(demo, mod)
    assert fm.self_access_count == 2


def test_mixed_call_and_plain_attribute_on_same_receiver():
    src = '''
class Outer:
    def demo(self, service):
        service.validate()
        return service.status
'''
    mod = parse_source(src, path="demo.py")
    fn = mod.classes[0].methods[0]
    fm = function_metrics(fn, mod)
    # 1 call + 1 plain attribute read on the same receiver -- must be 2, not
    # 3 (the call must count once, the plain read once).
    assert fm.external_access_count == 2


def test_delegating_through_owned_collaborator_is_not_counted_as_self_access():
    """Regression for a real false-negative found via a user-supplied sample
    file (DataProcessor.save_to_database): a method that ONLY delegates
    through `self.<collaborator>.method()` calls -- the single most common
    real-world Feature Envy shape -- was scoring self_access_count as high
    as external_access_count (one inflated self-hit per delegated call, from
    the inner `self.database_connection` attribute read used to reach the
    call), making `dominant_external_count > self_access_count` nearly
    impossible to satisfy. `self.database_connection` here is a stepping
    stone to `.execute()`/etc, not a standalone self-data access."""
    src = '''
class Repo:
    def save(self, user):
        self.database_connection.cursor()
        self.database_connection.execute(user)
        self.database_connection.commit()
        self.cache.set(user)
        self.cache.cleanup_expired()
'''
    mod = parse_source(src, path="demo.py")
    fn = mod.classes[0].methods[0]
    fm = function_metrics(fn, mod)
    assert fm.self_access_count == 0
    assert fm.external_access_count == 5
    assert fm.dominant_external_receiver == "self.database_connection"
    assert fm.dominant_external_count == 3

    th = LabelThresholds(long_method_statements=15, god_class_method_count=10, god_class_loc=192)
    assert is_feature_envy(fn, mod, th) is True


def test_genuine_self_attribute_read_still_counts_as_self_access():
    """A plain `self.x` read that is NOT the base of a longer chain (no
    further `.y`/`.z()` on the result) must still count as a real
    self-access -- the chained-base fix must not suppress ordinary
    self-attribute reads/assignments."""
    src = '''
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def shift(self, dx):
        self.x = self.x + dx
        return self.x
'''
    mod = parse_source(src, path="demo.py")
    shift = [m for m in mod.classes[0].methods if m.name == "shift"][0]
    fm = function_metrics(shift, mod)
    # assign target `self.x`, read `self.x` on the RHS, and `return self.x`
    # -- three genuine, non-chained self-attribute accesses.
    assert fm.self_access_count == 3
    assert fm.external_access_count == 0
