# -*- coding: utf-8 -*-
"""Rule-based (silver) label derivation for the three target smells, built
directly on the metrics in metrics.py rather than on a third-party tool.

Thresholds are corpus-relative percentiles, not fabricated fixed constants:
computed once over the whole training-repo pool and recorded in the dataset
report so the exact cut points are documented and reproducible. An absolute
floor is kept on each threshold (from Fowler/Lanza-style common practice) so
a tiny/atypical corpus can't push the percentile threshold down to something
trivial like a 5-line method.

These are SILVER labels: rule-based, not human-annotated. A stratified
sample must be manually reviewed (see scripts/build_dataset.py) and the
measured precision reported alongside any results — never presented as
ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ml.preprocessing.ast_parser import ClassInfo, FunctionInfo, ModuleInfo
from ml.preprocessing.metrics import class_metrics, function_metrics


@dataclass
class LabelThresholds:
    long_method_statements: int
    god_class_method_count: int
    god_class_loc: int
    feature_envy_min_external_calls: int = 3
    # Not corpus-percentile-derived like the fields above -- these are
    # categorical floors for the field-driven God Class branch (see
    # is_god_class). Investigated against the 9-repo TRAIN pool
    # (scratch_external_eval/godclass_formula_investigation*.py, not
    # committed as part of the pipeline, kept as an audit trail): among the
    # 28 TRAIN classes that clear the LOC floor but fall under the
    # method_count floor, field_count==0 cleanly separates 9 algorithm-only
    # classes (Black's StringSplitter/StringMerger/StringParenWrapper/
    # BaseStringSplitter, tornado's OAuthMixin/OpenIdMixin, pytest's
    # pytestPDB, gunicorn's TLVEncoder, requests' SessionRedirectMixin --
    # all genuinely cohesive, no shared state) from 19 classes with real
    # encapsulated state (tornado's HTTPRequest/34 fields, gunicorn's
    # UWSGIRequest/21, Message/14, HTTPServerRequest/13, etc.) -- so
    # "field_count >= 1" is used as-is, not tuned to any specific example.
    # god_class_min_methods_for_field_branch guards against a near-zero-
    # method giant data container (Data Class smell, a different
    # antipattern) being mislabeled God Class; no such case exists in TRAIN,
    # kept as a defensive floor for corpora this project hasn't seen.
    god_class_min_fields: int = 1
    god_class_min_methods_for_field_branch: int = 3

    @classmethod
    def from_corpus(
        cls,
        method_statement_counts: Sequence[int],
        class_method_counts: Sequence[int],
        class_locs: Sequence[int],
        percentile: float = 0.90,
        min_long_method_statements: int = 15,
        min_god_class_methods: int = 10,
        min_god_class_loc: int = 150,
        min_god_class_fields: int = 1,
        min_god_class_methods_for_field_branch: int = 3,
    ) -> "LabelThresholds":
        return cls(
            long_method_statements=max(_percentile(method_statement_counts, percentile), min_long_method_statements),
            god_class_method_count=max(_percentile(class_method_counts, percentile), min_god_class_methods),
            god_class_loc=max(_percentile(class_locs, percentile), min_god_class_loc),
            god_class_min_fields=min_god_class_fields,
            god_class_min_methods_for_field_branch=min_god_class_methods_for_field_branch,
        )


def _percentile(values: Sequence[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def is_long_method(fn: FunctionInfo, module: ModuleInfo, thresholds: LabelThresholds) -> bool:
    return function_metrics(fn, module).statement_count >= thresholds.long_method_statements


def is_god_class(cls: ClassInfo, thresholds: LabelThresholds) -> bool:
    m = class_metrics(cls)
    if m.method_count >= thresholds.god_class_method_count and m.loc >= thresholds.god_class_loc:
        return True
    # Field-driven branch: a large class carrying genuine encapsulated
    # state even when method_count falls short of the size-only floor --
    # e.g. DataProcessor (7 methods/211 loc/15 fields) and pip.PipSession
    # (7/244/3), both blocked by method_count alone under the branch above.
    # See LabelThresholds docstring for how the two floors here were
    # derived (not from these two examples).
    if (
        m.loc >= thresholds.god_class_loc
        and m.field_count >= thresholds.god_class_min_fields
        and m.method_count >= thresholds.god_class_min_methods_for_field_branch
    ):
        return True
    return False


def is_feature_envy(fn: FunctionInfo, module: ModuleInfo, thresholds: LabelThresholds) -> bool:
    if not fn.is_method or fn.name in ("__init__", "__new__"):
        return False
    if "staticmethod" in fn.decorators:
        return False
    if "classmethod" in fn.decorators:
        # Alternate-constructor factory methods (`from_crawler`, `from_item`,
        # ...) exist specifically to read an external object and build
        # self/cls from it — that is their contract, not misplaced logic,
        # same reasoning as excluding __init__ above.
        return False
    m = function_metrics(fn, module)
    if m.dominant_external_count < thresholds.feature_envy_min_external_calls:
        return False
    return m.dominant_external_count > m.self_access_count


def collect_corpus_stats(modules: List[ModuleInfo]):
    method_statement_counts, class_method_counts, class_locs = [], [], []
    for mod in modules:
        if not mod.ok:
            continue
        for cls in mod.classes:
            cm = class_metrics(cls)
            class_method_counts.append(cm.method_count)
            class_locs.append(cm.loc)
            for fn in cls.methods:
                method_statement_counts.append(function_metrics(fn, mod).statement_count)
        for fn in mod.functions:
            method_statement_counts.append(function_metrics(fn, mod).statement_count)
    return method_statement_counts, class_method_counts, class_locs
