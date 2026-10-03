# -*- coding: utf-8 -*-
"""Pure structural metrics computed from a parsed ModuleInfo. No thresholds
here — thresholding (what counts as "smelly") lives in label_rules.py so the
two concerns (measurement vs. decision) stay separable and the decision can
be justified against corpus statistics instead of a hardcoded magic number.
"""
from __future__ import annotations

import builtins
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from ml.preprocessing.ast_parser import ClassInfo, FunctionInfo, ModuleInfo

_BUILTIN_NAMES = set(dir(builtins))

# Generic dict/list/str/set method names. Calling .get()/.setdefault()/etc on
# ANY receiver (a **kwargs dict, a plain response dict, a parameter) is weak
# evidence of Feature Envy: it reflects normal data-structure manipulation,
# not a method that behaviorally belongs to another class. Manual review
# found this was the largest remaining source of false positives after
# excluding local/loop variables (e.g. `kwargs.setdefault(...)`,
# `response.get(...)`, `kw.setdefault(...)`) — real Feature Envy is about
# domain-specific method calls on a collaborator object, not generic
# container access.
_GENERIC_CONTAINER_METHODS = {
    "get", "setdefault", "update", "pop", "popitem", "keys", "values", "items",
    "append", "extend", "insert", "remove", "discard", "add", "sort", "reverse",
    "copy", "clear", "join", "split", "rsplit", "splitlines", "format", "format_map",
    "decode", "encode", "strip", "lstrip", "rstrip", "replace", "lower", "upper",
    "startswith", "endswith", "count", "index", "find", "rfind",
}


@dataclass
class FunctionMetrics:
    loc: int
    statement_count: int
    param_count: int
    self_access_count: int
    external_access_count: int
    dominant_external_receiver: Optional[str]
    dominant_external_count: int


@dataclass
class ClassMetrics:
    loc: int
    method_count: int
    field_count: int


def function_metrics(fn: FunctionInfo, module: ModuleInfo) -> FunctionMetrics:
    """Access-ratio metric for Feature Envy, excluding calls/attribute
    accesses on imported module names, builtins, and names local to the
    function itself (locally-created variables and loop variables — these
    are the method's own scratch state, not a collaborator object, and
    counting them as "external" produced most of the false positives found
    in manual review, e.g. `result = dict(...); result.update(...)` or
    `for rule in self.rules: ... rule.name`). This is in addition to the
    module-namespace exclusion that fixed the PyExamine-style false positive
    on calls like `click.echo(...)`.
    """
    imported = module.imported_names()
    excluded = imported | _BUILTIN_NAMES | fn.local_names | {"cls"}
    if fn.class_name:
        # A call like `Subprocess.initialize()` inside a method of class
        # Subprocess is the class referring to its own class-level state via
        # its literal name (instead of self/cls) — not envy of another class.
        excluded = excluded | {fn.class_name}

    self_count = 0
    external_receivers: Counter = Counter()

    for a in fn.attr_accesses:
        if a.attr in _GENERIC_CONTAINER_METHODS:
            # visit_Attribute fires for `x.get` in addition to visit_Call
            # firing for `x.get(...)` — without this, the call-side filter
            # above was bypassed via the attribute-access side, double
            # counting the same source location.
            continue
        if a.is_call:
            # This exact interaction (`x.method()`) is already counted once
            # below via fn.calls -- attr_accesses also carries an entry for
            # it (needed so graph_builder.py can still build an accesses/
            # uses edge to the receiver), but counting it here too would
            # double the metric for every call-based interaction while a
            # pure attribute read (`x.attr`, no call) only counts once --
            # see docs/robustness_audit_phase_a.md.
            continue
        if a.is_chained_base:
            # `self.database_connection` inside `self.database_connection.execute(...)`
            # -- a stepping stone to the real interaction, which fn.calls
            # already records once (receiver="self.database_connection").
            # Counting this too inflated self_access_count by exactly the
            # external count it should be compared against, for the single
            # most common real-world Feature Envy shape (delegating through
            # an owned collaborator reference) -- see AttrAccessInfo docstring.
            continue
        if a.receiver == "self":
            self_count += 1
        elif a.receiver in excluded:
            continue
        else:
            external_receivers[a.receiver] += 1

    for c in fn.calls:
        if c.receiver is None:
            continue
        method_name = c.callee.rsplit(".", 1)[-1]
        if method_name in _GENERIC_CONTAINER_METHODS:
            continue
        if c.receiver == "self":
            self_count += 1
        elif c.receiver in excluded:
            continue
        else:
            external_receivers[c.receiver] += 1

    if external_receivers:
        dominant_receiver, dominant_count = external_receivers.most_common(1)[0]
    else:
        dominant_receiver, dominant_count = None, 0

    return FunctionMetrics(
        loc=fn.loc,
        statement_count=fn.statement_count,
        param_count=len(fn.params),
        self_access_count=self_count,
        external_access_count=sum(external_receivers.values()),
        dominant_external_receiver=dominant_receiver,
        dominant_external_count=dominant_count,
    )


def class_metrics(cls: ClassInfo) -> ClassMetrics:
    return ClassMetrics(
        loc=cls.loc,
        method_count=len(cls.methods),
        field_count=len(cls.class_attrs),
    )
