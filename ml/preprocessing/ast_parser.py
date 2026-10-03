# -*- coding: utf-8 -*-
"""Parse a single Python source file into structural facts (imports, classes,
functions/methods, calls, attribute accesses) using the stdlib ast module.

This is the single source of truth both for graph construction (ml/graph) and
for rule-based smell-label derivation (ml/preprocessing/label_rules.py) — we
deliberately do not depend on a third-party smell-detection tool for labels,
since PyExamine's Feature Envy rule was found (empirically, see
docs/dataset_report.md) to misfire on calls to imported modules (e.g.
`click.echo(...)` flagged as envy toward a class named `click`). Deriving
labels from the same typed-edge data used for the graph lets us exclude that
failure mode explicitly (see label_rules.py).
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple


def _count_statements(node: ast.AST) -> int:
    """Count executable statements inside node's body, recursing into nested
    blocks (if/for/while/try/with) but NOT into nested function/class defs
    (those are separate units, counted independently when visited on their
    own). Unlike raw line count, this is immune to docstring/comment length —
    a 30-line docstring is one ast.Expr statement, not 30 lines of "logic" —
    which raw LOC was found (via manual label review) to conflate, producing
    false-positive Long Method labels on heavily-documented short functions.
    """
    count = 0
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            count += 1
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            count += _count_statements(child)
    return count


def _struct_hash(node: ast.AST) -> str:
    """Position-independent structural hash of an AST subtree, used for
    near-duplicate detection (ast.dump without line/col attributes ignores
    formatting/whitespace/comments, so two textually-different-but-
    structurally-identical functions hash the same)."""
    dumped = ast.dump(node, annotate_fields=False, include_attributes=False)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()


@dataclass
class ImportInfo:
    module: str
    names: List[str]
    lineno: int


@dataclass
class CallInfo:
    callee: str
    receiver: Optional[str]
    lineno: int


@dataclass
class AttrAccessInfo:
    receiver: str
    attr: str
    lineno: int
    # True when this Attribute IS the callee of a Call (`x.method()`), i.e.
    # the same syntactic interaction is ALSO recorded once in `calls`. Graph
    # edges (accesses/uses) should still be built from every AttrAccessInfo
    # regardless of this flag -- "the method touches this parameter/attribute"
    # is true either way. metrics.py uses this flag to avoid counting the
    # same call-based interaction twice (once via calls, once via
    # attr_accesses) when computing external_access_count/self_access_count
    # -- see docs/robustness_audit_phase_a.md.
    is_call: bool = False
    # True when this Attribute is merely the base of a LONGER chain, e.g.
    # `self.database_connection` inside `self.database_connection.execute()`
    # -- it is not a meaningful standalone access in its own right, it is a
    # stepping stone to reach the real collaborator interaction (already
    # recorded separately in `calls`, with receiver "self.database_connection").
    # Without this flag, every call through an owned collaborator reference
    # (the single most common real-world Feature Envy shape: a service class
    # delegating to `self.<collaborator>`) got counted as BOTH a self-access
    # AND an external access for the same syntactic interaction, inflating
    # self_access_count by exactly the external count it should have been
    # compared against -- making `dominant_external_count > self_access_count`
    # nearly impossible to satisfy for exactly this pattern. metrics.py
    # excludes these from both self and external counts entirely.
    is_chained_base: bool = False


@dataclass
class FunctionInfo:
    name: str
    qualname: str
    lineno: int
    end_lineno: int
    params: List[str]
    decorators: List[str]
    is_method: bool
    class_name: Optional[str]
    struct_hash: str = ""
    statement_count: int = 0
    calls: List[CallInfo] = field(default_factory=list)
    attr_accesses: List[AttrAccessInfo] = field(default_factory=list)
    nested_functions: List["FunctionInfo"] = field(default_factory=list)
    local_names: Set[str] = field(default_factory=set)
    # Set only for a function/closure nested directly inside another
    # function or method (its immediate enclosing def's own qualname) --
    # None for a true top-level function AND for a direct class method
    # (those are already fully described by class_name/is_method). Used by
    # graph_builder.py to wire a (method|function, contains, function) edge
    # to the closure's real parent instead of silently dropping it (see
    # docs/robustness_audit_phase_a.md).
    parent_qualname: Optional[str] = None

    @property
    def loc(self) -> int:
        return max(self.end_lineno - self.lineno + 1, 0)


@dataclass
class ClassInfo:
    name: str
    lineno: int
    end_lineno: int
    bases: List[str]
    struct_hash: str = ""
    methods: List[FunctionInfo] = field(default_factory=list)
    class_attrs: Set[str] = field(default_factory=set)
    # Unique identifier for this class: "Outer.Inner" for a class nested
    # inside another class/function, else just its own name. Always set;
    # used to key graph_builder.py's class index so two unrelated nested
    # classes with the same simple name (e.g. two different `class Meta:`
    # inner classes) never collide.
    qualname: str = ""
    # The enclosing class/function's own qualname when this class is
    # nested inside one; None for a true top-level class. See parent_qualname
    # on FunctionInfo for the same idea applied to functions.
    parent_qualname: Optional[str] = None

    @property
    def loc(self) -> int:
        return max(self.end_lineno - self.lineno + 1, 0)


@dataclass
class ModuleInfo:
    path: str
    imports: List[ImportInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    parse_error: Optional[str] = None
    loc: int = 0

    @property
    def ok(self) -> bool:
        return self.parse_error is None

    def imported_names(self) -> Set[str]:
        """All local identifiers bound by import statements in this module,
        e.g. `import click` -> {'click'}, `from x import y as z` -> {'z'}.
        Used to exclude module-namespace calls from Feature Envy detection.
        """
        names: Set[str] = set()
        for imp in self.imports:
            names.update(imp.names)
        return names


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: List[ImportInfo] = []
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []
        self._class_stack: List[ClassInfo] = []
        self._func_stack: List[FunctionInfo] = []
        # Tracks the innermost enclosing def/class regardless of kind, so
        # "is this a method, and who is its real structural parent" can be
        # answered correctly even when a class is nested inside a function
        # (or vice versa) -- _class_stack/_func_stack alone can't answer
        # "which of the two is innermost" and previously caused a method of
        # a class defined inside a function to be misclassified as a bare
        # function (is_method required _func_stack to be empty, not just
        # "no function between here and the nearest class").
        self._scope_stack: List[Tuple[str, object]] = []
        # (call.func Attribute node) ids already counted via visit_Call, so
        # visit_Attribute doesn't double-count `x.method()` as BOTH a call
        # AND a separate attribute access (see docs/robustness_audit_phase_a.md
        # -- this previously inflated external_access_count/
        # dominant_external_count 2x for every call-based interaction).
        self._call_callee_attr_ids: Set[int] = set()
        # (Attribute node) ids that are the `.value` of an OUTER Attribute,
        # e.g. `self.database_connection` inside `self.database_connection.execute`
        # -- marked here (when the outer node is visited, before recursion
        # reaches the inner one) so visit_Attribute can flag the inner access
        # as a non-standalone stepping stone. See AttrAccessInfo.is_chained_base.
        self._chained_base_ids: Set[int] = set()

    def _current_scope_qualname(self) -> Optional[str]:
        if not self._scope_stack:
            return None
        return self._scope_stack[-1][1].qualname

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.imports.append(ImportInfo(module=alias.name, names=[local], lineno=node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            self.imports.append(ImportInfo(module=mod, names=[local], lineno=node.lineno))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [self._name_of(b) for b in node.bases]
        parent_qualname = self._current_scope_qualname()
        qualname = f"{parent_qualname}.{node.name}" if parent_qualname else node.name
        cls = ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            bases=bases,
            struct_hash=_struct_hash(node),
            qualname=qualname,
            parent_qualname=parent_qualname,
        )
        self._class_stack.append(cls)
        self._scope_stack.append(("class", cls))
        self.generic_visit(node)
        self._scope_stack.pop()
        self._class_stack.pop()
        self.classes.append(cls)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(self, node) -> None:
        # Innermost enclosing def/class decides is_method -- NOT merely
        # "any function anywhere on the stack" (the old rule), which
        # misclassified a real instance method of a class defined inside a
        # function (e.g. a class-returning factory) as a bare function.
        innermost = self._scope_stack[-1] if self._scope_stack else None
        is_method = innermost is not None and innermost[0] == "class"
        class_name = innermost[1].name if is_method else None
        # A function/closure nested inside another function/method (not a
        # direct class method) records its real parent so it can become a
        # genuine graph node instead of being silently dropped.
        parent_qualname = innermost[1].qualname if (innermost is not None and innermost[0] == "function") else None

        if is_method:
            qualname = f"{class_name}.{node.name}"
        elif parent_qualname:
            qualname = f"{parent_qualname}.<locals>.{node.name}"
        else:
            qualname = node.name

        fn = FunctionInfo(
            name=node.name,
            qualname=qualname,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            params=[a.arg for a in node.args.args],
            decorators=[self._name_of(d) for d in node.decorator_list],
            is_method=is_method,
            class_name=class_name,
            struct_hash=_struct_hash(node),
            statement_count=_count_statements(node),
            parent_qualname=parent_qualname,
        )
        self._func_stack.append(fn)
        self._scope_stack.append(("function", fn))
        self.generic_visit(node)
        self._scope_stack.pop()
        self._func_stack.pop()

        if self._func_stack:
            self._func_stack[-1].nested_functions.append(fn)

        if is_method:
            # class_stack's top is always the innermost class regardless of
            # any interleaved function frames -- same class `innermost`
            # already points at when is_method is True.
            self._class_stack[-1].methods.append(fn)
        else:
            # Both a true top-level function (parent_qualname is None) and
            # a nested closure (parent_qualname set) land here -- the ONLY
            # thing that changes is whether parent_qualname is set, which
            # graph_builder.py uses to wire the correct `contains` edge.
            self.functions.append(fn)

    def visit_Call(self, node: ast.Call) -> None:
        if self._func_stack:
            receiver, callee = self._call_target(node.func)
            self._func_stack[-1].calls.append(CallInfo(callee=callee, receiver=receiver, lineno=node.lineno))
            if isinstance(node.func, ast.Attribute):
                # `x.method()` is a Call wrapping an Attribute -- generic_visit
                # below will visit that same Attribute node again via normal
                # recursion, recording an AttrAccessInfo for it too (needed
                # so graph_builder.py still builds an accesses/uses edge to
                # the receiver). Mark it as is_call so metrics.py can avoid
                # counting this one interaction twice (see AttrAccessInfo
                # docstring above) without losing the edge-building data.
                self._call_callee_attr_ids.add(id(node.func))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Attribute):
            # node.value (e.g. `self.database_connection` inside
            # `self.database_connection.execute`) is a stepping stone to
            # reach node's own attribute -- mark it before recursion visits
            # it, so it doesn't get counted as a standalone access.
            self._chained_base_ids.add(id(node.value))
        if self._func_stack and isinstance(node.value, ast.Name):
            is_call = id(node) in self._call_callee_attr_ids
            self._call_callee_attr_ids.discard(id(node))
            is_chained_base = id(node) in self._chained_base_ids
            self._chained_base_ids.discard(id(node))
            self._func_stack[-1].attr_accesses.append(
                AttrAccessInfo(
                    receiver=node.value.id, attr=node.attr, lineno=node.lineno,
                    is_call=is_call, is_chained_base=is_chained_base,
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._func_stack and self._class_stack:
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    self._class_stack[-1].class_attrs.add(target.attr)
        if self._func_stack:
            for target in node.targets:
                self._collect_local_names(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._func_stack:
            self._collect_local_names(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._func_stack:
            self._collect_local_names(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if self._func_stack:
            self._collect_local_names(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if self._func_stack:
            self._collect_local_names(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        if self._func_stack:
            self._collect_local_names(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        if self._func_stack:
            for item in node.items:
                if item.optional_vars is not None:
                    self._collect_local_names(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        if self._func_stack:
            for item in node.items:
                if item.optional_vars is not None:
                    self._collect_local_names(item.optional_vars)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if self._func_stack:
            self._collect_local_names(node.target)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if self._func_stack and node.name:
            self._func_stack[-1].local_names.add(node.name)
        self.generic_visit(node)

    def _collect_local_names(self, target: ast.AST) -> None:
        """Record names bound by an assignment/for/with target as local to
        the current function, so they can be excluded from Feature Envy's
        "external object" receiver count (a locally-created list/dict/loop
        variable is not a collaborator object, it is the method's own
        scratch state)."""
        if isinstance(target, ast.Name):
            if target.id != "self":
                self._func_stack[-1].local_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_local_names(elt)
        elif isinstance(target, ast.Starred):
            self._collect_local_names(target.value)
        # ast.Attribute / ast.Subscript targets (e.g. self.x, d[k]) bind no new name.

    @staticmethod
    def _name_of(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_Visitor._name_of(node.value)}.{node.attr}"
        if isinstance(node, ast.Call):
            return _Visitor._name_of(node.func)
        return type(node).__name__

    @classmethod
    def _call_target(cls, func_node):
        if isinstance(func_node, ast.Name):
            return None, func_node.id
        if isinstance(func_node, ast.Attribute):
            receiver = cls._name_of(func_node.value)
            return receiver, f"{receiver}.{func_node.attr}"
        return None, cls._name_of(func_node)


def parse_source(source: str, path: str = "<string>") -> ModuleInfo:
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as e:
        return ModuleInfo(path=path, parse_error=f"{type(e).__name__}: {e}")

    visitor = _Visitor()
    visitor.visit(tree)
    loc = source.count("\n") + 1 if source else 0
    return ModuleInfo(
        path=path,
        imports=visitor.imports,
        classes=visitor.classes,
        functions=visitor.functions,
        loc=loc,
    )


def parse_file(path: Path) -> ModuleInfo:
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return ModuleInfo(path=str(path), parse_error=f"{type(e).__name__}: {e}")
    return parse_source(source, path=str(path))
