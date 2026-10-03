# -*- coding: utf-8 -*-
"""Phase 2/3 dataset builder.

Walks the cloned repos (data/raw/<split>/<repo>), parses every .py file with
ml.preprocessing.ast_parser, extracts method- and class-level units, removes
exact/near duplicates via structural hash (repo-level split was already fixed
in configs/repos.yaml BEFORE this script ever runs, so this step only has to
guard against duplicate code that happens to appear in more than one repo —
e.g. a vendored copy of the same utility — which would otherwise leak across
splits), derives corpus-relative label thresholds from the TRAIN split only,
and applies those fixed thresholds to all three splits.

Output: data/processed/{split}_methods.jsonl, data/processed/{split}_classes.jsonl
         docs/dataset_report.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocessing.ast_parser import parse_file
from ml.preprocessing.label_rules import (
    LabelThresholds,
    collect_corpus_stats,
    is_feature_envy,
    is_god_class,
    is_long_method,
)
from ml.preprocessing.metrics import function_metrics

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"

EXCLUDE_DIR_NAMES = {
    "tests", "test", "testing", "docs", "doc", "examples", "example",
    "migrations", "vendor", "vendored", "node_modules", ".git",
    "build", "dist", "__pycache__", ".tox", ".eggs",
}


def iter_py_files(repo_dir: Path):
    """Yield .py files under repo_dir, skipping EXCLUDE_DIR_NAMES directories
    *within the repo*. Checks path parts relative to repo_dir only — checking
    the full absolute path would also match ancestor directories like the
    split folder itself (data/raw/test/...), wrongly excluding everything
    under the "test" split."""
    for path in repo_dir.rglob("*.py"):
        rel_parts = path.relative_to(repo_dir).parts
        if any(part in EXCLUDE_DIR_NAMES for part in rel_parts[:-1]):
            continue
        yield path


def parse_split(split: str):
    split_dir = RAW_DIR / split
    modules = []  # list of (repo, ModuleInfo)
    parse_errors = []
    file_count = 0
    for repo_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        repo = repo_dir.name
        for path in iter_py_files(repo_dir):
            file_count += 1
            mod = parse_file(path)
            if not mod.ok:
                parse_errors.append((repo, str(path.relative_to(repo_dir)), mod.parse_error))
                continue
            modules.append((repo, str(path.relative_to(repo_dir)), mod))
    return modules, parse_errors, file_count


def extract_units(split: str, modules):
    method_units = []
    class_units = []
    for repo, relpath, mod in modules:
        full_path = RAW_DIR / split / repo / relpath
        try:
            text_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            text_lines = []

        def snippet(lineno, end_lineno):
            if not text_lines:
                return ""
            return "\n".join(text_lines[lineno - 1:end_lineno])

        for cls in mod.classes:
            class_units.append({
                "split": split, "repo": repo, "file": relpath,
                "qualname": cls.name, "lineno": cls.lineno, "end_lineno": cls.end_lineno,
                "loc": cls.loc, "method_count": len(cls.methods), "field_count": len(cls.class_attrs),
                "struct_hash": cls.struct_hash, "source": snippet(cls.lineno, cls.end_lineno),
                "_cls": cls,
            })
            for fn in cls.methods:
                fm = function_metrics(fn, mod)
                method_units.append({
                    "split": split, "repo": repo, "file": relpath,
                    "qualname": fn.qualname, "lineno": fn.lineno, "end_lineno": fn.end_lineno,
                    "loc": fn.loc, "statement_count": fm.statement_count, "is_method": True,
                    "self_access_count": fm.self_access_count,
                    "external_access_count": fm.external_access_count,
                    "dominant_external_receiver": fm.dominant_external_receiver,
                    "dominant_external_count": fm.dominant_external_count,
                    "struct_hash": fn.struct_hash, "source": snippet(fn.lineno, fn.end_lineno),
                    "_fn": fn, "_mod": mod,
                })
        for fn in mod.functions:
            fm = function_metrics(fn, mod)
            method_units.append({
                "split": split, "repo": repo, "file": relpath,
                "qualname": fn.qualname, "lineno": fn.lineno, "end_lineno": fn.end_lineno,
                "loc": fn.loc, "statement_count": fm.statement_count, "is_method": False,
                "self_access_count": fm.self_access_count,
                "external_access_count": fm.external_access_count,
                "dominant_external_receiver": fm.dominant_external_receiver,
                "dominant_external_count": fm.dominant_external_count,
                "struct_hash": fn.struct_hash, "source": snippet(fn.lineno, fn.end_lineno),
                "_fn": fn, "_mod": mod,
            })
    return method_units, class_units


def dedup_cross_split(units: list) -> tuple:
    """Drop units whose struct_hash appears in more than one split (possible
    leakage). Within a single split, keep only the first occurrence of a
    repeated struct_hash (redundant boilerplate)."""
    hash_to_splits = defaultdict(set)
    for u in units:
        hash_to_splits[u["struct_hash"]].add(u["split"])

    cross_split_hashes = {h for h, splits in hash_to_splits.items() if len(splits) > 1}

    kept = []
    seen_within_split = set()
    dropped_cross = 0
    dropped_within = 0
    for u in units:
        h = u["struct_hash"]
        if h in cross_split_hashes:
            dropped_cross += 1
            continue
        key = (u["split"], h)
        if key in seen_within_split:
            dropped_within += 1
            continue
        seen_within_split.add(key)
        kept.append(u)
    return kept, dropped_cross, dropped_within


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    all_method_units = []
    all_class_units = []
    all_modules_by_split = {}
    report_lines = ["# RefactorGraph Dataset Report\n"]
    parse_error_summary = {}
    file_counts = {}

    for split in ("train", "val", "test"):
        print(f"[parse] {split} ...")
        modules, parse_errors, file_count = parse_split(split)
        file_counts[split] = file_count
        parse_error_summary[split] = parse_errors
        all_modules_by_split[split] = modules
        method_units, class_units = extract_units(split, modules)
        all_method_units.extend(method_units)
        all_class_units.extend(class_units)
        print(f"[parse] {split}: {file_count} files, {len(modules)} parsed ok, "
              f"{len(parse_errors)} parse errors, {len(method_units)} methods, {len(class_units)} classes")

    method_units, m_cross, m_within = dedup_cross_split(all_method_units)
    class_units, c_cross, c_within = dedup_cross_split(all_class_units)
    print(f"[dedup] methods: dropped {m_cross} cross-split, {m_within} within-split duplicates")
    print(f"[dedup] classes: dropped {c_cross} cross-split, {c_within} within-split duplicates")

    # Thresholds must come from ALL parsed train modules directly (not
    # reconstructed from the flattened method/class records), otherwise a
    # module with zero methods/functions silently drops out of the corpus
    # used to compute percentiles — this caused build_graphs.py, which does
    # use the full module list, to compute slightly different thresholds
    # from "the same" train split, producing inconsistent labels between the
    # flat dataset and the graph dataset for methods near the cutoff.
    train_mods = [mod for _repo, _relpath, mod in all_modules_by_split["train"]]
    method_statement_counts, class_method_counts, class_locs = collect_corpus_stats(train_mods)
    thresholds = LabelThresholds.from_corpus(method_statement_counts, class_method_counts, class_locs)
    print(f"[thresholds] {thresholds}")

    thresholds_path = ROOT / "configs" / "label_thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "long_method_statements": thresholds.long_method_statements,
                "god_class_method_count": thresholds.god_class_method_count,
                "god_class_loc": thresholds.god_class_loc,
                "feature_envy_min_external_calls": thresholds.feature_envy_min_external_calls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[thresholds] saved to {thresholds_path} — build_graphs.py must load this, not recompute")

    for u in method_units:
        fn, mod = u["_fn"], u["_mod"]
        u["long_method"] = is_long_method(fn, mod, thresholds)
        u["feature_envy"] = is_feature_envy(fn, mod, thresholds)
        del u["_fn"]
        del u["_mod"]

    for u in class_units:
        u["god_class"] = is_god_class(u["_cls"], thresholds)
        del u["_cls"]

    by_split_methods = defaultdict(list)
    by_split_classes = defaultdict(list)
    for u in method_units:
        by_split_methods[u["split"]].append(u)
    for u in class_units:
        by_split_classes[u["split"]].append(u)

    label_counts = {}
    for split in ("train", "val", "test"):
        ms = by_split_methods[split]
        cs = by_split_classes[split]
        lm = sum(1 for u in ms if u["long_method"])
        fe = sum(1 for u in ms if u["feature_envy"])
        gc = sum(1 for u in cs if u["god_class"])
        label_counts[split] = {
            "methods_total": len(ms), "long_method": lm, "feature_envy": fe,
            "classes_total": len(cs), "god_class": gc,
        }
        with open(PROCESSED_DIR / f"{split}_methods.jsonl", "w", encoding="utf-8") as f:
            for u in ms:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        with open(PROCESSED_DIR / f"{split}_classes.jsonl", "w", encoding="utf-8") as f:
            for u in cs:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")

    report_lines.append("## Files parsed\n")
    for split in ("train", "val", "test"):
        report_lines.append(f"- {split}: {file_counts[split]} .py files, "
                             f"{len(parse_error_summary[split])} parse errors")
    report_lines.append("\n## Deduplication\n")
    report_lines.append(f"- Methods: {m_cross} dropped (cross-split collision), {m_within} dropped (within-split duplicate)")
    report_lines.append(f"- Classes: {c_cross} dropped (cross-split collision), {c_within} dropped (within-split duplicate)")
    report_lines.append("\n## Label thresholds (derived from TRAIN split only, 90th percentile, floors applied)\n")
    report_lines.append(f"- Long Method: statement_count >= {thresholds.long_method_statements}")
    report_lines.append(f"- God/Large Class: method_count >= {thresholds.god_class_method_count} AND LOC >= {thresholds.god_class_loc}")
    report_lines.append(f"- Feature Envy: dominant external receiver access count >= {thresholds.feature_envy_min_external_calls} AND > self access count")
    report_lines.append("\n## Label counts per split\n")
    for split in ("train", "val", "test"):
        lc = label_counts[split]
        report_lines.append(
            f"- {split}: {lc['methods_total']} methods "
            f"(long_method={lc['long_method']}, feature_envy={lc['feature_envy']}); "
            f"{lc['classes_total']} classes (god_class={lc['god_class']})"
        )
    report_lines.append("\n## Known limitations\n")
    report_lines.append("- Labels are RULE-BASED (silver), derived from our own AST metrics, not human-annotated. "
                         "A stratified manual-review subset must be scored for precision before trusting these as gold.")
    report_lines.append("- Feature Envy heuristic excludes imported-module and builtin receivers but has no real type "
                         "inference; a parameter reassigned to a different type mid-method, or attribute access through "
                         "a deep alias, can still be mis-attributed.")
    report_lines.append("- Thresholds are percentile-based on this specific 9-repo train pool; not claimed universal.")

    (DOCS_DIR / "dataset_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nWrote processed dataset to {PROCESSED_DIR} and report to {DOCS_DIR / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
