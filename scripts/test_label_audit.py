# -*- coding: utf-8 -*-
"""LABEL AUDIT ONLY -- no model inference, no metrics. Explains exactly
which TEST entities changed label between the stale existing TEST data
(data/processed/graphs/test) and the corrected TEST data
(data/processed/graphs_test_corrected_original), and why.

Identity problem: the stale .pt tensors store no qualname/lineno, only
feature values in array-index order, and the nested-function/class fix
INSERTS new nodes (never reorders or removes pre-existing ones) --
confirmed by re-reading ml/graph/graph_builder.py's module docstring
this session ("a nested def was previously silently dropped entirely").
So a pre-existing entity's array index can shift between stale and
corrected, but its (loc, statement_count, param_count) fingerprint
(columns 0-2 -- untouched by the collaborator-chain fix, which only
touches self_access_count/column 3, and untouched by Experiment #1,
which only touches column 4) does not change. This script aligns the
two per-file sequences by that fingerprint, preserving relative order
(a subsequence alignment, via difflib.SequenceMatcher), which correctly
separates "pre-existing entity, label recomputed" from "brand-new node,
never existed in the stale count at all" without needing to reconstruct
the historical parser.

Every number in the final report is produced by this script, not
invented.
"""
from __future__ import annotations

import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ml.preprocessing.ast_parser import parse_file
from ml.preprocessing.label_rules import LabelThresholds, is_feature_envy, is_god_class, is_long_method
from ml.preprocessing.metrics import class_metrics, function_metrics

STALE_STRUCT_DIR = ROOT / "data" / "processed" / "graphs" / "test"
CORRECTED_STRUCT_DIR = ROOT / "data" / "processed" / "graphs_test_corrected_original"
RAW_TEST_DIR = ROOT / "data" / "raw" / "test"

import json
cfg = json.loads((ROOT / "configs" / "label_thresholds.json").read_text())
TH = LabelThresholds(
    long_method_statements=cfg["long_method_statements"],
    god_class_method_count=cfg["god_class_method_count"],
    god_class_loc=cfg["god_class_loc"],
    feature_envy_min_external_calls=cfg["feature_envy_min_external_calls"],
    god_class_min_fields=cfg["god_class_min_fields"],
    god_class_min_methods_for_field_branch=cfg["god_class_min_methods_for_field_branch"],
)


def align(stale_fp, corrected_fp):
    """stale_fp, corrected_fp: lists of (loc,stmt,param) tuples, in each
    side's own original order. Returns list of (stale_idx, corrected_idx)
    matched pairs (order-preserving longest common subsequence), plus
    unmatched indices on each side."""
    sm = SequenceMatcher(None, stale_fp, corrected_fp, autojunk=False)
    matched = []
    matched_stale, matched_corr = set(), set()
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            si, ci = block.a + k, block.b + k
            matched.append((si, ci))
            matched_stale.add(si)
            matched_corr.add(ci)
    unmatched_stale = [i for i in range(len(stale_fp)) if i not in matched_stale]
    unmatched_corr = [i for i in range(len(corrected_fp)) if i not in matched_corr]
    return matched, unmatched_stale, unmatched_corr


def main():
    fe_transitions = Counter()  # (stale_label, corrected_label) -> count, for MATCHED entities
    fe_new_entity_labels = Counter()  # corrected label, for entities with NO stale counterpart
    fe_examples = {"pos_to_neg": [], "neg_to_pos": [], "new_positive": []}

    lm_transitions = Counter()
    lm_new_entity_labels = Counter()
    lm_examples = {"pos_to_neg": [], "neg_to_pos": [], "new_positive": []}

    gc_transitions = Counter()
    gc_new_entity_labels = Counter()
    gc_examples = {"pos_to_neg": [], "neg_to_pos": [], "new_positive": []}

    stale_files = sorted(STALE_STRUCT_DIR.rglob("*.pt"))
    n_files_checked = 0
    n_class_count_mismatch_files = 0
    vanished_lm = [0, 0]  # [positives among vanished, total vanished] -- entities in stale with NO corrected counterpart
    vanished_fe = [0, 0]
    vanished_gc = [0, 0]

    for sp in stale_files:
        rel = sp.relative_to(STALE_STRUCT_DIR)
        cp = CORRECTED_STRUCT_DIR / rel
        if not cp.exists():
            continue
        d_stale = torch.load(sp, weights_only=False)
        d_corr = torch.load(cp, weights_only=False)
        n_files_checked += 1

        repo = rel.parts[0]
        relpath = str(Path(*rel.parts[1:]))[: -len(".pt")]
        src_path = RAW_TEST_DIR / repo / relpath
        mod = parse_file(src_path)
        if not mod.ok:
            continue

        # --- method + function fingerprint alignment (FE + LM) ---
        for ntype in ("method", "function"):
            xs, xc = d_stale[ntype].x, d_corr[ntype].x
            if xs.shape[0] == 0 and xc.shape[0] == 0:
                continue
            stale_fp = [tuple(row[:3].tolist()) for row in xs]
            corr_fp = [tuple(row[:3].tolist()) for row in xc]
            matched, unmatched_stale, unmatched_corr = align(stale_fp, corr_fp)

            # current qualnames in CURRENT parse order (== corrected order)
            if ntype == "method":
                names = [fn.qualname for cls in mod.classes for fn in cls.methods]
            else:
                names = [fn.qualname for fn in mod.functions]

            for si, ci in matched:
                s_lm = int(d_stale[ntype].y_long_method[si].item())
                c_lm = int(d_corr[ntype].y_long_method[ci].item())
                lm_transitions[(s_lm, c_lm)] += 1
                if s_lm != c_lm and ntype == "method":  # representative examples from methods only, for brevity
                    key = "pos_to_neg" if s_lm == 1 else "neg_to_pos"
                    if len(lm_examples[key]) < 8:
                        lm_examples[key].append((repo, relpath, names[ci] if ci < len(names) else "?", int(xc[ci, 1].item())))

                if ntype == "method":
                    s_fe = int(d_stale[ntype].y_feature_envy[si].item())
                    c_fe = int(d_corr[ntype].y_feature_envy[ci].item())
                    fe_transitions[(s_fe, c_fe)] += 1
                    if s_fe != c_fe:
                        key = "pos_to_neg" if s_fe == 1 else "neg_to_pos"
                        if len(fe_examples[key]) < 8:
                            qn = names[ci] if ci < len(names) else "?"
                            fn_obj = next((f for c in mod.classes for f in c.methods if f.qualname == qn), None)
                            fm = function_metrics(fn_obj, mod) if fn_obj else None
                            fe_examples[key].append((
                                repo, relpath, qn,
                                f"self={fm.self_access_count if fm else '?'} dom_ext={fm.dominant_external_count if fm else '?'}",
                            ))

            for ci in unmatched_corr:
                c_lm = int(d_corr[ntype].y_long_method[ci].item())
                lm_new_entity_labels[c_lm] += 1
                if c_lm == 1 and len(lm_examples["new_positive"]) < 8:
                    lm_examples["new_positive"].append((repo, relpath, names[ci] if ci < len(names) else "?", int(xc[ci, 1].item())))
                if ntype == "method":
                    c_fe = int(d_corr[ntype].y_feature_envy[ci].item())
                    fe_new_entity_labels[c_fe] += 1
                    if c_fe == 1 and len(fe_examples["new_positive"]) < 8:
                        fe_examples["new_positive"].append((repo, relpath, names[ci] if ci < len(names) else "?", "new node (nested def)"))

            if unmatched_stale:
                vanished_lm[0] += sum(int(d_stale[ntype].y_long_method[si].item()) for si in unmatched_stale)
                vanished_lm[1] += len(unmatched_stale)
                if ntype == "method":
                    vanished_fe[0] += sum(int(d_stale[ntype].y_feature_envy[si].item()) for si in unmatched_stale)
                    vanished_fe[1] += len(unmatched_stale)

        # --- class fingerprint alignment (GC): fingerprint = (loc, method_count) since field_count could itself shift if method/attr fix changed field detection; loc+method_count is the stable pair ---
        xs, xc = d_stale["class"].x, d_corr["class"].x
        if xs.shape[0] > 0 or xc.shape[0] > 0:
            stale_fp = [tuple(row[:2].tolist()) for row in xs]  # loc, method_count
            corr_fp = [tuple(row[:2].tolist()) for row in xc]
            matched, unmatched_stale, unmatched_corr = align(stale_fp, corr_fp)
            class_names = [c.name for c in mod.classes]
            if len(class_names) != xc.shape[0]:
                n_class_count_mismatch_files += 1

            for si, ci in matched:
                s_gc = int(d_stale["class"].y[si].item())
                c_gc = int(d_corr["class"].y[ci].item())
                gc_transitions[(s_gc, c_gc)] += 1
                if s_gc != c_gc:
                    key = "pos_to_neg" if s_gc == 1 else "neg_to_pos"
                    if len(gc_examples[key]) < 8:
                        cn = class_names[ci] if ci < len(class_names) else "?"
                        cls_obj = next((c for c in mod.classes if c.name == cn), None)
                        cm = class_metrics(cls_obj) if cls_obj else None
                        gc_examples[key].append((repo, relpath, cn, f"loc={cm.loc if cm else '?'} methods={cm.method_count if cm else '?'} fields={cm.field_count if cm else '?'}"))
            for ci in unmatched_corr:
                c_gc = int(d_corr["class"].y[ci].item())
                gc_new_entity_labels[c_gc] += 1
                if c_gc == 1 and len(gc_examples["new_positive"]) < 8:
                    cn = class_names[ci] if ci < len(class_names) else "?"
                    gc_examples["new_positive"].append((repo, relpath, cn, "new/shifted class entry"))
            if unmatched_stale:
                vanished_gc[0] += sum(int(d_stale["class"].y[si].item()) for si in unmatched_stale)
                vanished_gc[1] += len(unmatched_stale)

    print(f"files checked: {n_files_checked}")
    print(f"\n=== FEATURE ENVY ===")
    print("matched-entity transitions (stale_label -> corrected_label): counts:")
    for k, v in sorted(fe_transitions.items()):
        print(f"  {k}: {v}")
    print("new-entity (no stale counterpart) label counts:", dict(fe_new_entity_labels))
    for key, exs in fe_examples.items():
        print(f" examples[{key}]:")
        for e in exs:
            print("   ", e)

    print(f"\n=== LONG METHOD ===")
    for k, v in sorted(lm_transitions.items()):
        print(f"  {k}: {v}")
    print("new-entity label counts:", dict(lm_new_entity_labels))
    for key, exs in lm_examples.items():
        print(f" examples[{key}]:")
        for e in exs:
            print("   ", e)

    print(f"\n=== GOD CLASS ===")
    for k, v in sorted(gc_transitions.items()):
        print(f"  {k}: {v}")
    print("new-entity label counts:", dict(gc_new_entity_labels))
    print(f"files with class-count mismatch (stale vs corrected #classes): {n_class_count_mismatch_files}")
    for key, exs in gc_examples.items():
        print(f" examples[{key}]:")
        for e in exs:
            print("   ", e)

    # sanity totals
    fe_stale_total = sum(v for (s, c), v in fe_transitions.items() if s == 1)
    fe_corr_matched_total = sum(v for (s, c), v in fe_transitions.items() if c == 1)
    fe_corr_new_total = fe_new_entity_labels.get(1, 0)
    print(f"\n[vanished] entities present in STALE with no corrected counterpart at all "
          f"(would need explanation if non-zero): LM total={vanished_lm[1]} positives={vanished_lm[0]}, "
          f"FE total={vanished_fe[1]} positives={vanished_fe[0]}, GC total={vanished_gc[1]} positives={vanished_gc[0]}")
    print(f"\n[sanity] FE stale positives among matched pairs: {fe_stale_total}")
    print(f"[sanity] FE corrected positives among matched pairs: {fe_corr_matched_total}")
    print(f"[sanity] FE corrected positives among NEW (unmatched) entities: {fe_corr_new_total}")
    print(f"[sanity] FE corrected total (matched + new): {fe_corr_matched_total + fe_corr_new_total}  (expect 554)")


if __name__ == "__main__":
    main()
