# -*- coding: utf-8 -*-
"""ONE-TIME TEST hybrid patch (authorized: final TEST evaluation).

Same "verify byte-identical outside column 4, then patch" discipline as
scripts/experiment_fe_dominant_patch_hybrid.py (already proven correct
for TRAIN/VAL: 1061/1061 clean) and already structurally pre-verified
for TEST (scripts/rebuild_test_corrected_structural.py: identical
everywhere except at most column 4, 1574/1574, 0 unexpected diffs).

Base: data/processed/graphs_hybrid_test_corrected_original/ (current
parser + external_access_count, CodeBERT-augmented this session).
Column-4 source: data/processed/graphs_experiment_fe_dominant/test/
(current parser + dominant_external_count, structural only -- already
built by scripts/experiment_fe_dominant_test_eval.py step 1).
Output: data/processed/graphs_hybrid_test_corrected_dominant/ (NEW;
neither input directory is written to).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STRUCT_DIR = ROOT / "data" / "processed" / "graphs_experiment_fe_dominant" / "test"
BASE_HYBRID_DIR = ROOT / "data" / "processed" / "graphs_hybrid_test_corrected_original"
OUT_HYBRID_DIR = ROOT / "data" / "processed" / "graphs_hybrid_test_corrected_dominant"

CHECK_COLS = 4  # verify columns [0:4) match; patch column 4


def main():
    patched, mismatched, missing = 0, [], []
    struct_files = sorted(STRUCT_DIR.rglob("*.pt"))
    for sp in struct_files:
        rel = sp.relative_to(STRUCT_DIR)
        hp = BASE_HYBRID_DIR / rel
        if not hp.exists():
            missing.append(str(rel))
            continue
        d_struct = torch.load(sp, weights_only=False)
        d_hybrid = torch.load(hp, weights_only=False)

        ok = True
        for ntype in ("method", "function"):
            xs = d_struct[ntype].x
            xh = d_hybrid[ntype].x
            if xs.shape[0] != xh.shape[0]:
                ok = False
                break
            if xs.shape[0] == 0:
                continue
            if not torch.equal(xs[:, :CHECK_COLS], xh[:, :CHECK_COLS]):
                ok = False
                break
            if not torch.equal(d_struct[ntype].y_long_method, d_hybrid[ntype].y_long_method):
                ok = False
                break
            if not torch.equal(d_struct[ntype].y_feature_envy, d_hybrid[ntype].y_feature_envy):
                ok = False
                break
        if ok and d_struct["class"].x.shape[0] > 0:
            if not torch.equal(d_struct["class"].y, d_hybrid["class"].y):
                ok = False

        if not ok:
            mismatched.append(str(rel))
            continue

        for ntype in ("method", "function"):
            if d_hybrid[ntype].x.shape[0] == 0:
                continue
            new_x = d_hybrid[ntype].x.clone()
            new_x[:, 4] = d_struct[ntype].x[:, 4]
            d_hybrid[ntype].x = new_x

        out_path = OUT_HYBRID_DIR / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d_hybrid, out_path)
        patched += 1

    print(f"Patched (new TEST hybrid file written): {patched}")
    print(f"Missing base hybrid counterpart (skipped): {len(missing)}")
    for m in missing[:10]:
        print(f"  {m}")
    print(f"Mismatched (columns 0-3 or labels differ -- NOT patched): {len(mismatched)}")
    for m in mismatched[:10]:
        print(f"  {m}")

    if mismatched or missing:
        print("\n[WARNING] Not all files patched cleanly -- see above before trusting downstream evaluation.")
        raise SystemExit(1)
    else:
        print("\n[OK] Every TEST file verified byte-identical outside column 4 (method/function) before patching.")


if __name__ == "__main__":
    main()
