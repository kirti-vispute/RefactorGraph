# -*- coding: utf-8 -*-
"""Patch data/processed/graphs_hybrid/{train,val} label (y) tensors in place
from the freshly-rebuilt data/processed/graphs/{train,val} structural
graphs, WITHOUT re-running CodeBERT embedding.

Why this is safe: the God Class rule revision (docs/godclass_formula_revision.md)
only changed ml/preprocessing/label_rules.py::is_god_class. It did not touch
ast_parser.py, metrics.py, or graph_builder.py -- so node identity, node
order, node count, and every STRUCTURAL feature column are unchanged between
the previous hybrid build and the just-rebuilt structural graphs. Only the
god_class y values (and possibly long_method/feature_envy y counts, checked
per-file below, not assumed) can differ. Verified before trusting this on
the full corpus: for click/src/click/testing.py, struct and (pre-patch)
hybrid class.x first-3-columns were byte-identical for all 7 nodes, node
order/count matched, and only y differed (CliRunner: struct y=1, stale
hybrid y=0) -- exactly the expected shape of the fix.

For each (node_type, file) pair: if node counts AND the shared structural
feature columns (x[:, :n_struct_cols]) match exactly, copy y from struct to
hybrid in place. If they don't match for some node type, the file is
skipped and reported -- it needs a full CodeBERT re-embed, not a patch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STRUCT_DIR = ROOT / "data" / "processed" / "graphs"
HYBRID_DIR = ROOT / "data" / "processed" / "graphs_hybrid"
SPLITS = ["train", "val"]

# node_type -> number of leading structural columns in the hybrid x
# (matches ml.graph.graph_builder's per-type structural feature width)
STRUCT_COLS = {"class": 3, "method": 5, "function": 5}


def main():
    patched, skipped, mismatched = 0, 0, []
    for split in SPLITS:
        struct_files = sorted((STRUCT_DIR / split).rglob("*.pt"))
        for sp in struct_files:
            rel = sp.relative_to(STRUCT_DIR / split)
            hp = HYBRID_DIR / split / rel
            if not hp.exists():
                skipped += 1
                continue
            d_struct = torch.load(sp, weights_only=False)
            d_hybrid = torch.load(hp, weights_only=False)
            ok = True
            for ntype, n_struct_cols in STRUCT_COLS.items():
                if ntype not in d_struct.node_types or ntype not in d_hybrid.node_types:
                    continue
                xs = d_struct[ntype].x
                xh = d_hybrid[ntype].x
                if xs.shape[0] != xh.shape[0]:
                    ok = False
                    break
                if xs.shape[0] == 0:
                    continue
                if not torch.equal(xs[:, :n_struct_cols], xh[:, :n_struct_cols]):
                    ok = False
                    break
            if not ok:
                mismatched.append(str(rel))
                continue
            changed_this_file = False
            for ntype in STRUCT_COLS:
                if ntype not in d_struct.node_types or ntype not in d_hybrid.node_types:
                    continue
                if not hasattr(d_struct[ntype], "y") or not hasattr(d_hybrid[ntype], "y"):
                    continue
                ys, yh = d_struct[ntype].y, d_hybrid[ntype].y
                if ys.shape[0] == yh.shape[0] and ys.shape[0] > 0 and not torch.equal(ys, yh):
                    d_hybrid[ntype].y = ys.clone()
                    changed_this_file = True
            if changed_this_file:
                torch.save(d_hybrid, hp)
                patched += 1

    print(f"Patched (label changed): {patched}")
    print(f"Skipped (no hybrid counterpart): {skipped}")
    print(f"Mismatched (structural drift, needs full re-embed): {len(mismatched)}")
    for m in mismatched[:30]:
        print(f"  {m}")


if __name__ == "__main__":
    main()
