# -*- coding: utf-8 -*-
"""EXPERIMENT-ONLY hybrid-graph patch (authorized: docs/feature_envy_generalization_investigation.md
section 10). Avoids a ~1-2hr CodeBERT re-embedding pass: only ONE
structural column (index 4: external_access_count -> dominant_external_count)
changed for method/function nodes; CodeBERT embeddings depend only on
source text/snippets, which are unchanged, so they are copied verbatim
from the existing data/processed/graphs_hybrid/{train,val} (the
currently-deployed candidate's own training data -- read-only, never
written to) rather than recomputed. Same "verify byte-identical, then
patch" discipline already used and documented in this project
(docs/godclass_formula_revision.md section 9's label-patch).

Output: data/processed/graphs_hybrid_experiment_fe_dominant/{train,val}
(new directory -- production data/processed/graphs_hybrid/{train,val}
is only ever read, never written).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STRUCT_DIR = ROOT / "data" / "processed" / "graphs_experiment_fe_dominant"
PROD_HYBRID_DIR = ROOT / "data" / "processed" / "graphs_hybrid"
OUT_HYBRID_DIR = ROOT / "data" / "processed" / "graphs_hybrid_experiment_fe_dominant"
SPLITS = ["train", "val"]

# Columns 0-3 (loc, statement_count, param_count, self_access_count) must
# be byte-identical between the experiment structural rebuild and the
# production hybrid file -- only column 4 (index 4) is expected to
# differ (external_access_count -> dominant_external_count). Verified
# per-file before trusting the patch; any mismatch outside column 4
# aborts that file (reported, not silently patched).
CHECK_COLS = 4  # verify columns [0:4) match; patch column 4


def main():
    patched, mismatched, missing = 0, [], []
    for split in SPLITS:
        struct_files = sorted((STRUCT_DIR / split).rglob("*.pt"))
        for sp in struct_files:
            rel = sp.relative_to(STRUCT_DIR / split)
            hp = PROD_HYBRID_DIR / split / rel
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
                # y tensors (labels) must also match exactly -- confirms
                # this experiment changed ONLY the feature, never the label.
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

            # Patch: copy the hybrid file, replace column 4 for method/function.
            for ntype in ("method", "function"):
                if d_hybrid[ntype].x.shape[0] == 0:
                    continue
                new_x = d_hybrid[ntype].x.clone()
                new_x[:, 4] = d_struct[ntype].x[:, 4]
                d_hybrid[ntype].x = new_x

            out_path = OUT_HYBRID_DIR / split / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(d_hybrid, out_path)
            patched += 1

    print(f"Patched (new experiment hybrid file written): {patched}")
    print(f"Missing production hybrid counterpart (skipped): {len(missing)}")
    for m in missing[:10]:
        print(f"  {m}")
    print(f"Mismatched (columns 0-3 or labels differ -- NOT patched, needs investigation): {len(mismatched)}")
    for m in mismatched[:10]:
        print(f"  {m}")

    if mismatched or missing:
        print("\n[WARNING] Not all files patched cleanly -- see above before trusting downstream training.")
    else:
        print("\n[OK] Every file verified byte-identical outside column 4 (method/function) before patching.")


if __name__ == "__main__":
    main()
