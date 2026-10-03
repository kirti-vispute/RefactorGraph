# -*- coding: utf-8 -*-
"""Robustness Phase 4/5: run the retrained, corrected-data checkpoint
(models/hybrid_class_pool_tuned_fixed_data, see docs/post_fix_retrain_comparison.md)
against:
  (Phase 5) scratch_external_eval/manual/*.py -- 4 independently-written,
    silver-label-verified targeted cases (clear Long Method, clear God Class,
    clear Feature Envy, clean/no-smell).
  (Phase 4) scratch_external_eval/zikazaki/*.py -- real files from
    ZikaZaki/code-smells-python (MIT license), verified against OUR OWN
    silver-label rules (not trusted from filenames) earlier in this phase.

This is a read-only evaluation script: it does not train on these files,
does not touch TRAIN/VAL/TEST, and does not modify the checkpoint. It exists
because backend/app/inference.py intentionally still serves the FROZEN old
checkpoint (models/hybrid_class_pool_tuned) -- switching the live backend to
a new model is a deliberate final decision (Phase 8/V), not an automatic
side effect of retraining.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModel, AutoTokenizer

from ml.graph.graph_builder import build_hetero_graph
from ml.models.gat_baseline import HeteroGAT
from ml.preprocessing.ast_parser import parse_file
from ml.preprocessing.label_rules import LabelThresholds
from scripts.augment_graphs_with_codebert import _snippet
from scripts.extract_codebert_embeddings import mean_pool
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "hybrid_class_pool_tuned_fixed_data"
_DUMMY_THRESHOLDS = LabelThresholds(long_method_statements=15, god_class_method_count=10, god_class_loc=192)


def load_model():
    device = torch.device("cpu")
    params = json.loads((MODEL_DIR / "best_params.json").read_text(encoding="utf-8"))
    model = HeteroGAT(
        hidden_dim=params["hidden_dim"], heads=params["heads"], dropout=params["dropout"],
        node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", weights_only=True))
    model.eval()
    raw_norm = torch.load(MODEL_DIR / "norm_stats.pt", weights_only=False)
    norm_stats = {nt: (torch.tensor(m), torch.tensor(s)) for nt, (m, s) in raw_norm.items()}
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    codebert.eval()
    return device, model, norm_stats, tokenizer, codebert


@torch.no_grad()
def embed(sources, tokenizer, codebert, device):
    if not sources:
        return torch.zeros((0, codebert.config.hidden_size))
    enc = tokenizer(sources, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
    out = codebert(**enc).last_hidden_state
    return mean_pool(out, enc["attention_mask"]).cpu()


@torch.no_grad()
def predict_file(path, device, model, norm_stats, tokenizer, codebert):
    mod = parse_file(path)
    if not mod.ok:
        return {"error": mod.parse_error}

    data = build_hetero_graph(mod, _DUMMY_THRESHOLDS).data
    text_lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    class_sources, method_sources, function_sources = [], [], []
    method_names, function_names, class_names = [], [], []
    for cls in mod.classes:
        class_sources.append(_snippet(text_lines, cls.lineno, cls.end_lineno))
        class_names.append(cls.qualname)
        for fn in cls.methods:
            method_sources.append(_snippet(text_lines, fn.lineno, fn.end_lineno))
            method_names.append(fn.qualname)
    for fn in mod.functions:
        function_sources.append(_snippet(text_lines, fn.lineno, fn.end_lineno))
        function_names.append(fn.qualname)

    data["class"].x = torch.cat([data["class"].x, embed(class_sources, tokenizer, codebert, device)], dim=1)
    data["method"].x = torch.cat([data["method"].x, embed(method_sources, tokenizer, codebert, device)], dim=1)
    data["function"].x = torch.cat([data["function"].x, embed(function_sources, tokenizer, codebert, device)], dim=1)

    for nt, (mean, std) in norm_stats.items():
        if data[nt].x.shape[0] > 0:
            data[nt].x = (data[nt].x - mean) / std

    h = model(data.x_dict, data.edge_index_dict)
    out = {"long_method": [], "feature_envy": [], "god_class": []}

    if "method" in h and data["method"].x.shape[0] > 0:
        lm = torch.sigmoid(model.predict_long_method(h, "method")).tolist()
        fe = torch.sigmoid(model.predict_feature_envy(h)).tolist()
        for i, name in enumerate(method_names):
            out["long_method"].append((name, round(lm[i], 3)))
            out["feature_envy"].append((name, round(fe[i], 3)))
    if "function" in h and data["function"].x.shape[0] > 0:
        lm_f = torch.sigmoid(model.predict_long_method(h, "function")).tolist()
        for i, name in enumerate(function_names):
            out["long_method"].append((name, round(lm_f[i], 3)))
    if "class" in h and data["class"].x.shape[0] > 0:
        gc = torch.sigmoid(model.predict_god_class(h)).tolist()
        for i, name in enumerate(class_names):
            out["god_class"].append((name, round(gc[i], 3)))

    return out


def main():
    device, model, norm_stats, tokenizer, codebert = load_model()

    for group, pattern in [("MANUAL (Phase 5)", "scratch_external_eval/manual/*.py"),
                            ("ZIKAZAKI (Phase 4)", "scratch_external_eval/zikazaki/*.py")]:
        print(f"\n{'=' * 70}\n{group}\n{'=' * 70}")
        for path in sorted(glob.glob(str(ROOT / pattern))):
            print(f"\n--- {Path(path).name} ---")
            result = predict_file(path, device, model, norm_stats, tokenizer, codebert)
            if "error" in result:
                print("  PARSE ERROR:", result["error"])
                continue
            for task in ("long_method", "feature_envy", "god_class"):
                for name, prob in result[task]:
                    flag = " <<< PREDICTED POSITIVE" if prob >= 0.5 else ""
                    if prob >= 0.5 or prob >= 0.2:  # show anything non-trivially close too
                        print(f"  [{task}] {name}: {prob}{flag}")


if __name__ == "__main__":
    main()
