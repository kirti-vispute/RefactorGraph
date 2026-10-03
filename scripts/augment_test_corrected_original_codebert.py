# -*- coding: utf-8 -*-
"""ONE-TIME CodeBERT augmentation of the corrected_original TEST
structural graphs (authorized: final TEST evaluation, user-approved
scope). Same procedure as scripts/augment_graphs_with_codebert.py,
parameterized to a different source/output directory instead of
modifying that production script or its TRAIN/VAL-only SPLITS list.

Input: data/processed/graphs_test_corrected_original/ (already built,
verified against corrected_dominant, 1574/1574 files, current parser).
Output: data/processed/graphs_hybrid_test_corrected_original/ (NEW,
separate from the historical data/processed/graphs_hybrid/test, which
is never read or written by this script).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from scripts.build_dataset import parse_split
from scripts.extract_codebert_embeddings import mean_pool

STRUCT_DIR = ROOT / "data" / "processed" / "graphs_test_corrected_original"
OUT_DIR = ROOT / "data" / "processed" / "graphs_hybrid_test_corrected_original"
RAW_TEST_DIR = ROOT / "data" / "raw" / "test"

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 256
BATCH_SIZE = 16


def _snippet(text_lines, lineno, end_lineno):
    if not text_lines:
        return ""
    return "\n".join(text_lines[lineno - 1:end_lineno])


@torch.no_grad()
def embed_all(sources, tokenizer, model, device):
    if not sources:
        return torch.zeros((0, model.config.hidden_size))
    all_embs = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE), desc="embedding", leave=False):
        batch = sources[i:i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
        out = model(**enc).last_hidden_state
        pooled = mean_pool(out, enc["attention_mask"])
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    if OUT_DIR.exists() and any(OUT_DIR.rglob("*.pt")):
        print(f"[skip] {OUT_DIR} already populated, not re-augmenting")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("[parse] test ...")
    modules, _parse_errors, _file_count = parse_split("test")
    print(f"[parse] test: {len(modules)} modules")

    class_sources, method_sources, function_sources = [], [], []
    file_plan = []
    for repo, relpath, mod in modules:
        full_path = RAW_TEST_DIR / repo / relpath
        try:
            text_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            text_lines = []

        c_off, m_off, f_off = len(class_sources), len(method_sources), len(function_sources)
        for cls in mod.classes:
            class_sources.append(_snippet(text_lines, cls.lineno, cls.end_lineno))
            for fn in cls.methods:
                method_sources.append(_snippet(text_lines, fn.lineno, fn.end_lineno))
        for fn in mod.functions:
            function_sources.append(_snippet(text_lines, fn.lineno, fn.end_lineno))

        c_n = len(class_sources) - c_off
        m_n = len(method_sources) - m_off
        f_n = len(function_sources) - f_off
        file_plan.append((repo, relpath, c_off, c_n, m_off, m_n, f_off, f_n))

    print(f"[test] n_class={len(class_sources)} n_method={len(method_sources)} n_function={len(function_sources)}")
    class_emb = embed_all(class_sources, tokenizer, model, device)
    method_emb = embed_all(method_sources, tokenizer, model, device)
    function_emb = embed_all(function_sources, tokenizer, model, device)

    n_saved = 0
    for repo, relpath, c_off, c_n, m_off, m_n, f_off, f_n in tqdm(file_plan, desc="augment test"):
        in_path = STRUCT_DIR / repo / (relpath.replace("\\", "/") + ".pt")
        if not in_path.exists():
            raise FileNotFoundError(f"{in_path} missing")
        data = torch.load(in_path, weights_only=False)

        assert data["class"].x.shape[0] == c_n
        assert data["method"].x.shape[0] == m_n
        assert data["function"].x.shape[0] == f_n

        data["class"].x = torch.cat([data["class"].x, class_emb[c_off:c_off + c_n]], dim=1)
        data["method"].x = torch.cat([data["method"].x, method_emb[m_off:m_off + m_n]], dim=1)
        data["function"].x = torch.cat([data["function"].x, function_emb[f_off:f_off + f_n]], dim=1)

        out_path = OUT_DIR / repo / (relpath.replace("\\", "/") + ".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        n_saved += 1

    print(f"[done] saved {n_saved} hybrid TEST graphs (corrected_original) to {OUT_DIR}")


if __name__ == "__main__":
    main()
