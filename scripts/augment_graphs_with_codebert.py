# -*- coding: utf-8 -*-
"""Phase 9 step 1: augment the Phase 4/5 structure-only graphs with frozen
CodeBERT semantics, producing the hybrid graph dataset for Phase 9 fusion
training.

For every saved graph (data/processed/graphs/{split}/{repo}/{relpath}.pt),
re-parses the same source file, re-derives the source snippet for every
class/method/function node in EXACTLY the iteration order
ml/graph/graph_builder.py uses to build that graph's node features
(mod.classes enumerate order for "class"; mod.classes -> cls.methods order
for "method"; mod.functions order for "function"), embeds each snippet with
frozen microsoft/codebert-base (same mean-pooling as
extract_codebert_embeddings.py), and concatenates the 768-dim embedding onto
the existing structural feature vector for that node. module/attribute/
parameter/import nodes are left untouched (no source-text unit of their own
to embed).

A shape assertion per file catches any ordering drift between this script
and graph_builder.py immediately, rather than silently misaligning
embeddings to the wrong nodes.

TRAIN and VAL only — TEST graphs are augmented once, in the final Phase 16
evaluation step, same discipline as extract_codebert_embeddings.py.

Output: data/processed/graphs_hybrid/{split}/{repo}/{relpath}.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from scripts.build_dataset import RAW_DIR, parse_split
from scripts.extract_codebert_embeddings import mean_pool

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "data" / "processed" / "graphs"
HYBRID_GRAPH_DIR = ROOT / "data" / "processed" / "graphs_hybrid"

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
SPLITS = ["train", "val"]


def _snippet(text_lines: list, lineno: int, end_lineno: int) -> str:
    if not text_lines:
        return ""
    return "\n".join(text_lines[lineno - 1:end_lineno])


@torch.no_grad()
def embed_all(sources: list, tokenizer, model, device) -> torch.Tensor:
    if not sources:
        return torch.zeros((0, model.config.hidden_size))
    all_embs = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE), desc="embedding", leave=False):
        batch = sources[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt",
        ).to(device)
        out = model(**enc).last_hidden_state
        pooled = mean_pool(out, enc["attention_mask"])
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    for split in SPLITS:
        out_split_dir = HYBRID_GRAPH_DIR / split
        if out_split_dir.exists() and any(out_split_dir.rglob("*.pt")):
            print(f"skip {split} (graphs_hybrid/{split} already populated)")
            continue

        print(f"[parse] {split} ...")
        modules, _parse_errors, _file_count = parse_split(split)
        print(f"[parse] {split}: {len(modules)} modules")

        # Pass 1: collect every snippet across the whole split (one big batch
        # for the transformer forward pass) plus per-file offsets so the
        # resulting embeddings can be sliced back out per file/node-type.
        class_sources, method_sources, function_sources = [], [], []
        file_plan = []  # (repo, relpath, mod, c_off, c_n, m_off, m_n, f_off, f_n)
        for repo, relpath, mod in modules:
            full_path = RAW_DIR / split / repo / relpath
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

        print(f"[{split}] n_class={len(class_sources)} n_method={len(method_sources)} n_function={len(function_sources)}")
        class_emb = embed_all(class_sources, tokenizer, model, device)
        method_emb = embed_all(method_sources, tokenizer, model, device)
        function_emb = embed_all(function_sources, tokenizer, model, device)

        # Pass 2: reload each saved structural graph, concatenate the
        # matching embedding slice onto its existing node features, save to
        # the hybrid directory.
        n_saved = 0
        for repo, relpath, c_off, c_n, m_off, m_n, f_off, f_n in tqdm(file_plan, desc=f"augment {split}"):
            in_path = GRAPH_DIR / split / repo / (relpath.replace("\\", "/") + ".pt")
            if not in_path.exists():
                raise FileNotFoundError(
                    f"{in_path} missing — run scripts/build_graphs.py before this script"
                )
            data = torch.load(in_path, weights_only=False)

            assert data["class"].x.shape[0] == c_n, (
                f"{in_path}: class count mismatch graph={data['class'].x.shape[0]} vs reparsed={c_n}"
            )
            assert data["method"].x.shape[0] == m_n, (
                f"{in_path}: method count mismatch graph={data['method'].x.shape[0]} vs reparsed={m_n}"
            )
            assert data["function"].x.shape[0] == f_n, (
                f"{in_path}: function count mismatch graph={data['function'].x.shape[0]} vs reparsed={f_n}"
            )

            data["class"].x = torch.cat([data["class"].x, class_emb[c_off:c_off + c_n]], dim=1)
            data["method"].x = torch.cat([data["method"].x, method_emb[m_off:m_off + m_n]], dim=1)
            data["function"].x = torch.cat([data["function"].x, function_emb[f_off:f_off + f_n]], dim=1)

            out_path = HYBRID_GRAPH_DIR / split / repo / (relpath.replace("\\", "/") + ".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, out_path)
            n_saved += 1

        print(f"[{split}] saved {n_saved} hybrid graphs to {HYBRID_GRAPH_DIR / split}")


if __name__ == "__main__":
    main()
