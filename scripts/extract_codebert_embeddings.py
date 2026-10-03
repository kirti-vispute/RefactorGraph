# -*- coding: utf-8 -*-
"""Phase 8 step 1: precompute frozen CodeBERT (microsoft/codebert-base)
embeddings for every method/function/class source snippet in TRAIN and VAL
(TEST is left untouched, same discipline as every prior phase).

Frozen, not fine-tuned, per the staged training plan (frozen CodeBERT
baseline first; fine-tuning is a later phase). Embeddings are computed once
and cached to disk so the classifier-training step (train_codebert_baseline.py)
can iterate cheaply on CPU without re-running the transformer forward pass.

Pooling: mean-pool over the attention mask, not the raw [CLS] token.
CodeBERT is pretrained with MLM + replaced-token-detection, not a
sentence-pair/NSP objective, so its [CLS] embedding has no particular
training signal for whole-snippet representation. Mean pooling over all
real (non-padding) token embeddings is the standard, better-justified choice
for a frozen feature-extraction baseline.

Truncation: max_length=256 sub-word tokens. This truncates some long
methods/classes (documented, not hidden) — this is a known limitation of the
frozen-embedding baseline, revisited if it turns out to matter after
comparing against Phase 6/7.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
EMB_DIR = ROOT / "models" / "codebert_embeddings"

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 256
BATCH_SIZE = 16

SPLITS = ["train", "val"]
KINDS = ["methods", "classes"]


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts  # (B, H) mean-pooled over real (non-padding) tokens


@torch.no_grad()
def embed_sources(sources: list, tokenizer, model, device) -> torch.Tensor:
    all_embs = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE), desc="embedding", leave=False):
        batch = sources[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        ).to(device)
        out = model(**enc).last_hidden_state  # (B, T, 768)
        pooled = mean_pool(out, enc["attention_mask"])
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for kind in KINDS:
            jsonl_path = PROCESSED_DIR / f"{split}_{kind}.jsonl"
            out_path = EMB_DIR / f"{split}_{kind}.pt"
            if out_path.exists():
                print(f"skip {out_path} (already exists)")
                continue
            records = load_jsonl(jsonl_path)
            sources = [r["source"] for r in records]
            print(f"{split}/{kind}: n={len(sources)}")
            embs = embed_sources(sources, tokenizer, model, device)
            assert embs.shape == (len(records), model.config.hidden_size)
            torch.save(embs, out_path)
            print(f"saved {out_path} shape={tuple(embs.shape)}")


if __name__ == "__main__":
    main()
