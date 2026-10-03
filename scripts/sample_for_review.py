# -*- coding: utf-8 -*-
"""Stratified sample of silver-labeled units for manual precision review.

Pulls N positives + N negatives per smell type from the TRAIN split (review
happens on train, not test, to keep the test set untouched per the
leakage-prevention rules) and writes them to a human-readable markdown file
with source snippets so each one can be read and judged True/False Positive.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "label_review_sample.md"

random.seed(42)
N_PER_BUCKET = 15


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sample_bucket(units, key, value, n):
    pool = [u for u in units if u[key] == value]
    return random.sample(pool, min(n, len(pool)))


def main():
    methods = load_jsonl(PROCESSED / "train_methods.jsonl")
    classes = load_jsonl(PROCESSED / "train_classes.jsonl")

    lm_pos = sample_bucket(methods, "long_method", True, N_PER_BUCKET)
    lm_neg = sample_bucket([m for m in methods if m["loc"] >= 15], "long_method", False, N_PER_BUCKET)
    fe_pos = sample_bucket(methods, "feature_envy", True, N_PER_BUCKET)
    fe_neg = sample_bucket([m for m in methods if m["dominant_external_count"] >= 1], "feature_envy", False, N_PER_BUCKET)
    gc_pos = sample_bucket(classes, "god_class", True, N_PER_BUCKET)
    gc_neg = sample_bucket([c for c in classes if c["method_count"] >= 5], "god_class", False, N_PER_BUCKET)

    lines = ["# Silver Label Review Sample (train split only)\n",
             "Each item: predicted label, source, blank verdict line to fill in "
             "(TP / FP / TN / FN) after reading the code.\n"]

    def emit(title, items, label_key, positive):
        lines.append(f"\n## {title} ({'predicted positive' if positive else 'predicted negative'})\n")
        for i, u in enumerate(items, 1):
            lines.append(f"\n### {title} #{i} — {u['repo']}/{u['file']}::{u['qualname']} (line {u['lineno']})\n")
            lines.append(f"predicted `{label_key}={u[label_key]}`")
            if "loc" in u:
                lines.append(f", loc={u['loc']}")
            if "statement_count" in u:
                lines.append(f", statement_count={u['statement_count']}")
            if "dominant_external_receiver" in u:
                lines.append(f", dominant_external_receiver={u['dominant_external_receiver']}, "
                              f"dominant_external_count={u['dominant_external_count']}, "
                              f"self_access_count={u['self_access_count']}")
            if "method_count" in u:
                lines.append(f", method_count={u['method_count']}")
            lines.append("\n```python\n" + u["source"][:2000] + "\n```\n")
            lines.append("VERDICT: \n")

    emit("Long Method", lm_pos, "long_method", True)
    emit("Long Method", lm_neg, "long_method", False)
    emit("Feature Envy", fe_pos, "feature_envy", True)
    emit("Feature Envy", fe_neg, "feature_envy", False)
    emit("God/Large Class", gc_pos, "god_class", True)
    emit("God/Large Class", gc_neg, "god_class", False)

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} — {len(lm_pos)+len(lm_neg)} long_method, {len(fe_pos)+len(fe_neg)} feature_envy, "
          f"{len(gc_pos)+len(gc_neg)} god_class items")


if __name__ == "__main__":
    main()
