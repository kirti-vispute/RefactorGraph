# -*- coding: utf-8 -*-
"""Phase 16: final, one-time TEST evaluation.

TEST (data/raw/test/{celery,django,pandas}) has been held out untouched
through every phase up to and including Phase 13e's Design B tuning
(train_hybrid_baseline.py, tune_hybrid_class_pool_optuna.py,
tune_hybrid_class_pool_fixed_arch.py all read only TRAIN/VAL). This script
is the single, first TEST touch in the project, run once, after all
architecture/hyperparameter/loss-weight decisions were already made on
VAL. No decision made here feeds back into training or search — this is
report-only.

Step 1: augment the existing structural TEST graphs
(data/processed/graphs/test/**/*.pt, already built by scripts/build_graphs.py)
with frozen CodeBERT embeddings, same procedure as
scripts/augment_graphs_with_codebert.py (which explicitly deferred TEST to
this step) -> data/processed/graphs_hybrid/test/**/*.pt. Skipped if already
populated (idempotent re-run of this script does not re-touch TEST via the
network/model twice).

Step 2: evaluate the 3 checkpoints referenced throughout Phase 13's VAL
comparisons -- Phase 10 hybrid (models/hybrid_tuned), Design B untuned
(models/hybrid_class_pool), Design B tuned (models/hybrid_class_pool_tuned,
selected in docs/class_pool_fixed_arch_report.md as the best config across
all three Design B search variants) -- on the same TEST split, each using
its OWN saved norm_stats (fit on TRAIN only, never refit here). Reports
per-task F1 + macro-F1 for all three, matching the VAL comparison table
structure exactly, so the TEST numbers can be read directly against the
already-published VAL numbers.

Output: data/processed/graphs_hybrid/test/**/*.pt (new),
docs/final_test_evaluation_report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ml.models.gat_baseline import HeteroGAT
from scripts.augment_graphs_with_codebert import GRAPH_DIR, _snippet, embed_all
from scripts.build_dataset import RAW_DIR, parse_split
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS, apply_norm, run_eval

ROOT = Path(__file__).resolve().parents[1]
HYBRID_GRAPH_DIR = ROOT / "data" / "processed" / "graphs_hybrid"
DOCS_DIR = ROOT / "docs"
BATCH_SIZE = 64

CHECKPOINTS = [
    ("Phase 10 Hybrid", ROOT / "models" / "hybrid_tuned", False),
    ("Design B untuned", ROOT / "models" / "hybrid_class_pool", True),
    ("Design B tuned (DEPLOYED/frozen)", ROOT / "models" / "hybrid_class_pool_tuned", True),
    # Added for the final decision-point re-evaluation
    # (docs/godclass_formula_revision.md sections 9-12): God-Class-rule
    # fix + collaborator-chain fix, retrained, seed=43 selected via
    # non-cherry-picked multi-seed macro-F1 comparison. This is the
    # candidate under consideration to replace the deployed checkpoint
    # above -- see that doc for the full evidence trail.
    ("God-Class-fix candidate (seed=43)", ROOT / "models" / "hybrid_class_pool_tuned_fixed_data", True),
]


def augment_test_split():
    out_split_dir = HYBRID_GRAPH_DIR / "test"
    if out_split_dir.exists() and any(out_split_dir.rglob("*.pt")):
        print("[augment] test already populated, skipping CodeBERT pass")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[augment] device={device}")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    model.eval()

    print("[augment] parsing test ...")
    modules, _parse_errors, _file_count = parse_split("test")
    print(f"[augment] test: {len(modules)} modules")

    class_sources, method_sources, function_sources = [], [], []
    file_plan = []
    for repo, relpath, mod in modules:
        full_path = RAW_DIR / "test" / repo / relpath
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

    print(f"[augment] n_class={len(class_sources)} n_method={len(method_sources)} n_function={len(function_sources)}")
    class_emb = embed_all(class_sources, tokenizer, model, device)
    method_emb = embed_all(method_sources, tokenizer, model, device)
    function_emb = embed_all(function_sources, tokenizer, model, device)

    n_saved = 0
    for repo, relpath, c_off, c_n, m_off, m_n, f_off, f_n in tqdm(file_plan, desc="augment test"):
        in_path = GRAPH_DIR / "test" / repo / (relpath.replace("\\", "/") + ".pt")
        if not in_path.exists():
            raise FileNotFoundError(f"{in_path} missing -- run scripts/build_graphs.py first")
        data = torch.load(in_path, weights_only=False)

        assert data["class"].x.shape[0] == c_n
        assert data["method"].x.shape[0] == m_n
        assert data["function"].x.shape[0] == f_n

        data["class"].x = torch.cat([data["class"].x, class_emb[c_off:c_off + c_n]], dim=1)
        data["method"].x = torch.cat([data["method"].x, method_emb[m_off:m_off + m_n]], dim=1)
        data["function"].x = torch.cat([data["function"].x, function_emb[f_off:f_off + f_n]], dim=1)

        out_path = HYBRID_GRAPH_DIR / "test" / repo / (relpath.replace("\\", "/") + ".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        n_saved += 1

    print(f"[augment] saved {n_saved} hybrid test graphs to {HYBRID_GRAPH_DIR / 'test'}")


def load_norm_stats(path: Path) -> dict:
    raw = torch.load(path, weights_only=False)
    return {nt: (torch.tensor(m), torch.tensor(s)) for nt, (m, s) in raw.items()}


def evaluate_checkpoint(name: str, model_dir: Path, class_method_pool: bool, test_graphs_raw: list) -> dict:
    params = json.loads((model_dir / ("best_params.json" if (model_dir / "best_params.json").exists() else "params.json")).read_text(encoding="utf-8"))
    norm_stats = load_norm_stats(model_dir / "norm_stats.pt")

    graphs = [g.clone() for g in test_graphs_raw]
    apply_norm(graphs, norm_stats)
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    model = HeteroGAT(hidden_dim=params["hidden_dim"], heads=params["heads"], dropout=params["dropout"],
                       node_feature_dims=HYBRID_NODE_DIMS, edge_types=None,
                       class_method_pool=class_method_pool).to(device)
    state = torch.load(model_dir / "model.pt", weights_only=True)
    model.load_state_dict(state)

    metrics = run_eval(model, loader, device)
    print(f"[eval] {name}: " + " ".join(
        f"{task}_f1={v['f1']:.3f}" if v else f"{task}_f1=n/a" for task, v in metrics.items()
    ))
    return metrics


def macro_f1(metrics: dict):
    f1s = [v["f1"] for v in metrics.values() if v is not None]
    return sum(f1s) / len(f1s) if f1s else None


def main():
    augment_test_split()

    print("[load] reading hybrid test graphs ...")
    test_graphs_raw = [torch.load(p, weights_only=False) for p in sorted((HYBRID_GRAPH_DIR / "test").rglob("*.pt"))]
    print(f"[load] test={len(test_graphs_raw)}")

    results = {}
    for name, model_dir, class_method_pool in CHECKPOINTS:
        results[name] = evaluate_checkpoint(name, model_dir, class_method_pool, test_graphs_raw)

    lines = ["# Final TEST Evaluation Report (Phase 16)\n"]
    lines.append(
        "One-time TEST evaluation, run after all architecture/hyperparameter/loss-weight "
        "decisions were made on TRAIN/VAL alone (Phase 9 through Phase 13e). This is the "
        f"first and only script in the project to read data/raw/test. n_test_graphs={len(test_graphs_raw)}.\n"
    )
    lines.append(
        "\n## Comparison: " + " vs ".join(name for name, _, _ in CHECKPOINTS) + " -- TEST split\n"
    )
    header = "| task | " + " | ".join(name for name, _, _ in CHECKPOINTS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    for task in ("long_method", "feature_envy", "god_class"):
        row = [task]
        for name, _, _ in CHECKPOINTS:
            v = results[name].get(task)
            row.append(f"{v['f1']:.3f}" if v else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    macro_row = ["**macro-F1**"]
    for name, _, _ in CHECKPOINTS:
        mf1 = macro_f1(results[name])
        macro_row.append(f"{mf1:.3f}" if mf1 is not None else "n/a")
    lines.append("| " + " | ".join(macro_row) + " |")

    lines.append("\n## Raw metrics per checkpoint\n")
    for name, _, _ in CHECKPOINTS:
        lines.append(f"\n### {name}\n")
        for task, v in results[name].items():
            if v is None:
                lines.append(f"- {task}: no eligible nodes in test split\n")
                continue
            lines.append(
                f"- {task}: precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} "
                f"roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f} "
                f"(n={v['n']}, positives={v['n_positive']}, tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']})"
            )

    (DOCS_DIR / "final_test_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report to {DOCS_DIR / 'final_test_evaluation_report.md'}")


if __name__ == "__main__":
    main()
