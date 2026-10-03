# -*- coding: utf-8 -*-
"""Loads the final selected model (Design B tuned, God-Class-fix candidate,
models/hybrid_class_pool_tuned_fixed_data, seed=43 -- see
docs/final_test_evaluation_report.md Phase 16 and
docs/godclass_formula_revision.md sections 9-14 for the full selection
trail) once at process startup and runs it on arbitrary Python source
submitted to the API. Promoted from the previously-deployed
models/hybrid_class_pool_tuned (TEST macro-F1 0.650) after a real,
TEST-measured improvement (macro-F1 0.677) driven by a corrected God
Class label formula and a collaborator-chain double-counting fix,
retrained and seed-selected on a non-cherry-picked (macro-F1) criterion.

Same preprocessing pipeline as training, applied to a single in-memory file
instead of the on-disk dataset: ast_parser.parse_source -> graph_builder.
build_hetero_graph (structural features + the ACTUAL graph edges, Phase 17)
-> frozen CodeBERT embeddings concatenated onto class/method/function nodes
(same procedure as scripts/augment_graphs_with_codebert.py) -> the model's
own saved norm_stats (fit on TRAIN only, never refit here) -> HeteroGAT
forward pass. Model weights, architecture, and this pipeline are frozen --
Phase 17 only adds READ access to data graph_builder already computes
(the HeteroData's own edge_index_dict, and the structural feature values
already used to build the model's input) that Phase 16's response
discarded; it does not change what is computed for prediction.

Node ids are f"{node_type}:{index}", where index matches the row order
ml/graph/graph_builder.py used to build that node type's feature tensor
(see ml/graph/node_naming.py) -- the same order the model's own output
tensors are in, so predictions line up with graph nodes by construction,
not by name-matching.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModel, AutoTokenizer

from ml.graph.graph_builder import build_hetero_graph
from ml.graph.node_naming import build_node_records
from ml.models.gat_baseline import HeteroGAT
from ml.preprocessing.ast_parser import ModuleInfo, parse_source
from ml.preprocessing.label_rules import LabelThresholds
from ml.preprocessing.metrics import class_metrics, function_metrics
from scripts.augment_graphs_with_codebert import _snippet
from scripts.extract_codebert_embeddings import mean_pool
from scripts.train_hybrid_baseline import HYBRID_NODE_DIMS

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "experiment_fe_dominant" / "seed_43"
CODEBERT_NAME = "microsoft/codebert-base"
MAX_LENGTH = 256

# Only affects the `y` ground-truth labels graph_builder attaches to nodes
# (see ml/graph/graph_builder.py docstring), never the feature vectors the
# model reads -- a placeholder here has zero effect on predictions.
_DUMMY_THRESHOLDS = LabelThresholds(long_method_statements=15, god_class_method_count=10, god_class_loc=150)


def _explanation_for(node_type: str, mod: ModuleInfo, class_idx: Optional[int], method_like_fn) -> str:
    """A real, already-computed structural summary -- not a generated
    natural-language explanation -- built from the exact metrics
    graph_builder used as this node's own feature vector (see
    ml/preprocessing/metrics.py). Reused rather than re-derived from the
    (CodeBERT-augmented, normalized) tensor so the numbers stay
    human-readable."""
    if node_type == "class":
        cm = class_metrics(mod.classes[class_idx])
        return f"loc={cm.loc}, methods={cm.method_count}, fields={cm.field_count}"
    fm = function_metrics(method_like_fn, mod)
    return (
        f"loc={fm.loc}, statements={fm.statement_count}, params={fm.param_count}, "
        f"self_access={fm.self_access_count}, external_access={fm.external_access_count}"
    )


class ModelBundle:
    def __init__(self):
        self.device = torch.device("cpu")

        params = json.loads((MODEL_DIR / "best_params.json").read_text(encoding="utf-8"))
        self.model = HeteroGAT(
            hidden_dim=params["hidden_dim"], heads=params["heads"], dropout=params["dropout"],
            node_feature_dims=HYBRID_NODE_DIMS, edge_types=None, class_method_pool=True,
        ).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_DIR / "model.pt", weights_only=True))
        self.model.eval()

        raw_norm = torch.load(MODEL_DIR / "norm_stats.pt", weights_only=False)
        self.norm_stats = {nt: (torch.tensor(m), torch.tensor(s)) for nt, (m, s) in raw_norm.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(CODEBERT_NAME)
        self.codebert = AutoModel.from_pretrained(CODEBERT_NAME).to(self.device)
        self.codebert.eval()

    @torch.no_grad()
    def _embed(self, sources: List[str]) -> torch.Tensor:
        if not sources:
            return torch.zeros((0, self.codebert.config.hidden_size))
        enc = self.tokenizer(
            sources, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt",
        ).to(self.device)
        out = self.codebert(**enc).last_hidden_state
        return mean_pool(out, enc["attention_mask"]).cpu()

    def _apply_norm(self, data):
        for nt, (mean, std) in self.norm_stats.items():
            if data[nt].x.shape[0] > 0:
                data[nt].x = (data[nt].x - mean) / std

    @torch.no_grad()
    def predict(self, source: str, filename: str = "input.py") -> Dict[str, object]:
        mod = parse_source(source, path=filename)
        if not mod.ok:
            raise ValueError(f"parse error: {mod.parse_error}")

        data = build_hetero_graph(mod, _DUMMY_THRESHOLDS).data
        records = build_node_records(mod, filename)

        # id lookup for edge endpoints, built before any tensor mutation --
        # index order is fixed by graph_builder and never changes below.
        node_ids: Dict[str, List[str]] = {
            nt: [f"{nt}:{i}" for i in range(len(recs))] for nt, recs in records.items()
        }

        graph_nodes = [
            {
                "id": node_ids[nt][i], "type": nt, "name": rec["name"],
                "line_start": rec["lineno"], "line_end": rec["end_lineno"],
            }
            for nt, recs in records.items() for i, rec in enumerate(recs)
        ]

        graph_edges = [
            {"source": node_ids[src_t][int(data[src_t, rel, dst_t].edge_index[0, e])],
             "target": node_ids[dst_t][int(data[src_t, rel, dst_t].edge_index[1, e])],
             "type": rel}
            for (src_t, rel, dst_t) in data.edge_types
            for e in range(data[src_t, rel, dst_t].edge_index.shape[1])
        ]

        text_lines = source.splitlines()
        class_sources, method_sources, function_sources = [], [], []
        for cls in mod.classes:
            class_sources.append(_snippet(text_lines, cls.lineno, cls.end_lineno))
            for fn in cls.methods:
                method_sources.append(_snippet(text_lines, fn.lineno, fn.end_lineno))
        for fn in mod.functions:
            function_sources.append(_snippet(text_lines, fn.lineno, fn.end_lineno))

        data["class"].x = torch.cat([data["class"].x, self._embed(class_sources)], dim=1)
        data["method"].x = torch.cat([data["method"].x, self._embed(method_sources)], dim=1)
        data["function"].x = torch.cat([data["function"].x, self._embed(function_sources)], dim=1)

        self._apply_norm(data)

        h = self.model(data.x_dict, data.edge_index_dict)

        def probs_for(logits: torch.Tensor) -> List[float]:
            return torch.sigmoid(logits).tolist()

        results: Dict[str, list] = {"long_method": [], "feature_envy": [], "god_class": []}
        method_recs = records["method"]
        function_recs = records["function"]
        class_recs = records["class"]

        method_like_fns = [fn for c in mod.classes for fn in c.methods]

        if "method" in h and data["method"].x.shape[0] > 0:
            lm = probs_for(self.model.predict_long_method(h, "method"))
            fe = probs_for(self.model.predict_feature_envy(h))
            for i, rec in enumerate(method_recs):
                class_name = rec["name"].rsplit(".", 1)[0]
                explanation = _explanation_for("method", mod, None, method_like_fns[i])
                common = {
                    "node_id": node_ids["method"][i], "name": rec["name"], "node_type": "method",
                    "file": filename, "class_name": class_name,
                    "line_start": rec["lineno"], "line_end": rec["end_lineno"], "explanation": explanation,
                }
                results["long_method"].append({**common, "probability": lm[i], "predicted": lm[i] >= 0.5})
                results["feature_envy"].append({**common, "probability": fe[i], "predicted": fe[i] >= 0.65})

        if "function" in h and data["function"].x.shape[0] > 0:
            lm_f = probs_for(self.model.predict_long_method(h, "function"))
            for i, rec in enumerate(function_recs):
                explanation = _explanation_for("function", mod, None, mod.functions[i])
                results["long_method"].append({
                    "node_id": node_ids["function"][i], "name": rec["name"], "node_type": "function",
                    "file": filename, "class_name": None,
                    "line_start": rec["lineno"], "line_end": rec["end_lineno"], "explanation": explanation,
                    "probability": lm_f[i], "predicted": lm_f[i] >= 0.5,
                })

        if "class" in h and data["class"].x.shape[0] > 0:
            gc = probs_for(self.model.predict_god_class(h))
            for i, rec in enumerate(class_recs):
                explanation = _explanation_for("class", mod, i, None)
                results["god_class"].append({
                    "node_id": node_ids["class"][i], "name": rec["name"], "node_type": "class",
                    "file": filename, "class_name": rec["name"],
                    "line_start": rec["lineno"], "line_end": rec["end_lineno"], "explanation": explanation,
                    "probability": gc[i], "predicted": gc[i] >= 0.5,
                })

        n_detected = sum(1 for task in results.values() for p in task if p["predicted"])

        return {
            "summary": {
                "filename": filename,
                "n_classes": len(class_recs), "n_methods": len(method_recs), "n_functions": len(function_recs),
                "n_nodes": len(graph_nodes), "n_edges": len(graph_edges), "n_detected": n_detected,
            },
            "graph": {"nodes": graph_nodes, "edges": graph_edges},
            **results,
        }
