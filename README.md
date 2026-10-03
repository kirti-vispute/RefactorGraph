# RefactorGraph

RefactorGraph analyzes Python source code and detects three classic code
smells — **Long Method**, **Large/God Class**, and **Feature Envy** — by
combining a class's/method's *structure* (via a code graph) with its
*semantics* (via CodeBERT embeddings), and recommends a matching
refactoring technique for each detected smell.

## Why

Code that works correctly can still be hard to maintain, extend, or
understand. Long methods, bloated classes, and misplaced logic (a method
that leans more on another class's data than its own) are well-known
signs of this. RefactorGraph detects these three patterns automatically
and points at the standard refactoring technique for each.

## Target Smells

| Smell | Recommended Refactoring |
|---|---|
| Long Method | Extract Method |
| Feature Envy | Move Method |
| God Class | Split Class |

## How It Works

Python source → **AST parsing**
(`ml/preprocessing/ast_parser.py`) → a **heterogeneous code graph**
(module / class / method / function / attribute / parameter / import
nodes, 14 relation types — `ml/graph/graph_builder.py`) → per-node
**structural features** (LOC, statement counts, method/field counts,
self- vs. external-access ratios) fused with **frozen CodeBERT
embeddings** of each node's own source snippet (`microsoft/codebert-base`,
never fine-tuned) → a **2-layer HeteroGAT** (`ml/models/gat_baseline.py`)
→ three independent prediction heads, one per smell. The God Class head
uses **Design B**: it reads a class's own embedding concatenated with a
mean-pool of its methods' embeddings (`class_method_pool=True`), giving
it visibility into its methods that the plain message-passing schema
doesn't otherwise provide.

### Live Analysis (what `/analyze` actually does)

```
Source Code → AST/Graph Construction → Model Inference → Smell Predictions → Refactoring Recommendations
```

Each prediction includes a **Model Score** — the model's raw sigmoid
output, not a calibrated probability or an accuracy figure — and a
structural-metrics explanation (the node's own LOC/statement/access
counts), returned alongside every prediction.

### Offline Explainability (GNNExplainer)

GNNExplainer (`torch_geometric.explain`) is used **separately, offline**,
to analyze model behavior on the validation set and identify which graph
nodes/edges and structural features contributed most to specific
predictions — see `docs/explainability_report_candidate.md`. **It is not
invoked by the live `/analyze` endpoint** — the live explanation field is
the structural-metrics summary described above, not a GNNExplainer
attribution. PGExplainer was considered and deliberately not used (see
`scripts/explain.py`'s module docstring for why).

## Architecture

- **Backend**: FastAPI (`backend/`) — loads the frozen model checkpoint
  once at startup, exposes `/health` and `/analyze`.
- **Frontend**: React + Tailwind CSS + Cytoscape.js (`frontend/`) —
  submits source code, renders the returned code graph interactively,
  and displays per-node smell predictions and refactoring
  recommendations.
- **ML pipeline**: `ml/` (AST parsing, graph construction, model),
  `scripts/` (dataset build, training, hyperparameter search,
  evaluation, promotion — see individual script docstrings for the
  project's phase-by-phase history).

## Results

Repository-level train/val/test split (9/3/3 repos, zero overlap), all
labels rule-derived ("silver") from real, cloned open-source repositories
— see `docs/dataset_report.md` and `docs/godclass_formula_revision.md`
for the full methodology.

**Final corrected TEST evaluation of the deployed seed-43 model**
(`docs/final_test_evaluation_AB_report.md`):

| Task | F1 |
|---|---|
| Long Method | 0.7740 |
| Feature Envy | 0.5272 |
| God Class | 0.8075 |
| **Macro-F1** | **0.7029** |

The historical production result was Long Method 0.740, Feature Envy
0.525, God Class 0.767, and Macro-F1 0.677, measured on the earlier
TEST representation (`docs/final_test_evaluation_report.md`). It is
preserved for reference and is not directly comparable to the corrected
evaluation above. The current deployed checkpoint is
`models/experiment_fe_dominant/seed_43/`; live Feature Envy predictions
use a 0.65 decision threshold, while the reported offline TEST metrics
used 0.50 for all three smells. The TEST split is closed. Evaluation
methodology and deployment status are recorded in
`docs/final_test_evaluation_AB_report.md` and `docs/PROJECT_FREEZE_REPORT.md`.

## Running It

Requires Python 3.13+ and Node.js.

```bash
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

The frontend (default `http://localhost:5173`) talks to the backend at
`http://localhost:8000`.

## Testing

```bash
pytest tests/                 # backend/ML — 94 tests
cd frontend && npx vitest run # frontend — 25 tests
```

## What This Project Does Not Do

- It does not automatically rewrite or refactor source code — it
  recommends a refactoring technique per detected smell.
- Model Score is not a calibrated probability and not an accuracy
  figure — no reliability/calibration study backs it.
- It does not integrate PyExamine as a component (its known false-positive
  pattern was used only as a reference while fixing this project's own
  metric code).
- It does not use PGExplainer.
- It does not run GNNExplainer as part of a live request.
