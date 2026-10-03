# -*- coding: utf-8 -*-
"""Request/response models for the /analyze endpoint.

Phase 17 extends the Phase 16 response with the actual graph the backend
already builds (previously computed and discarded) and per-prediction
identity/location fields, while keeping every original field
(name/node_type/probability/predicted) unchanged for backward compatibility."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    source: str
    filename: str = "input.py"


class GraphNode(BaseModel):
    id: str
    type: str
    name: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class Graph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class Prediction(BaseModel):
    node_id: str
    name: str
    node_type: str
    probability: float
    predicted: bool
    file: str
    class_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    explanation: str


class AnalysisSummary(BaseModel):
    filename: str
    n_classes: int
    n_methods: int
    n_functions: int
    n_nodes: int
    n_edges: int
    n_detected: int


class AnalyzeResponse(BaseModel):
    summary: AnalysisSummary
    graph: Graph
    long_method: List[Prediction]
    feature_envy: List[Prediction]
    god_class: List[Prediction]
