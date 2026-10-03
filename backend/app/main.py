# -*- coding: utf-8 -*-
"""FastAPI service exposing the final selected model (Design B tuned) for
single-file Python code smell prediction. Serving only -- no training,
no metric logging beyond what's already in docs/final_test_evaluation_report.md.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import ModelBundle
from .schemas import AnalyzeRequest, AnalyzeResponse

_bundle: ModelBundle | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _bundle
    _bundle = ModelBundle()
    yield


app = FastAPI(title="Code Smell Detector API", version="0.1.0", lifespan=_lifespan)

# Local demo tool, no auth/cookies -- open CORS so the Vite dev server (or a
# static build served from anywhere) can call the API directly.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        return _bundle.predict(req.source, req.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
