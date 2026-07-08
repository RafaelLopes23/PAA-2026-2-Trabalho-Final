"""FastAPI server for the project demonstration."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import HealthResponse, MethodsResponse, QueryRequest, QueryResponse
from src.pipeline import MethodUnavailableError, PipelineError, QueryPipeline, build_default_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = build_default_pipeline()
    yield


app = FastAPI(
    title="PAA Movie QA",
    version="0.1.0",
    description="Movie question-answering system with semantic search and TinyLlama.",
    lifespan=lifespan,
)


def _get_pipeline(request: Request) -> QueryPipeline:
    return request.app.state.pipeline


from fastapi.responses import HTMLResponse


@app.get("/", response_model=MethodsResponse)
def root(request: Request) -> HTMLResponse | MethodsResponse:
    # Serve the graphical interface for browser requests.
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        static_index = os.path.join(os.path.dirname(__file__), "static/index.html")
        if os.path.exists(static_index):
            with open(static_index, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

    pipeline = _get_pipeline(request)
    return MethodsResponse(
        available_methods=pipeline.available_methods(),
        methods=pipeline.methods_payload(),
    )


@app.get("/gui", response_class=HTMLResponse)
def gui() -> HTMLResponse:
    """Serve the semantic search graphical interface."""
    static_index = os.path.join(os.path.dirname(__file__), "static/index.html")
    if not os.path.exists(static_index):
        raise HTTPException(status_code=404, detail="Graphical interface not found.")
    with open(static_index, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline = _get_pipeline(request)
    status = "ready" if pipeline.available_methods() else "degraded"
    return HealthResponse(
        status=status,
        movies_count=pipeline.movies_count,
        available_methods=pipeline.available_methods(),
        methods=pipeline.methods_payload(),
    )


@app.get("/methods", response_model=MethodsResponse)
def methods(request: Request) -> MethodsResponse:
    pipeline = _get_pipeline(request)
    return MethodsResponse(
        available_methods=pipeline.available_methods(),
        methods=pipeline.methods_payload(),
    )


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    pipeline = _get_pipeline(request)
    try:
        result = pipeline.query(
            question=payload.question,
            method=payload.method,
            top_k=payload.top_k,
        )
        return QueryResponse(**result)
    except MethodUnavailableError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": str(exc),
                "available_methods": pipeline.available_methods(),
                "methods": pipeline.methods_payload(),
            },
        ) from exc
    except PipelineError as exc:
        raise HTTPException(status_code=503, detail={"message": str(exc)}) from exc


if __name__ == "__main__":
    host = os.getenv("PAA_API_HOST", "127.0.0.1")
    port = int(os.getenv("PAA_API_PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)
