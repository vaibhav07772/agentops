"""AgentOps — FastAPI Server"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.orchestrator.graph_v4 import AgentOpsV4
from src.mlops.tracker import get_experiment_summary

# ─────────────────────────────────────
# App
# ─────────────────────────────────────
app = FastAPI(
    title="AgentOps API",
    description="Multi-Agent System with RAG, Guardrails & MLOps",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init AgentOps (singleton)
ops = None


def get_ops():
    global ops
    if ops is None:
        ops = AgentOpsV4(max_retries=1, enable_tracking=True)
    return ops


# ─────────────────────────────────────
# Models
# ─────────────────────────────────────
class QueryRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=5000, description="User task or question")
    max_retries: int = Field(default=1, ge=0, le=3)


class QueryResponse(BaseModel):
    task: str
    route: str
    blocked: bool
    block_reason: Optional[str] = None
    output: str
    latency: float
    total_tokens: int
    quality: Optional[float] = None
    retries: int
    output_valid: bool
    citations: list = []
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


# ─────────────────────────────────────
# Routes
# ─────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """API root"""
    return {
        "name": "AgentOps API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0",
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Run AgentOps on a task.

    Returns route, output, metrics, and validation.
    """
    try:
        agent_ops = get_ops()
        result = agent_ops.run(request.task)

        ov = result.get("output_validation", {})

        return QueryResponse(
            task=request.task,
            route=result.get("route", "unknown"),
            blocked=result.get("blocked", False),
            block_reason=result.get("block_reason") or None,
            output=result.get("final_output", ""),
            latency=result.get("latency", 0.0),
            total_tokens=result.get("total_tokens", 0),
            quality=result.get("quality"),
            retries=result.get("retries", 0),
            output_valid=ov.get("valid", False),
            citations=ov.get("citations", []),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Info"])
def stats():
    """Get experiment stats from MLflow"""
    try:
        return get_experiment_summary()
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────
# Run
# ─────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )