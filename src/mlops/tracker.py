"""AgentOps — MLflow Experiment Tracking"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager

import mlflow
from rich.console import Console

console = Console()

# ─────────────────────────────────────
# Config
# ─────────────────────────────────────
MLFLOW_DIR = "data/mlflow"
MLFLOW_TRACKING_URI = f"file:///{Path(MLFLOW_DIR).absolute()}"
EXPERIMENT_NAME = "agentops"

Path(MLFLOW_DIR).mkdir(parents=True, exist_ok=True)


class AgentOpsTracker:
    """MLflow tracker for agent runs"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        self.current_run = None
        console.print(f"[green]✅ MLflow tracking at {MLFLOW_TRACKING_URI}[/green]")

    @contextmanager
    def start_run(self, task: str, route: str):
        """Context manager for tracking a run"""
        run_name = f"{route}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            self.current_run = run

            # Log task metadata
            mlflow.log_param("task", task[:250])
            mlflow.log_param("route", route)
            mlflow.log_param("timestamp", datetime.now().isoformat())

            start_time = time.time()

            try:
                yield run
            finally:
                # Log context manager's internal duration separately
                # (NOT the user's actual latency)
                tracking_duration = time.time() - start_time
                mlflow.log_metric("tracking_duration_seconds", tracking_duration)
                console.print(f"[dim]📊 Run logged (tracking: {tracking_duration:.3f}s)[/dim]")

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics"""
        if not self.current_run:
            return

        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            except Exception as e:
                console.print(f"[yellow]⚠ Failed to log {key}: {e}[/yellow]")

    def log_artifacts(self, artifacts: Dict[str, Any]):
        """Log artifacts (dict → JSON file)"""
        if not self.current_run:
            return

        artifact_dir = Path("data/experiments/_tmp_artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_file = artifact_dir / f"output_{datetime.now().strftime('%H%M%S')}.json"
        artifact_file.write_text(json.dumps(artifacts, indent=2, default=str), encoding="utf-8")

        try:
            mlflow.log_artifact(str(artifact_file))
        except Exception as e:
            console.print(f"[yellow]⚠ Artifact log failed: {e}[/yellow]")
        finally:
            try:
                artifact_file.unlink()
            except Exception:
                pass

    def log_run(
        self,
        task: str,
        route: str,
        total_tokens: int,
        latency: float,
        quality: Optional[float] = None,
        retries: int = 0,
        cost_usd: Optional[float] = None,
    ):
        """Log a complete run in one call"""
        with self.start_run(task=task, route=route):
            mlflow.log_metric("total_tokens", total_tokens)
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("retries", retries)

            if quality is not None:
                mlflow.log_metric("quality", quality)

            if cost_usd is not None:
                mlflow.log_metric("cost_usd", cost_usd)

            # Compute cost if not provided (Groq pricing estimate)
            if cost_usd is None:
                # GPT-OSS-120B on Groq: ~$0.15 per 1M tokens (approximate)
                estimated_cost = (total_tokens / 1_000_000) * 0.15
                mlflow.log_metric("cost_usd_est", estimated_cost)


# ─────────────────────────────────────
# Helper functions
# ─────────────────────────────────────
def get_experiment_summary() -> Dict:
    """Get summary of all runs"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if not experiment:
        return {"error": "No experiment found"}

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    if not runs:
        return {"total_runs": 0}

    total_tokens = sum(r.data.metrics.get("total_tokens", 0) for r in runs)
    avg_latency = sum(r.data.metrics.get("latency_seconds", 0) for r in runs) / len(runs)
    avg_quality = sum(r.data.metrics.get("quality", 0) for r in runs if r.data.metrics.get("quality")) 

    return {
        "total_runs": len(runs),
        "total_tokens": total_tokens,
        "avg_latency": avg_latency,
        "experiment_id": experiment.experiment_id,
    }


# ─────────────────────────────────────
# Test
# ─────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]AgentOps — MLflow Tracker Test[/bold magenta]")
    console.print("="*60)

    tracker = AgentOpsTracker()

    console.print("\n[cyan]📊 Logging 3 test runs...[/cyan]\n")

    tracker.log_run(
        task="What is the critic pattern?",
        route="rag",
        total_tokens=1125,
        latency=2.5,
        quality=0.85,
        retries=0,
    )

    tracker.log_run(
        task="Compare Python frameworks",
        route="multi_agent",
        total_tokens=16422,
        latency=45.3,
        quality=0.60,
        retries=1,
    )

    tracker.log_run(
        task="Explain RAG chunking",
        route="rag",
        total_tokens=1050,
        latency=2.1,
        quality=0.90,
        retries=0,
    )

    console.print("\n[bold]Experiment Summary:[/bold]")
    summary = get_experiment_summary()
    for key, value in summary.items():
        console.print(f"  {key}: {value}")

    console.print(f"\n[green]✅ View in MLflow UI:[/green]")
    console.print(f"[dim]  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}[/dim]")