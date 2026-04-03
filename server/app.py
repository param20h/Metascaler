"""
FastAPI server exposing the OpenEnv SQL Optimizer environment.

Endpoints:
  POST /reset        → Observation
  POST /step         → {observation, reward, done, info}
  GET  /state        → state dict
  GET  /tasks        → list of tasks + action schema
  GET  /grader       → grader score for last completed episode
  POST /baseline     → trigger baseline inference on all 3 tasks
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from env.environment import SQLOptimizerEnv
from env.models import Action, Observation, Reward
from env.tasks import TASKS

app = FastAPI(
    title="SQL Query Optimizer — OpenEnv",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to rewrite "
        "and optimise SQL queries across three difficulty levels."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful, per-process)
_env = SQLOptimizerEnv()


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class GraderResponse(BaseModel):
    task_id: Optional[int]
    grader_score: float
    cumulative_score: float
    done: bool


class TaskInfo(BaseModel):
    id: int
    name: str
    difficulty: str
    description: str
    action_schema: Dict[str, Any]


class BaselineResponse(BaseModel):
    task_results: Dict[str, float]
    message: str


def _parse_end_payload(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if not line.startswith("[END] "):
            continue
        payload_text = line[len("[END] ") :].strip()
        import json

        return json.loads(payload_text)
    raise ValueError("Could not find [END] payload in inference output")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

def _health_payload() -> Dict[str, str]:
    return {"status": "ok", "environment": "sql-query-optimizer", "version": "1.0.0"}


@app.get("/", summary="Health check")
def health() -> Dict[str, str]:
    return _health_payload()


@app.get("/web", include_in_schema=False)
@app.get("/web/", include_in_schema=False)
def web_health() -> Dict[str, str]:
    return _health_payload()


@app.post("/reset", response_model=Observation, summary="Start / restart an episode")
def reset(req: ResetRequest) -> Observation:
    """Reset the environment for a given task_id (1=easy, 2=medium, 3=hard)."""
    try:
        obs = _env.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step", response_model=StepResponse, summary="Submit an action")
def step(action: Action) -> StepResponse:
    """Advance the environment by submitting an Action."""
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", summary="Return current internal state")
def state() -> Dict[str, Any]:
    """Return the current internal state of the environment."""
    return _env.state()


@app.get("/tasks", response_model=list[TaskInfo], summary="List tasks + action schema")
def list_tasks() -> list[TaskInfo]:
    """Return all tasks with descriptions and the action schema."""
    action_schema = Action.model_json_schema()
    return [
        TaskInfo(
            id=t.id,
            name=t.name,
            difficulty=t.difficulty,
            description=t.description,
            action_schema=action_schema,
        )
        for t in TASKS.values()
    ]


@app.get("/grader", response_model=GraderResponse, summary="Grader score for last episode")
def grader() -> GraderResponse:
    """Return the grader score after the current/last episode."""
    s = _env.state()
    if s.get("status") == "not_started":
        raise HTTPException(status_code=400, detail="No episode started. Call /reset first.")
    return GraderResponse(
        task_id=s.get("task_id"),
        grader_score=s.get("last_grader_score", 0.0),
        cumulative_score=s.get("cumulative_score", 0.0),
        done=s.get("done", False),
    )


@app.post("/baseline", response_model=BaselineResponse, summary="Run baseline inference on all tasks")
def baseline() -> BaselineResponse:
    """
    Trigger the baseline inference script (inference.py) and return scores.
    Requires API_BASE_URL, MODEL_NAME, and HF_TOKEN to be set in the environment.
    """
    required_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required environment variables: {', '.join(missing)}",
        )
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Inference script failed:\n{result.stderr}",
            )
        payload = _parse_end_payload(result.stdout)
        return BaselineResponse(
            task_results=payload.get("task_results", {}),
            message="Baseline completed successfully.",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Inference script timed out after 1200s.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
