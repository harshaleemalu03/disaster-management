"""
OpenEnv HTTP Server v2.1 — Disaster Response Coordination System
================================================================
FastAPI server implementing the full OpenEnv REST interface.
Compatible with Hugging Face Spaces (port 7860, non-root user).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel as _PB

from env.environment import DisasterResponseEnv, VALID_TASKS
from env.models import ActionWrapper
from env.world import build_text_grid, build_html_view

# ── App Initialization ────────────────────────────────────────────────────────

app = FastAPI(title="OpenEnv Disaster Response Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
_sessions: Dict[str, DisasterResponseEnv] = {}

# ── Root Endpoint (SAFE) ──────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "🚀 OpenEnv Disaster Response Server is running",
        "docs": "/docs",
        "health": "/health"
    }

# ── Request Models ────────────────────────────────────────────────────────────

class ResetRequest(_PB):
    task_id: str = "task1_prioritization"
    seed: int = 42


class StepRequest(_PB):
    session_id: str
    action: Dict[str, Any]

# ── Meta Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.get("/tasks", tags=["meta"])
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task1_prioritization",
                "name": "Incident Prioritization",
                "difficulty": "easy",
                "max_steps": 10,
            },
            {
                "id": "task2_resource_allocation",
                "name": "Resource Allocation",
                "difficulty": "medium",
                "max_steps": 20,
            },
            {
                "id": "task3_dynamic_coordination",
                "name": "Dynamic Coordination",
                "difficulty": "hard",
                "max_steps": 30,
            },
        ]
    }

# ── Core OpenEnv Endpoints ────────────────────────────────────────────────────

@app.post("/reset", tags=["openenv"])
def reset(req: ResetRequest):
    if req.task_id not in VALID_TASKS:
        raise HTTPException(400, f"Invalid task_id. Choose from {VALID_TASKS}")

    session_id = str(uuid.uuid4())
    env = DisasterResponseEnv(task_id=req.task_id, seed=req.seed)

    obs = env.reset()
    _sessions[session_id] = env

    return {
        "session_id": session_id,
        "task_id": req.task_id,
        "observation": obs.model_dump(),
    }


@app.post("/step", tags=["openenv"])
def step(req: StepRequest):
    env = _sessions.get(req.session_id)

    if not env:
        raise HTTPException(404, "Session not found. Call /reset first.")

    try:
        action = ActionWrapper(**req.action)
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(422, f"Invalid action: {str(e)}")

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state/{session_id}", tags=["openenv"])
def state(session_id: str):
    env = _sessions.get(session_id)

    if not env:
        raise HTTPException(404, "Session not found.")

    return env.state()


@app.get("/grade/{session_id}", tags=["openenv"])
def grade(session_id: str):
    env = _sessions.get(session_id)

    if not env:
        raise HTTPException(404, "Session not found.")

    result = env.grade()

    return {
        "session_id": session_id,
        "grader_score": result.grader_score,
        "details": result.model_dump(),
    }


@app.delete("/session/{session_id}", tags=["openenv"])
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": session_id}

    raise HTTPException(404, "Session not found.")

# ── Agent Endpoint ────────────────────────────────────────────────────────────

@app.get("/render/{session_id}", tags=["agent"])
def render(session_id: str):
    env = _sessions.get(session_id)

    if not env:
        raise HTTPException(404, "Session not found.")

    return {"text": env.render()}

# ── Visualization ─────────────────────────────────────────────────────────────

@app.get("/visualize/{session_id}", response_class=PlainTextResponse)
def visualize(session_id: str):
    env = _sessions.get(session_id)

    if not env or env._obs is None:
        raise HTTPException(404, "Session not ready.")

    obs = env._obs

    grid = build_text_grid(obs.incidents, obs.resources)

    return f"""
Step: {obs.timestep}/{obs.max_timesteps}
Active: {obs.active_count} | Resolved: {obs.resolved_count}

{grid}
"""


@app.get("/view/{session_id}", response_class=HTMLResponse)
def view(session_id: str):
    env = _sessions.get(session_id)

    if not env or env._obs is None:
        raise HTTPException(404, "Session not ready.")

    obs = env._obs

    return build_html_view(
        incidents=obs.incidents,
        resources=obs.resources,
        timestep=obs.timestep,
        task_id=obs.task_id,
    )

# ── Schema Endpoints ──────────────────────────────────────────────────────────

@app.get("/schema/observation")
def observation_schema():
    return DisasterResponseEnv().observation_schema()


@app.get("/schema/action")
def action_schema():
    return DisasterResponseEnv().action_schema()


@app.get("/schema/reward")
def reward_schema():
    return DisasterResponseEnv().reward_schema()

# ── Run (Local Only) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)