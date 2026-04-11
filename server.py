"""
OpenEnv HTTP Server v2.2 — Disaster Response Coordination System
================================================================
FastAPI server — OpenEnv spec compliant, HF Spaces ready.

FIXED in v2.2:
  - /health endpoint registered first so it's immediately available
  - Startup errors are caught and logged, server still starts
  - All routes explicitly ordered: health > meta > openenv > viz
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel as _PB

# ── App created FIRST — before any potentially-failing imports ────────────────
app = FastAPI(
    title       = "Disaster Response Coordination System",
    description = "OpenEnv environment for emergency dispatch coordination.",
    version     = "2.2.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
_sessions: Dict[str, Any] = {}
_import_error: str = ""

# ── Register /health IMMEDIATELY so healthcheck never gets 404 ────────────────
@app.get("/health", tags=["meta"])
def health():
    """Liveness probe — always returns 200, even during startup."""
    return {"status": "ok", "active_sessions": len(_sessions), "error": _import_error}

# ── Now do the potentially-failing env imports ────────────────────────────────
try:
    from env.environment import DisasterResponseEnv, VALID_TASKS
    from env.models import ActionWrapper
    from env.world import build_text_grid, build_html_view
    _env_ready = True
except Exception as e:
    _env_ready     = False
    _import_error  = str(e)
    VALID_TASKS    = ["task1_prioritization", "task2_resource_allocation", "task3_dynamic_coordination"]

    # Stub classes so routes can be registered without crashing
    class DisasterResponseEnv:  # type: ignore
        def __init__(self, *a, **kw): raise RuntimeError(f"Env not loaded: {_import_error}")

    class ActionWrapper:  # type: ignore
        def __init__(self, **kw): self.__dict__.update(kw)

    def build_text_grid(*a, **kw): return "env not loaded"
    def build_html_view(*a, **kw): return "<html><body>env not loaded</body></html>"


# ── Request schemas ───────────────────────────────────────────────────────────
class ResetRequest(_PB):
    task_id: str = "task1_prioritization"
    seed:    int = 42

class StepRequest(_PB):
    session_id: str
    action:     Dict[str, Any]


# ── Meta endpoints ────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
def root():
    return {
        "name":        "Disaster Response Coordination System",
        "version":     "2.2.0",
        "env_ready":   _env_ready,
        "valid_tasks": VALID_TASKS,
        "docs":        "/docs",
    }

@app.get("/tasks", tags=["meta"])
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task1_prioritization",
                "name": "Incident Prioritization",
                "difficulty": "easy",
                "max_steps": 10,
                "primary_action": "reprioritize",
                "baseline_score": 1.00,
            },
            {
                "id": "task2_resource_allocation",
                "name": "Resource Allocation with Travel Delays",
                "difficulty": "medium",
                "max_steps": 20,
                "primary_action": "assign_resource",
                "baseline_score": 0.61,
            },
            {
                "id": "task3_dynamic_coordination",
                "name": "Dynamic Multi-step Coordination",
                "difficulty": "hard",
                "max_steps": 30,
                "primary_action": "assign_resource",
                "baseline_score": 0.27,
            },
        ]
    }


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────
@app.post("/reset", tags=["openenv"])
def reset(req: ResetRequest):
    if not _env_ready:
        raise HTTPException(503, f"Environment not ready: {_import_error}")
    if req.task_id not in VALID_TASKS:
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'. Valid: {VALID_TASKS}")
    sid = str(uuid.uuid4())
    env = DisasterResponseEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[sid] = env
    return {
        "session_id":  sid,
        "task_id":     req.task_id,
        "observation": obs.model_dump(),
    }

@app.post("/step", tags=["openenv"])
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call POST /reset first.")
    try:
        action              = ActionWrapper(**req.action)
        obs, reward, done, info = env.step(action)
    except Exception as exc:
        raise HTTPException(422, f"Action error: {exc}")
    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }

@app.get("/state/{session_id}", tags=["openenv"])
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    return env.state()

@app.get("/grade/{session_id}", tags=["openenv"])
def grade(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    result      = env.grade()
    description = env.grade_description()
    return {
        "session_id":   session_id,
        "result":       result.model_dump(),
        "grader_score": result.grader_score,
        "description":  description,
    }

@app.delete("/session/{session_id}", tags=["openenv"])
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(404, "Session not found.")

@app.get("/render/{session_id}", tags=["agent"])
def render(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    return {"text": env.render()}

@app.get("/metrics", tags=["meta"])
def metrics():
    return {"active_sessions": len(_sessions), "env_ready": _env_ready}


# ── Visualization ─────────────────────────────────────────────────────────────
@app.get("/visualize/{session_id}", tags=["visualization"], response_class=PlainTextResponse)
def visualize(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    if env._obs is None:
        raise HTTPException(400, "No observation yet — call /reset first.")
    obs = env._obs
    grid = build_text_grid(obs.incidents, obs.resources)
    header = (
        f"DISASTER MAP: {obs.task_id} | Step {obs.timestep}/{obs.max_timesteps}\n"
        f"Active:{obs.active_count} Resolved:{obs.resolved_count} "
        f"Failed:{obs.failed_count} Lives:{obs.total_lives_saved}\n"
    )
    legend = "\nLegend: 🔥Fire 🏥Medical 🌊Flood ⚡Quake 💥Accident ☣️Hazmat | 🚑🚒🛟🚓🚁\n"
    return header + grid + legend

@app.get("/view/{session_id}", tags=["visualization"], response_class=HTMLResponse)
def view(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    if env._obs is None:
        raise HTTPException(400, "No observation yet — call /reset first.")
    obs = env._obs
    return build_html_view(obs.incidents, obs.resources, obs.timestep, obs.task_id)


# ── Schema ────────────────────────────────────────────────────────────────────
@app.get("/schema/observation", tags=["schema"])
def obs_schema():
    if not _env_ready:
        raise HTTPException(503, "Environment not ready")
    return DisasterResponseEnv().observation_schema()

@app.get("/schema/action", tags=["schema"])
def action_schema():
    if not _env_ready:
        raise HTTPException(503, "Environment not ready")
    return DisasterResponseEnv().action_schema()

@app.get("/schema/reward", tags=["schema"])
def reward_schema():
    if not _env_ready:
        raise HTTPException(503, "Environment not ready")
    return DisasterResponseEnv().reward_schema()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
