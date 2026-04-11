"""
OpenEnv HTTP Server v2.3 — Disaster Response Coordination System
================================================================
FastAPI server — OpenEnv spec compliant, HF Spaces ready.

KEY FIXES in v2.3:
  - POST /reset accepts empty/null body (ResetRequest fully optional)
  - /health registered FIRST — always returns 200 regardless of import state
  - Beautiful HTML dashboard at GET /view/{session_id}
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# ── App created FIRST — before any env imports ────────────────────────────────
app = FastAPI(
    title       = "Disaster Response Coordination System",
    description = "OpenEnv environment for emergency dispatch coordination.",
    version     = "2.3.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

_sessions: Dict[str, Any] = {}
_import_error: str         = ""
_env_ready:    bool        = False

# ── /health FIRST — always 200, even before env loads ────────────────────────
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "env_ready": _env_ready, "sessions": len(_sessions)}

# ── Env imports (potentially slow — must come AFTER /health registration) ─────
try:
    from env.environment import DisasterResponseEnv, VALID_TASKS
    from env.models import ActionWrapper
    from env.world import build_text_grid, build_html_view
    _env_ready = True
except Exception as _e:
    _import_error = str(_e)
    VALID_TASKS   = ["task1_prioritization", "task2_resource_allocation", "task3_dynamic_coordination"]

    class DisasterResponseEnv:  # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError(f"Env failed to load: {_import_error}")

    class ActionWrapper:  # type: ignore
        def __init__(self, **kw): self.__dict__.update(kw)

    def build_text_grid(*a, **kw) -> str:
        return "env not loaded"

    def build_html_view(*a, **kw) -> str:
        return "<html><body style='background:#0d1117;color:#e6edf3;font-family:sans-serif;padding:2rem'><h1>⚠️ Environment not loaded</h1></body></html>"


# ── Meta endpoints ────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {
        "name":        "Disaster Response Coordination System",
        "version":     "2.3.0",
        "env_ready":   _env_ready,
        "valid_tasks": VALID_TASKS,
        "endpoints": {
            "reset":     "POST /reset",
            "step":      "POST /step",
            "state":     "GET  /state/{session_id}",
            "grade":     "GET  /grade/{session_id}",
            "render":    "GET  /render/{session_id}",
            "visualize": "GET  /visualize/{session_id}  (text grid)",
            "view":      "GET  /view/{session_id}       (HTML dashboard)",
            "tasks":     "GET  /tasks",
            "health":    "GET  /health",
        },
        "docs": "/docs",
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
async def reset(request: Request):
    """
    Start a new episode. Body is FULLY OPTIONAL — works with empty/null body.
    Defaults: task_id='task1_prioritization', seed=42.
    """
    if not _env_ready:
        raise HTTPException(503, f"Environment not ready: {_import_error}")

    # Parse body safely — empty / null / missing body all default gracefully
    task_id = "task1_prioritization"
    seed    = 42
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = body.get("task_id", task_id)
            seed    = int(body.get("seed", seed))
    except Exception:
        pass   # empty body → use defaults

    if task_id not in VALID_TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")

    sid = str(uuid.uuid4())
    env = DisasterResponseEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    _sessions[sid] = env

    return {
        "session_id":  sid,
        "task_id":     task_id,
        "observation": obs.model_dump(),
    }


@app.post("/step", tags=["openenv"])
async def step(request: Request):
    """Execute one action step. Returns observation, reward, done, info."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(422, "Request body must be valid JSON with session_id and action.")

    session_id = body.get("session_id")
    action_dict = body.get("action")
    if not session_id or action_dict is None:
        raise HTTPException(422, "Body must contain 'session_id' and 'action'.")

    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call POST /reset first.")

    try:
        action              = ActionWrapper(**action_dict)
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
    """Rich text state for LLM agent consumption."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    return {"text": env.render()}


@app.get("/metrics", tags=["meta"])
def metrics():
    return {"active_sessions": len(_sessions), "env_ready": _env_ready}


# ── Visualization endpoints ───────────────────────────────────────────────────

@app.get("/visualize/{session_id}", tags=["visualization"], response_class=PlainTextResponse)
def visualize(session_id: str):
    """
    10×10 text emoji grid of the disaster scene.
    🔥🏥🌊⚡💥☣️ = incidents | 🚑🚒🛟🚓🚁 = resources
    """
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    if env._obs is None:
        raise HTTPException(400, "No observation yet — call /reset first.")
    obs  = env._obs
    grid = build_text_grid(obs.incidents, obs.resources)
    header = (
        f"DISASTER MAP: {obs.task_id} | Step {obs.timestep}/{obs.max_timesteps}\n"
        f"Active:{obs.active_count}  Resolved:{obs.resolved_count}  "
        f"Failed:{obs.failed_count}  Lives:{obs.total_lives_saved}\n"
        f"Fairness Gini:{obs.fairness.gini_coefficient:.3f}  "
        f"Urban:{obs.fairness.urban_response_rate:.0%}  "
        f"Rural:{obs.fairness.rural_response_rate:.0%}\n\n"
    )
    legend = (
        "\nLegend: 🔥Fire 🏥Medical 🌊Flood ⚡Quake 💥Accident ☣️Hazmat | "
        "🚑Ambulance 🚒FireTruck 🛟Rescue 🚓Police 🚁Helicopter"
    )
    return header + grid + legend


@app.get("/view/{session_id}", tags=["visualization"], response_class=HTMLResponse)
def view(session_id: str):
    """
    Beautiful real-time HTML dashboard with:
    - 12×12 grid map with pulsing high-severity incidents
    - Incident table with inline severity bars
    - Resource status table (FREE / IN TRANSIT / BUSY)
    - Fairness bars (Urban/Suburban/Rural response rates + Gini)
    - Cascade fire events log
    - Step action history
    - Auto-refreshes every 3 seconds
    """
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    if env._obs is None:
        raise HTTPException(400, "No observation — call /reset first.")
    obs = env._obs
    return build_html_view(
        incidents         = obs.incidents,
        resources         = obs.resources,
        timestep          = obs.timestep,
        task_id           = obs.task_id,
        max_timesteps     = obs.max_timesteps,
        fairness          = obs.fairness,
        step_logs         = obs.step_log,
        cascade_events    = obs.cascade_events,
        resolved_count    = obs.resolved_count,
        failed_count      = obs.failed_count,
        active_count      = obs.active_count,
        lives_saved       = obs.total_lives_saved,
        cumulative_reward = obs.cumulative_reward,
    )


# ── JSON Schema endpoints ─────────────────────────────────────────────────────

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
