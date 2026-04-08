"""
OpenEnv HTTP Server v2.1 — Disaster Response Coordination System
================================================================
FastAPI server implementing the full OpenEnv REST interface.
Compatible with Hugging Face Spaces (port 7860, non-root user).

Core OpenEnv endpoints:
  POST /reset                → start episode, return observation
  POST /step                 → execute action, return (obs, reward, done, info)
  GET  /state/{session_id}   → full internal state dict
  GET  /grade/{session_id}   → deterministic grader score [0, 1]

Additional endpoints:
  GET  /health               → liveness probe (200 OK)
  GET  /tasks                → list all task definitions
  GET  /render/{session_id}  → rich text state for LLM agents
  GET  /visualize/{session_id} → 10×10 text emoji grid
  GET  /view/{session_id}    → HTML live map (auto-refresh)
  GET  /schema/observation   → JSON schema
  GET  /schema/action        → JSON schema
  GET  /schema/reward        → JSON schema
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel as _PB

from env.environment import DisasterResponseEnv, VALID_TASKS
from env.models import ActionWrapper
from env.world import build_text_grid, build_html_view

# ── App ───────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import requests

app = FastAPI()

@app.get("/")
def root():
    try:
        res = requests.post(
            "http://localhost:7860/reset",
            json={"task_id": "task3_dynamic_coordination", "seed": 42}
        )
        session_id = res.json().get("session_id")
        return RedirectResponse(url=f"/view/{session_id}")
    except:
        return {"status": "running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id → DisasterResponseEnv
_sessions: Dict[str, DisasterResponseEnv] = {}


# ── Request bodies ────────────────────────────────────────────────────────────

class ResetRequest(_PB):
    task_id: str = "task1_prioritization"
    seed:    int = 42


class StepRequest(_PB):
    session_id: str
    action:     Dict[str, Any]

@app.get("/health", tags=["meta"])
def health():
    """Liveness probe. Returns 200 + {status: ok} when server is ready."""
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.get("/tasks", tags=["meta"])
def list_tasks():
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id":            "task1_prioritization",
                "name":          "Incident Prioritization",
                "difficulty":    "easy",
                "max_steps":     10,
                "primary_action": "reprioritize",
                "description":   (
                    "Order 6 incidents by urgency each step. "
                    "Graded via Kendall-tau + fairness score."
                ),
                "baseline_score": 1.00,
            },
            {
                "id":            "task2_resource_allocation",
                "name":          "Resource Allocation with Travel Delays",
                "difficulty":    "medium",
                "max_steps":     20,
                "primary_action": "assign_resource",
                "description":   (
                    "Assign 6 typed resources to 5 incidents with real travel delays. "
                    "Resources in transit before arriving."
                ),
                "baseline_score": 0.61,
            },
            {
                "id":            "task3_dynamic_coordination",
                "name":          "Dynamic Multi-step Coordination",
                "difficulty":    "hard",
                "max_steps":     30,
                "primary_action": "assign_resource",
                "description":   (
                    "Dynamic: incidents escalate, new ones spawn (p=0.28/step), "
                    "fires cascade-spread. Continuously optimize 6 resources."
                ),
                "baseline_score": 0.27,
            },
        ]
    }


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset", tags=["openenv"])
def reset(req: ResetRequest):
    """
    Start a new episode.
    Returns session_id + initial observation (OpenEnv spec).
    """
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
    """
    Execute one action step.
    Returns observation, reward, done, info (OpenEnv spec).
    """
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call POST /reset first.")
    try:
        action          = ActionWrapper(**req.action)
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
    """Return full serialized internal environment state (OpenEnv spec)."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    return env.state()


@app.get("/grade/{session_id}", tags=["openenv"])
def grade(session_id: str):
    """
    Run the deterministic grader.
    Returns grader_score in [0.0, 1.0] with full breakdown (OpenEnv spec).
    """
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
    """Clean up a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(404, "Session not found.")


# ── Agent-facing endpoints ────────────────────────────────────────────────────

@app.get("/render/{session_id}", tags=["agent"])
def render(session_id: str):
    """Rich text state for LLM agent consumption (Unicode/emoji grid + tables)."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    return {"text": env.render()}


# ── Visualization endpoints ───────────────────────────────────────────────────

@app.get(
    "/visualize/{session_id}",
    tags=["visualization"],
    response_class=PlainTextResponse,
)
def visualize(session_id: str):
    """
    10×10 text emoji grid of the disaster scene.

    Legend:
      🔥 🏥 🌊 ⚡ 💥 ☣️  = active incidents (by type)
      🚑 🚒 🛟 🚓 🚁     = resources
      [ ]                = empty cell

    Y axis: top = north (y=100), bottom = south (y=0).
    """
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    if env._obs is None:
        raise HTTPException(400, "No observation yet — call /reset first.")

    obs       = env._obs
    grid_text = build_text_grid(obs.incidents, obs.resources)
    header = (
        f"╔══ DISASTER MAP: {obs.task_id} │ Step {obs.timestep}/{obs.max_timesteps} ══╗\n"
        f"Active:{obs.active_count}  Resolved:{obs.resolved_count}  "
        f"Failed:{obs.failed_count}  Lives:{obs.total_lives_saved}\n"
        f"Fairness Gini:{obs.fairness.gini_coefficient:.3f}  "
        f"Urban:{obs.fairness.urban_response_rate:.0%}  "
        f"Rural:{obs.fairness.rural_response_rate:.0%}\n"
    )
    legend = (
        "\nLegend: 🔥Fire 🏥Medical 🌊Flood ⚡Quake 💥Accident ☣️Hazmat | "
        "🚑Ambulance 🚒FireTruck 🛟Rescue 🚓Police 🚁Helicopter\n"
        f"╚{'═'*58}╝"
    )
    return header + grid_text + legend


@app.get(
    "/view/{session_id}",
    tags=["visualization"],
    response_class=HTMLResponse,
)
def view(session_id: str):
    """
    Self-contained HTML live map. Auto-refreshes every 5 seconds.
    Colour-coded cells: incident type colour / green=free / orange=transit / red=busy.
    Open in browser while running inference to watch the episode unfold.
    """
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    if env._obs is None:
        raise HTTPException(400, "No observation yet — call /reset first.")

    obs = env._obs
    return build_html_view(
        incidents = obs.incidents,
        resources = obs.resources,
        timestep  = obs.timestep,
        task_id   = obs.task_id,
    )


# ── JSON Schema endpoints ─────────────────────────────────────────────────────

@app.get("/schema/observation", tags=["schema"])
def obs_schema():
    return DisasterResponseEnv().observation_schema()


@app.get("/schema/action", tags=["schema"])
def action_schema():
    return DisasterResponseEnv().action_schema()


@app.get("/schema/reward", tags=["schema"])
def reward_schema():
    return DisasterResponseEnv().reward_schema()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
