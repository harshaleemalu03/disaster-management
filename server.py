"""
OpenEnv HTTP Server v2.1 — Disaster Response Coordination System
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel as _PB

from env.environment import DisasterResponseEnv, VALID_TASKS
from env.models import ActionWrapper
from env.world import build_text_grid

# ── App ─────────────────────────────────────────────────────

app = FastAPI(title="Disaster Response AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, DisasterResponseEnv] = {}

# ── ROOT ────────────────────────────────────────────────────

@app.get("/")
def root():
    env = DisasterResponseEnv(task_id="task3_dynamic_coordination", seed=42)
    env.reset()
    sid = str(uuid.uuid4())
    _sessions[sid] = env
    return RedirectResponse(url=f"/view/{sid}")

# ── MODELS ─────────────────────────────────────────────────

class ResetRequest(_PB):
    task_id: str = "task1_prioritization"
    seed: int = 42

class StepRequest(_PB):
    session_id: str
    action: Dict[str, Any]

# ── CORE ───────────────────────────────────────────────────

# ✅ FIXED: body is OPTIONAL
@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()  # default values

    if req.task_id not in VALID_TASKS:
        raise HTTPException(400, "Invalid task_id")

    sid = str(uuid.uuid4())
    env = DisasterResponseEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[sid] = env

    return {
        "session_id": sid,
        "task_id": req.task_id,
        "observation": obs.model_dump(),
    }

@app.post("/step")
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if not env:
        raise HTTPException(404, "Session not found")

    action = ActionWrapper(**req.action)
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }

@app.get("/state/{session_id}")
def state(session_id: str):
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(404, "Session not found")
    return env.state()

@app.get("/grade/{session_id}")
def grade(session_id: str):
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(404, "Session not found")

    result = env.grade()
    return {
        "grader_score": result.grader_score,
        "details": result.model_dump(),
    }

# ── UI ─────────────────────────────────────────────────────

@app.get("/view/{session_id}", response_class=HTMLResponse)
def view(session_id: str):
    env = _sessions.get(session_id)

    if not env or env._obs is None:
        raise HTTPException(404, "Session not ready")

    obs = env._obs
    grid = build_text_grid(obs.incidents, obs.resources)

    return f"""
    <html>
    <head>
        <title>Disaster AI Dashboard</title>
        <meta http-equiv="refresh" content="3">
        <style>
            body {{
                margin: 0;
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: white;
            }}
            .container {{
                display: flex;
                height: 100vh;
            }}
            .left {{
                width: 70%;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .grid {{
                background: #020617;
                padding: 25px;
                border-radius: 15px;
                font-size: 18px;
            }}
            .right {{
                width: 30%;
                padding: 30px;
                background: rgba(255,255,255,0.05);
            }}
            .card {{
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 10px;
                background: rgba(255,255,255,0.08);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="left">
                <div class="grid">
                    <pre>{grid}</pre>
                </div>
            </div>
            <div class="right">
                <h2>🚨 Disaster AI</h2>
                <div class="card">Task: {obs.task_id}</div>
                <div class="card">Step: {obs.timestep}/{obs.max_timesteps}</div>
                <div class="card">Active: {obs.active_count}</div>
                <div class="card">Resolved: {obs.resolved_count}</div>
                <div class="card">Lives Saved: {obs.total_lives_saved}</div>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/visualize/{session_id}", response_class=PlainTextResponse)
def visualize(session_id: str):
    env = _sessions.get(session_id)
    if not env or env._obs is None:
        raise HTTPException(404, "Session not ready")

    return build_text_grid(env._obs.incidents, env._obs.resources)

# ── RUN ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)