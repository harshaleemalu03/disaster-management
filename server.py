"""
OpenEnv HTTP Server v2.1 — Disaster Response Coordination System
================================================================
FastAPI server implementing the full OpenEnv REST interface.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

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

@app.post("/reset")
def reset(req: ResetRequest):
    if req.task_id not in VALID_TASKS:
        raise HTTPException(400, f"Invalid task_id")

    sid = str(uuid.uuid4())
    env = DisasterResponseEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[sid] = env

    return {"session_id": sid, "observation": obs.model_dump()}

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
    return {"score": result.grader_score, "details": result.model_dump()}

# ── BEAUTIFUL UI ───────────────────────────────────────────

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
                box-shadow: 0 0 30px rgba(0,255,255,0.2);
                font-size: 18px;
            }}

            .right {{
                width: 30%;
                padding: 30px;
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(10px);
            }}

            h1 {{
                color: #38bdf8;
                margin-bottom: 20px;
            }}

            .card {{
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 10px;
                background: rgba(255,255,255,0.08);
            }}

            .value {{
                font-size: 22px;
                font-weight: bold;
                color: #22c55e;
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
                <h1>🚨 Disaster AI</h1>

                <div class="card">
                    <div>Task</div>
                    <div class="value">{obs.task_id}</div>
                </div>

                <div class="card">
                    <div>Step</div>
                    <div class="value">{obs.timestep}/{obs.max_timesteps}</div>
                </div>

                <div class="card">
                    <div>Active Incidents</div>
                    <div class="value">{obs.active_count}</div>
                </div>

                <div class="card">
                    <div>Resolved</div>
                    <div class="value">{obs.resolved_count}</div>
                </div>

                <div class="card">
                    <div>Lives Saved</div>
                    <div class="value">{obs.total_lives_saved}</div>
                </div>

            </div>

        </div>
    </body>
    </html>
    """

# ── TEXT VIEW ──────────────────────────────────────────────

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