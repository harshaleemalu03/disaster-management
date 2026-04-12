"""
OpenEnv HTTP Server v2.4 — Disaster Response Coordination System
================================================================
KEY FIXES:
  - GET / returns beautiful HTML landing page (visible in HF Space iframe)
  - GET /demo auto-creates a live session and redirects to /view/{id}
  - POST /reset accepts null/empty body (no Pydantic binding)
  - /health registered FIRST — always 200
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

app = FastAPI(
    title       = "Disaster Response Coordination System",
    description = "OpenEnv environment for emergency dispatch coordination.",
    version     = "2.4.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, Any] = {}
_import_error: str = ""
_env_ready:    bool = False

# ── /health REGISTERED FIRST ──────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "env_ready": _env_ready, "sessions": len(_sessions)}

# ── Env imports ───────────────────────────────────────────────────────────────
try:
    from env.environment import DisasterResponseEnv, VALID_TASKS
    from env.models import ActionWrapper
    from env.world import build_text_grid, build_html_view
    _env_ready = True
except Exception as _e:
    _import_error = str(_e)
    VALID_TASKS   = ["task1_prioritization", "task2_resource_allocation", "task3_dynamic_coordination"]

    class DisasterResponseEnv:  # type: ignore
        def __init__(self, *a, **kw): raise RuntimeError(f"Env failed: {_import_error}")
    class ActionWrapper:  # type: ignore
        def __init__(self, **kw): self.__dict__.update(kw)
    def build_text_grid(*a, **kw) -> str: return "env not loaded"
    def build_html_view(*a, **kw) -> str: return "<html><body>env not loaded</body></html>"


# ── Landing page ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["ui"])
def landing():
    """
    Beautiful landing page — visible as the HF Space default view.
    Shows what the environment is, links to live demo and API docs.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🚨 Disaster Response Coordination System</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117;
  color: #e6edf3;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  min-height: 100vh;
}
.hero {
  background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
  border-bottom: 1px solid #30363d;
  padding: 60px 40px;
  text-align: center;
}
h1 {
  font-size: 2.8rem;
  font-weight: 800;
  background: linear-gradient(90deg, #58a6ff, #a371f7, #f78166);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 16px;
}
.subtitle {
  font-size: 1.1rem;
  color: #8b949e;
  max-width: 600px;
  margin: 0 auto 32px;
  line-height: 1.6;
}
.btn-row { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
.btn {
  padding: 12px 28px;
  border-radius: 8px;
  font-size: 15px;
  font-weight: 600;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: transform 0.15s, box-shadow 0.15s;
  cursor: pointer;
}
.btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.4); }
.btn-primary { background: linear-gradient(135deg, #238636, #2ea043); color: #fff; border: none; }
.btn-secondary { background: transparent; color: #58a6ff; border: 2px solid #58a6ff; }
.btn-purple { background: linear-gradient(135deg, #6f42c1, #a371f7); color: #fff; border: none; }
.badge-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 20px; }
.badge {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
}
.section { padding: 48px 40px; max-width: 1100px; margin: 0 auto; }
.section-title {
  font-size: 1.4rem;
  font-weight: 700;
  margin-bottom: 24px;
  color: #e6edf3;
}
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
.card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 12px;
  padding: 24px;
  transition: border-color 0.2s;
}
.card:hover { border-color: #58a6ff55; }
.card-icon { font-size: 2rem; margin-bottom: 12px; }
.card-title { font-size: 1rem; font-weight: 700; margin-bottom: 8px; }
.card-desc { font-size: 13px; color: #8b949e; line-height: 1.6; }
.task-row {
  display: flex;
  align-items: center;
  gap: 16px;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 16px 20px;
  margin-bottom: 12px;
}
.task-badge {
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 700;
  min-width: 60px;
  text-align: center;
}
.score-bar-wrap { flex: 1; }
.score-bar-bg { background: #21262d; border-radius: 4px; height: 8px; }
.score-bar-fill { height: 8px; border-radius: 4px; }
.score-label { font-size: 12px; color: #8b949e; margin-bottom: 4px; }
code {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 12px 16px;
  display: block;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  color: #a371f7;
  overflow-x: auto;
  margin: 8px 0;
}
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.feature {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 14px 16px;
  font-size: 13px;
}
.feature-icon { font-size: 1.4rem; margin-bottom: 6px; }
.feature-name { font-weight: 600; margin-bottom: 4px; }
.feature-desc { color: #8b949e; font-size: 12px; line-height: 1.5; }
footer {
  text-align: center;
  padding: 32px;
  border-top: 1px solid #21262d;
  color: #484f58;
  font-size: 13px;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.live { display:inline-block; width:8px; height:8px; background:#00c853;
        border-radius:50%; animation:blink 1.5s infinite; margin-right:6px; }
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div style="font-size:3rem;margin-bottom:16px">🚨</div>
  <h1>Disaster Response Coordination</h1>
  <p class="subtitle">
    A high-fidelity AI environment where agents coordinate emergency dispatch —
    allocating ambulances, fire trucks, rescue teams and helicopters across
    a dynamic disaster landscape with real travel delays, fire cascades,
    and fairness-aware decision making.
  </p>
  <div class="btn-row">
    <a href="/demo" class="btn btn-primary">▶ Launch Live Demo</a>
    <a href="/docs" class="btn btn-secondary">📖 API Docs</a>
    <a href="/tasks" class="btn btn-purple">📋 View Tasks</a>
  </div>
  <div class="badge-row">
    <span class="badge" style="background:#238636;color:#fff">OpenEnv Compliant</span>
    <span class="badge" style="background:#1f6feb;color:#fff">3 Tasks</span>
    <span class="badge" style="background:#6f42c1;color:#fff">Real-World Sim</span>
    <span class="badge" style="background:#b08800;color:#fff">Fairness-Aware</span>
    <span class="badge" style="background:#8b1a1a;color:#fff">Cascade Dynamics</span>
  </div>
</div>

<!-- FEATURES -->
<div class="section">
  <div class="section-title">🔥 Key Innovations</div>
  <div class="feature-grid">
    <div class="feature">
      <div class="feature-icon">🗺️</div>
      <div class="feature-name">Real Geographic Simulation</div>
      <div class="feature-desc">100×100 km grid with urban, suburban, and rural zones. Incidents happen at real coordinates.</div>
    </div>
    <div class="feature">
      <div class="feature-icon">🚗</div>
      <div class="feature-name">Distance-Based Travel</div>
      <div class="feature-desc">Resources travel at realistic speeds (ambulance 96 km/h, helicopter 300 km/h) with zone traffic factors.</div>
    </div>
    <div class="feature">
      <div class="feature-icon">🔥</div>
      <div class="feature-name">Fire Cascade Spread</div>
      <div class="feature-desc">Unattended fires spread to adjacent areas. Each cascade penalises the agent score by −0.04.</div>
    </div>
    <div class="feature">
      <div class="feature-icon">⚖️</div>
      <div class="feature-name">Fairness-Aware Dispatch</div>
      <div class="feature-desc">Gini coefficient tracks response equity across urban/suburban/rural zones. Rural neglect is penalised.</div>
    </div>
    <div class="feature">
      <div class="feature-icon">📈</div>
      <div class="feature-name">Incident Escalation</div>
      <div class="feature-desc">Unattended incidents worsen over time. Fire: +0.06 severity/step, people ×1.12/step.</div>
    </div>
    <div class="feature">
      <div class="feature-icon">🎯</div>
      <div class="feature-name">Dense Reward Signal</div>
      <div class="feature-desc">Every step: 0.35×life_saving + 0.25×response_time + 0.20×efficiency + 0.10×fairness.</div>
    </div>
  </div>
</div>

<!-- TASKS -->
<div class="section" style="padding-top:0">
  <div class="section-title">📊 Three Tasks (Easy → Hard)</div>
  <div class="task-row">
    <span class="task-badge" style="background:#0d4b1e;color:#3fb950">EASY</span>
    <div style="flex:1">
      <div style="font-weight:600;margin-bottom:4px">Task 1 — Incident Prioritization</div>
      <div style="font-size:13px;color:#8b949e">Order 6 incidents by urgency each step. Graded by Kendall-tau ordering accuracy + fairness.</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:18px;font-weight:700;color:#3fb950">1.00</div>
      <div style="font-size:11px;color:#8b949e">baseline</div>
    </div>
  </div>
  <div class="task-row">
    <span class="task-badge" style="background:#3d2900;color:#d29922">MEDIUM</span>
    <div style="flex:1">
      <div style="font-weight:600;margin-bottom:4px">Task 2 — Resource Allocation</div>
      <div style="font-size:13px;color:#8b949e">Assign 6 typed resources to 5 incidents. Resources travel before arriving. Match types: ambulance→medical, fire truck→fire.</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:18px;font-weight:700;color:#d29922">0.61</div>
      <div style="font-size:11px;color:#8b949e">baseline</div>
    </div>
  </div>
  <div class="task-row">
    <span class="task-badge" style="background:#4b0000;color:#f85149">HARD</span>
    <div style="flex:1">
      <div style="font-weight:600;margin-bottom:4px">Task 3 — Dynamic Multi-step Coordination</div>
      <div style="font-size:13px;color:#8b949e">Incidents escalate, new ones spawn (p=0.28/step), fires cascade-spread. Continuously redeploy 6 resources over 30 steps.</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:18px;font-weight:700;color:#f85149">0.27</div>
      <div style="font-size:11px;color:#8b949e">baseline</div>
    </div>
  </div>
</div>

<!-- QUICK START -->
<div class="section" style="padding-top:0">
  <div class="section-title">⚡ Quick Start</div>
  <div class="cards">
    <div class="card">
      <div class="card-icon">🌐</div>
      <div class="card-title">Live Dashboard</div>
      <div class="card-desc" style="margin-bottom:12px">Click "Launch Live Demo" above to see the real-time map with incidents and resources.</div>
      <a href="/demo" class="btn btn-primary" style="font-size:13px;padding:8px 16px">Open Dashboard →</a>
    </div>
    <div class="card">
      <div class="card-icon">🐍</div>
      <div class="card-title">Python API</div>
      <code>import requests
r = requests.post("/reset", json={"task_id":"task3_dynamic_coordination"})
sid = r.json()["session_id"]
# Then GET /view/{sid} for live map</code>
    </div>
    <div class="card">
      <div class="card-icon">🤖</div>
      <div class="card-title">Run Inference</div>
      <code>API_BASE_URL=https://your.hf.space \\
MODEL_NAME=gpt-4o \\
HF_TOKEN=your_token \\
python inference.py</code>
    </div>
  </div>
</div>

<!-- ENDPOINTS -->
<div class="section" style="padding-top:0">
  <div class="section-title">🔌 API Endpoints</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px">
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px">
      <span style="color:#f85149;font-family:monospace;font-size:13px">POST</span>
      <span style="font-family:monospace;font-size:13px;margin-left:8px">/reset</span>
      <div style="font-size:12px;color:#8b949e;margin-top:4px">Start episode. Body optional — defaults to task1, seed=42.</div>
    </div>
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px">
      <span style="color:#f85149;font-family:monospace;font-size:13px">POST</span>
      <span style="font-family:monospace;font-size:13px;margin-left:8px">/step</span>
      <div style="font-size:12px;color:#8b949e;margin-top:4px">Execute action. Returns observation, reward, done, info.</div>
    </div>
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px">
      <span style="color:#3fb950;font-family:monospace;font-size:13px">GET</span>
      <span style="font-family:monospace;font-size:13px;margin-left:8px">/grade/{id}</span>
      <div style="font-size:12px;color:#8b949e;margin-top:4px">Deterministic grader score [0.0–1.0] with breakdown.</div>
    </div>
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px">
      <span style="color:#3fb950;font-family:monospace;font-size:13px">GET</span>
      <span style="font-family:monospace;font-size:13px;margin-left:8px">/view/{id}</span>
      <div style="font-size:12px;color:#8b949e;margin-top:4px">Live HTML dashboard — grid map, tables, fairness, cascades.</div>
    </div>
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px">
      <span style="color:#3fb950;font-family:monospace;font-size:13px">GET</span>
      <span style="font-family:monospace;font-size:13px;margin-left:8px">/visualize/{id}</span>
      <div style="font-size:12px;color:#8b949e;margin-top:4px">Emoji text grid: 🔥🚑 in a 10×10 terminal map.</div>
    </div>
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px">
      <span style="color:#3fb950;font-family:monospace;font-size:13px">GET</span>
      <span style="font-family:monospace;font-size:13px;margin-left:8px">/demo</span>
      <div style="font-size:12px;color:#8b949e;margin-top:4px">Auto-creates session + redirects to live dashboard.</div>
    </div>
  </div>
</div>

<footer>
  Disaster Response Coordination System v2.4 &nbsp;·&nbsp; OpenEnv &nbsp;·&nbsp;
  Built for the Meta × Scaler OpenEnv Hackathon
</footer>

</body>
</html>"""


# ── Demo endpoint — auto session → redirect to dashboard ─────────────────────

@app.get("/demo", response_class=HTMLResponse, tags=["ui"])
def demo(task: str = "task3_dynamic_coordination"):
    """
    Auto-creates a session for the given task and shows the live dashboard.
    Runs a few heuristic steps so the map has interesting content.
    """
    if not _env_ready:
        return HTMLResponse("<html><body style='background:#0d1117;color:#e6edf3;padding:2rem'>"
                           f"<h2>⚠️ Environment loading... {_import_error}</h2></body></html>")

    if task not in VALID_TASKS:
        task = "task3_dynamic_coordination"

    try:
        env = DisasterResponseEnv(task_id=task, seed=42)
        obs = env.reset()
        sid = str(uuid.uuid4())
        _sessions[sid] = env

        # Run a few heuristic steps so the map has incidents + resources placed
        from env.models import ActionWrapper, ActionType
        _COMPAT = {
            "ambulance":   {"medical":1.0,"accident":0.9,"fire":0.45,"flood":0.40,"earthquake":0.55,"hazmat":0.25},
            "fire_truck":  {"fire":1.0,"hazmat":0.85,"accident":0.35,"medical":0.10,"flood":0.20,"earthquake":0.30},
            "rescue_team": {"earthquake":1.0,"flood":0.95,"fire":0.50,"accident":0.60,"hazmat":0.40,"medical":0.30},
            "police":      {"accident":1.0,"hazmat":0.60,"fire":0.30,"medical":0.15,"flood":0.25,"earthquake":0.35},
            "helicopter":  {"flood":1.0,"earthquake":0.90,"fire":0.70,"medical":0.80,"accident":0.50,"hazmat":0.40},
        }
        steps = {"task1_prioritization": 3, "task2_resource_allocation": 6, "task3_dynamic_coordination": 10}
        for _ in range(steps.get(task, 6)):
            active = [i for i in obs.incidents if i.is_active]
            free   = [r for r in obs.resources if r.available]
            if task == "task1_prioritization":
                ordered = sorted(active, key=lambda i: -i.urgency_score())
                a = ActionWrapper(action_type=ActionType.REPRIORITIZE,
                                  ordered_incident_ids=[i.id for i in ordered])
            elif active and free:
                target = max(active, key=lambda i: i.urgency_score())
                def sc(r, t=target):
                    c = _COMPAT.get(r.rtype_str(), {}).get(t.itype_str(), 0.15)
                    d = r.location.distance_to(t.location)
                    return 0.55 * c + 0.30 * (1 - d / 141.42)
                best = max(free, key=sc)
                a = ActionWrapper(action_type=ActionType.ASSIGN_RESOURCE,
                                  resource_id=best.id, incident_id=target.id)
            else:
                a = ActionWrapper(action_type=ActionType.WAIT)
            obs, _, done, _ = env.step(a)
            if done:
                break

        # Serve the live dashboard directly (not a redirect — avoids session loss)
        return HTMLResponse(build_html_view(
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
        ))
    except Exception as exc:
        return HTMLResponse(
            f"<html><body style='background:#0d1117;color:#e6edf3;padding:2rem'>"
            f"<h2>⚠️ Demo error: {exc}</h2></body></html>"
        )


# ── Meta endpoints ────────────────────────────────────────────────────────────

@app.get("/tasks", tags=["meta"])
def list_tasks():
    return {
        "tasks": [
            {"id":"task1_prioritization","name":"Incident Prioritization","difficulty":"easy","max_steps":10,"baseline_score":1.00},
            {"id":"task2_resource_allocation","name":"Resource Allocation","difficulty":"medium","max_steps":20,"baseline_score":0.61},
            {"id":"task3_dynamic_coordination","name":"Dynamic Coordination","difficulty":"hard","max_steps":30,"baseline_score":0.27},
        ]
    }


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset", tags=["openenv"])
async def reset(request: Request):
    """POST /reset — body is fully optional. Null/empty body uses defaults."""
    if not _env_ready:
        raise HTTPException(503, f"Environment not ready: {_import_error}")
    task_id = "task1_prioritization"
    seed    = 42
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = body.get("task_id", task_id)
            seed    = int(body.get("seed", seed))
    except Exception:
        pass
    if task_id not in VALID_TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'. Valid: {VALID_TASKS}")
    sid = str(uuid.uuid4())
    env = DisasterResponseEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    _sessions[sid] = env
    return {"session_id": sid, "task_id": task_id, "observation": obs.model_dump()}


@app.post("/step", tags=["openenv"])
async def step(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(422, "Body must be JSON with session_id and action.")
    session_id  = body.get("session_id")
    action_dict = body.get("action")
    if not session_id or action_dict is None:
        raise HTTPException(422, "Body must contain 'session_id' and 'action'.")
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call POST /reset first.")
    try:
        action               = ActionWrapper(**action_dict)
        obs, reward, done, info = env.step(action)
    except Exception as exc:
        raise HTTPException(422, f"Action error: {exc}")
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}


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
    safe_score  = round(max(0.0001, min(0.9999, float(result.grader_score))), 4)
    result.grader_score = safe_score
    return {"session_id": session_id, "result": result.model_dump(),
            "grader_score": safe_score, "description": description}


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


# ── Visualization ─────────────────────────────────────────────────────────────

@app.get("/visualize/{session_id}", tags=["visualization"], response_class=PlainTextResponse)
def visualize(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    if env._obs is None:
        raise HTTPException(400, "No observation — call /reset first.")
    obs  = env._obs
    grid = build_text_grid(obs.incidents, obs.resources)
    return (f"DISASTER MAP: {obs.task_id} | Step {obs.timestep}/{obs.max_timesteps}\n"
            f"Active:{obs.active_count} Resolved:{obs.resolved_count} "
            f"Failed:{obs.failed_count} Lives:{obs.total_lives_saved}\n\n"
            + grid +
            "\n\nLegend: 🔥Fire 🏥Medical 🌊Flood ⚡Quake 💥Accident ☣️Hazmat | 🚑🚒🛟🚓🚁")


@app.get("/view/{session_id}", tags=["visualization"], response_class=HTMLResponse)
def view(session_id: str):
    """Real-time HTML dashboard — incidents, resources, fairness, cascades."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    if env._obs is None:
        raise HTTPException(400, "No observation — call /reset first.")
    obs = env._obs
    return HTMLResponse(build_html_view(
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
    ))


# ── Schema ────────────────────────────────────────────────────────────────────

@app.get("/schema/observation", tags=["schema"])
def obs_schema():
    if not _env_ready: raise HTTPException(503, "Environment not ready")
    return DisasterResponseEnv().observation_schema()

@app.get("/schema/action", tags=["schema"])
def action_schema():
    if not _env_ready: raise HTTPException(503, "Environment not ready")
    return DisasterResponseEnv().action_schema()

@app.get("/schema/reward", tags=["schema"])
def reward_schema():
    if not _env_ready: raise HTTPException(503, "Environment not ready")
    return DisasterResponseEnv().reward_schema()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
