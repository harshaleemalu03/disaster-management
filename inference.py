#!/usr/bin/env python3
"""
Inference Script v2.5 — Disaster Response Coordination System
=============================================================
MANDATORY environment variables (per hackathon spec):
    API_BASE_URL   OpenEnv server URL  (e.g. http://localhost:7860)
    MODEL_NAME     LLM model name      (e.g. gpt-4o)
    HF_TOKEN       API key / HF token

STDOUT FORMAT (strictly per spec):
    [START] task=<task_name> env=<env_name> model=<model_name>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

IMPORTANT: All imports wrapped in try/except — script never crashes at import time.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# SAFE STDLIB IMPORTS ONLY at top level — no third-party deps
# ---------------------------------------------------------------------------
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
SEED         = int(os.getenv("SEED",     "42"))
ENV_NAME     = "disaster_response_coordination"

TASKS = [
    "task1_prioritization",
    "task2_resource_allocation",
    "task3_dynamic_coordination",
]

# ---------------------------------------------------------------------------
# Safe third-party imports — NEVER crash at module level
# ---------------------------------------------------------------------------
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests     = None  # type: ignore
    _HAS_REQUESTS = False

try:
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except ImportError:
    _OpenAI     = None  # type: ignore
    _HAS_OPENAI = False

# ---------------------------------------------------------------------------
# OpenAI client — lazily built, never crashes at import
# ---------------------------------------------------------------------------
_llm_client: Optional[Any] = None

def _get_llm_client() -> Optional[Any]:
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    if not _HAS_OPENAI:
        return None
    if not HF_TOKEN:
        return None
    try:
        _llm_client = _OpenAI(
            api_key  = HF_TOKEN,
            base_url = API_BASE_URL + "/v1",
        )
        return _llm_client
    except Exception:
        return None

# ---------------------------------------------------------------------------
# HTTP helpers — all safe, return None on any error
# ---------------------------------------------------------------------------
def _post(path: str, body: Dict, timeout: int = 30) -> Optional[Dict]:
    if not _HAS_REQUESTS:
        return None
    try:
        r = _requests.post(f"{API_BASE_URL}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _get(path: str, timeout: int = 30) -> Optional[Dict]:
    if not _HAS_REQUESTS:
        return None
    try:
        r = _requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _wait_for_server(max_wait: int = 120) -> bool:
    """Poll /health until 200 or timeout. Never raises."""
    if not _HAS_REQUESTS:
        return False
    deadline = time.time() + max_wait
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = _requests.get(f"{API_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"# server ready (attempt {attempt})", file=sys.stderr, flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

# ---------------------------------------------------------------------------
# OpenEnv HTTP client
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> Tuple[Optional[str], Dict]:
    data = _post("/reset", {"task_id": task_id, "seed": SEED})
    if data is None:
        return None, {}
    return data.get("session_id"), data.get("observation", {})

def env_step(sid: str, action: Dict) -> Tuple[Dict, float, bool, Dict]:
    data = _post("/step", {"session_id": sid, "action": action})
    if data is None:
        return {}, 0.0, True, {}
    return (
        data.get("observation", {}),
        float(data.get("reward", {}).get("value", 0.0)),
        bool(data.get("done", True)),
        data.get("info", {}),
    )

def env_render(sid: str) -> str:
    data = _get(f"/render/{sid}")
    return data.get("text", "") if data else ""

def env_grade(sid: str) -> Tuple[float, str]:
    data = _get(f"/grade/{sid}")
    if data is None:
        return 0.0, ""
    return float(data.get("grader_score", 0.0)), data.get("description", "")

# ---------------------------------------------------------------------------
# Physics lookup (self-contained — no env import needed)
# ---------------------------------------------------------------------------
_COMPAT: Dict[str, Dict[str, float]] = {
    "ambulance":   {"medical":1.0,"accident":0.9,"fire":0.45,"flood":0.40,"earthquake":0.55,"hazmat":0.25},
    "fire_truck":  {"fire":1.0,"hazmat":0.85,"accident":0.35,"medical":0.10,"flood":0.20,"earthquake":0.30},
    "rescue_team": {"earthquake":1.0,"flood":0.95,"fire":0.50,"accident":0.60,"hazmat":0.40,"medical":0.30},
    "police":      {"accident":1.0,"hazmat":0.60,"fire":0.30,"medical":0.15,"flood":0.25,"earthquake":0.35},
    "helicopter":  {"flood":1.0,"earthquake":0.90,"fire":0.70,"medical":0.80,"accident":0.50,"hazmat":0.40},
}
_MAX_DIST = 141.42

def _rscore(res: Dict, inc: Dict) -> float:
    c = _COMPAT.get(res.get("resource_type", ""), {}).get(inc.get("incident_type", ""), 0.15)
    rx, ry = res.get("location", {}).get("x", 0), res.get("location", {}).get("y", 0)
    ix, iy = inc.get("location", {}).get("x", 0), inc.get("location", {}).get("y", 0)
    d = ((rx - ix) ** 2 + (ry - iy) ** 2) ** 0.5
    return 0.55 * c + 0.30 * (1.0 - d / _MAX_DIST) + 0.15 * inc.get("severity", 0.0)

# ---------------------------------------------------------------------------
# Heuristic agent — deterministic, no imports needed
# ---------------------------------------------------------------------------
def heuristic_action(obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Strong greedy heuristic — never raises, no external dependencies."""
    try:
        incidents = obs.get("incidents", [])
        resources = obs.get("resources", [])
        active    = [i for i in incidents if i.get("status") in ("active", "contained")]
        free      = [r for r in resources  if r.get("available", False)]

        if task_id == "task1_prioritization":
            ordered = sorted(
                active,
                key=lambda i: (-i.get("urgency_score", 0.0), -i.get("severity", 0.0), i.get("id", "")),
            )
            return {"action_type": "reprioritize",
                    "ordered_incident_ids": [i["id"] for i in ordered]}

        if not active or not free:
            return {"action_type": "wait"}

        unattended  = [i for i in active if not i.get("assigned_resources", [])]
        pool        = unattended if unattended else active
        pool_sorted = sorted(pool, key=lambda i: (-i.get("urgency_score", 0.0), i.get("id", "")))

        for target in pool_sorted:
            scored = sorted(free, key=lambda r: (-_rscore(r, target), r.get("id", "")))
            if scored:
                return {"action_type": "assign_resource",
                        "resource_id": scored[0]["id"],
                        "incident_id": target["id"]}

        return {"action_type": "wait"}
    except Exception:
        return {"action_type": "wait"}

# ---------------------------------------------------------------------------
# LLM agent — only called if openai installed AND HF_TOKEN set
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an AI emergency dispatch coordinator.
Output ONE JSON action per step. No markdown. No explanation. JSON only.

ACTIONS:
  {"action_type":"assign_resource","resource_id":"RES-01","incident_id":"INC-003"}
  {"action_type":"reprioritize","ordered_incident_ids":["INC-003","INC-001"]}
  {"action_type":"recall_resource","resource_id":"RES-02"}
  {"action_type":"wait"}

RULES:
- Task 1: reprioritize by urgency_score DESC
- Tasks 2/3: assign_resource to highest urgency unattended incident
- Match types: ambulance→medical/accident, fire_truck→fire/hazmat, rescue_team→earthquake/flood
- NEVER assign available=false or resolved/failed incidents"""

def llm_action(obs_text: str, task_id: str) -> Dict[str, Any]:
    """Call LLM. Returns {} on any failure — caller uses heuristic fallback."""
    client = _get_llm_client()
    if client is None:
        return {}
    try:
        resp = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Task: {task_id}\n\n{obs_text}\n\nYour JSON action:"},
            ],
            temperature = 0.0,
            max_tokens  = 200,
            seed        = SEED,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Run one episode — emits [START] / [STEP]×N / [END]
# NEVER raises. NEVER calls sys.exit. Always returns normally.
# ---------------------------------------------------------------------------
def run_task(task_id: str, use_llm: bool) -> Dict[str, Any]:
    # ── [START] ──────────────────────────────────────────────────────────
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    sid:              Optional[str]  = None
    obs_dict:         Dict[str, Any] = {}
    per_step_rewards: List[float]    = []
    step_num:         int            = 0
    success:          bool           = True

    try:
        sid, obs_dict = env_reset(task_id)
        if sid is None:
            print(f"[END] success=false steps=0 rewards=", flush=True)
            return {"task_id": task_id, "success": False, "steps": 0,
                    "rewards": [], "grader_score": 0.0}
    except Exception:
        print(f"[END] success=false steps=0 rewards=", flush=True)
        return {"task_id": task_id, "success": False, "steps": 0,
                "rewards": [], "grader_score": 0.0}

    while True:
        step_num  += 1
        error_str  = "null"
        r_val      = 0.0
        done       = False

        # Choose action
        action: Dict[str, Any] = {}
        try:
            if use_llm:
                obs_text = env_render(sid)
                if obs_text:
                    action = llm_action(obs_text, task_id)
        except Exception:
            action = {}

        if not action or "action_type" not in action:
            action = heuristic_action(obs_dict, task_id)

        # Execute step
        try:
            obs_dict, r_val, done, _info = env_step(sid, action)
            per_step_rewards.append(r_val)
        except Exception as exc:
            error_str = str(exc).replace("\n", " ")[:100]
            success   = False
            done      = True
            per_step_rewards.append(0.0)

        # ── [STEP] ───────────────────────────────────────────────────────
        try:
            action_str = json.dumps(action, separators=(",", ":"))
        except Exception:
            action_str = '{"action_type":"wait"}'

        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={r_val:.2f} done={str(done).lower()} error={error_str}",
            flush=True,
        )

        if done:
            break

    # Grade
    grader_score = 0.0
    try:
        grader_score, _ = env_grade(sid) if sid else (0.0, "")
    except Exception:
        pass

    # ── [END] ─────────────────────────────────────────────────────────────
    rewards_str = ",".join(f"{r:.2f}" for r in per_step_rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id":      task_id,
        "success":      success,
        "steps":        step_num,
        "rewards":      per_step_rewards,
        "grader_score": grader_score,
    }

# ---------------------------------------------------------------------------
# Summary (stderr — never pollutes stdout log format)
# ---------------------------------------------------------------------------
def _print_summary(results: List[Dict]) -> None:
    try:
        diff_lbl = {
            "task1_prioritization":       "EASY  ",
            "task2_resource_allocation":  "MEDIUM",
            "task3_dynamic_coordination": "HARD  ",
        }
        lines = [
            "",
            "╔════════════════════════════════════════════════════════╗",
            "║    DISASTER RESPONSE — BASELINE INFERENCE SUMMARY     ║",
            "╠════════════════════════════════════════════════════════╣",
        ]
        for r in results:
            bar = "█" * int(r["grader_score"] * 28) + "░" * (28 - int(r["grader_score"] * 28))
            d   = diff_lbl.get(r["task_id"], "      ")
            lines.append(f"║ {d} │ {r['task_id'][:23]:<23} │ {bar} {r['grader_score']:.4f} ║")
        lines.append("╠════════════════════════════════════════════════════════╣")
        overall = sum(r["grader_score"] for r in results) / max(len(results), 1)
        bar     = "█" * int(overall * 28) + "░" * (28 - int(overall * 28))
        lines.append(f"║ OVERALL │ {'':23} │ {bar} {overall:.4f} ║")
        lines.append("╚════════════════════════════════════════════════════════╝")
        print("\n".join(lines), file=sys.stderr, flush=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Main — NEVER raises, NEVER calls sys.exit, always exits 0
# ---------------------------------------------------------------------------
def main() -> None:
    use_llm = _HAS_OPENAI and bool(HF_TOKEN)

    print(
        f"# Disaster Response v2.5  model={MODEL_NAME}  seed={SEED}  "
        f"llm={use_llm}  requests={_HAS_REQUESTS}  openai={_HAS_OPENAI}  "
        f"server={API_BASE_URL}",
        file=sys.stderr, flush=True,
    )

    # If requests not available, emit [END] for all tasks and exit cleanly
    if not _HAS_REQUESTS:
        print("# WARNING: requests not installed — emitting empty results", file=sys.stderr, flush=True)
        for task_id in TASKS:
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=", flush=True)
        return

    # Wait for server (up to 120 s)
    server_ready = _wait_for_server(max_wait=120)
    if not server_ready:
        print("# WARNING: server not ready — emitting empty results", file=sys.stderr, flush=True)
        for task_id in TASKS:
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=", flush=True)
        return

    results: List[Dict] = []
    for task_id in TASKS:
        try:
            result = run_task(task_id, use_llm)
        except Exception as exc:
            # Absolute last-resort catch — should never reach here
            print(f"# UNEXPECTED: {exc}", file=sys.stderr, flush=True)
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=", flush=True)
            result = {"task_id": task_id, "success": False, "steps": 0,
                      "rewards": [], "grader_score": 0.0}
        results.append(result)
        time.sleep(0.1)

    _print_summary(results)
    # Normal return → exit code 0


if __name__ == "__main__":
    main()
