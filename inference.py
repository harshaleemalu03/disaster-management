#!/usr/bin/env python3
"""
Inference Script — Disaster Response Coordination System
=========================================================
MANDATORY variables (per hackathon spec):
    API_BASE_URL   The LLM API base URL  (e.g. https://api.openai.com/v1 or HF endpoint)
    MODEL_NAME     The model identifier  (e.g. gpt-4o)
    HF_TOKEN       Your Hugging Face / OpenAI API key

The OpenEnv server always runs locally on localhost:7860 (via Docker/uvicorn).
All LLM calls go through: OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

STDOUT FORMAT (strictly per spec — any deviation causes incorrect scoring):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - rewards is comma-separated list of per-step rewards (2dp each).
    - All fields on a single line with no newlines within a line.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — MANDATORY env vars per hackathon spec
# ---------------------------------------------------------------------------

# LLM endpoint — where OpenAI-compatible API calls are sent
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

# OpenEnv server — always localhost when running via Docker
OPENENV_SERVER = os.getenv("OPENENV_SERVER", "http://localhost:7860")

SEED     = int(os.getenv("SEED", "42"))
ENV_NAME = "disaster_response_coordination"

TASKS = [
    "task1_prioritization",
    "task2_resource_allocation",
    "task3_dynamic_coordination",
]

# ---------------------------------------------------------------------------
# OpenAI client — always initialised with API_BASE_URL + HF_TOKEN
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key  = HF_TOKEN if HF_TOKEN else "sk-placeholder",
    base_url = API_BASE_URL,
)

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


def _dist(a: Dict, b: Dict) -> float:
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5


def _rscore(res: Dict, inc: Dict) -> float:
    c = _COMPAT.get(res["resource_type"], {}).get(inc["incident_type"], 0.15)
    d = _dist(res["location"], inc["location"])
    return 0.55 * c + 0.30 * (1.0 - d / _MAX_DIST) + 0.15 * inc.get("severity", 0.0)


# ---------------------------------------------------------------------------
# Heuristic agent — deterministic, strong baseline (no LLM required)
# ---------------------------------------------------------------------------

def heuristic_action(obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Deterministic greedy heuristic:
      Task 1:   sort by urgency_score DESC → reprioritize
      Task 2/3: pick highest-urgency unattended incident,
                assign nearest compatible free resource → assign_resource
    Secondary sort on ID guarantees identical output for identical state.
    """
    incidents = obs.get("incidents", [])
    resources = obs.get("resources", [])
    active    = [i for i in incidents if i.get("status") in ("active", "contained")]
    free      = [r for r in resources  if r.get("available", False)]

    # ── Task 1: reprioritize ──────────────────────────────────────────────
    if task_id == "task1_prioritization":
        ordered = sorted(
            active,
            key=lambda i: (
                -i.get("urgency_score", 0.0),
                -i.get("severity", 0.0),
                -i.get("people_affected", 0),
                i["id"],
            ),
        )
        return {
            "action_type":          "reprioritize",
            "ordered_incident_ids": [i["id"] for i in ordered],
        }

    # ── Tasks 2 & 3: assign resource ──────────────────────────────────────
    if not active or not free:
        return {"action_type": "wait"}

    unattended  = [i for i in active if not i.get("assigned_resources", [])]
    pool        = unattended if unattended else active
    pool_sorted = sorted(
        pool,
        key=lambda i: (
            -i.get("urgency_score", 0.0),
            -i.get("severity", 0.0),
            i.get("steps_to_expiry", 999),
            i["id"],
        ),
    )

    for target in pool_sorted:
        scored = sorted(free, key=lambda r: (-_rscore(r, target), r["id"]))
        if scored:
            return {
                "action_type": "assign_resource",
                "resource_id": scored[0]["id"],
                "incident_id": target["id"],
            }

    return {"action_type": "wait"}


# ---------------------------------------------------------------------------
# LLM agent — uses OpenAI client with API_BASE_URL + HF_TOKEN
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an AI emergency dispatch coordinator.

Each step you receive the current disaster scene and must output ONE JSON action.
No markdown, no explanation — ONLY the JSON object.

AVAILABLE ACTIONS:
  {"action_type":"assign_resource","resource_id":"RES-01","incident_id":"INC-003"}
  {"action_type":"reprioritize","ordered_incident_ids":["INC-003","INC-001","INC-002"]}
  {"action_type":"recall_resource","resource_id":"RES-02"}
  {"action_type":"wait"}

DECISION RULES:
- Task 1: ALWAYS use reprioritize, order by urgency_score field (highest first)
- Tasks 2/3: use assign_resource; prefer unattended high-urgency incidents
- Match types: ambulance→medical/accident, fire_truck→fire/hazmat, rescue_team→earthquake/flood
- NEVER assign a resource where available=false
- NEVER assign to status=resolved or status=failed incidents
- Consider steps_to_expiry — near-zero incidents need immediate help

Output ONLY valid JSON. Nothing else."""


def llm_action(obs_text: str, task_id: str) -> Dict[str, Any]:
    """
    Call the LLM via OpenAI client (API_BASE_URL + HF_TOKEN).
    Returns parsed action dict, or empty dict on failure (triggers heuristic fallback).
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Task: {task_id}\n\n{obs_text}\n\nYour JSON action:"},
            ],
            temperature=0.0,
            max_tokens=200,
            seed=SEED,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if the model adds them
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {}   # fallback to heuristic


# ---------------------------------------------------------------------------
# OpenEnv server HTTP client (talks to localhost:7860)
# ---------------------------------------------------------------------------

def _post(path: str, body: Dict) -> Dict:
    r = requests.post(f"{OPENENV_SERVER}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _get(path: str) -> Dict:
    r = requests.get(f"{OPENENV_SERVER}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def env_reset(task_id: str) -> Tuple[str, Dict]:
    data = _post("/reset", {"task_id": task_id, "seed": SEED})
    return data["session_id"], data["observation"]


def env_step(sid: str, action: Dict) -> Tuple[Dict, float, bool, Dict]:
    """Returns (observation_dict, reward_value, done, info)."""
    data = _post("/step", {"session_id": sid, "action": action})
    return (
        data["observation"],
        data["reward"]["value"],
        data["done"],
        data["info"],
    )


def env_render(sid: str) -> str:
    try:
        return _get(f"/render/{sid}").get("text", "")
    except Exception:
        return ""


def env_grade(sid: str) -> Tuple[float, str]:
    try:
        data = _get(f"/grade/{sid}")
        return data.get("grader_score", 0.0), data.get("description", "")
    except Exception:
        return 0.0, ""


# ---------------------------------------------------------------------------
# Run one task episode — emits strict [START] / [STEP] / [END] log
# ---------------------------------------------------------------------------

def run_task(task_id: str, use_llm: bool) -> Dict[str, Any]:
    """
    Runs one full episode for task_id.
    Always emits exactly:
      1 × [START]
      N × [STEP]  (one per env.step() call)
      1 × [END]   (even on exception)
    """
    # ── [START] ──────────────────────────────────────────────────────────
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    sid:              Optional[str]  = None
    obs_dict:         Dict[str, Any] = {}
    per_step_rewards: List[float]    = []
    step_num:         int            = 0
    success:          bool           = True

    try:
        sid, obs_dict = env_reset(task_id)
    except Exception as exc:
        # Server unreachable — emit [END] immediately
        print(f"[END] success=false steps=0 rewards=")
        return {"task_id": task_id, "success": False, "steps": 0,
                "rewards": [], "grader_score": 0.0}

    while True:
        step_num  += 1
        error_str  = "null"
        r_val      = 0.0
        done       = False

        # Choose action: try LLM first, fall back to heuristic
        action: Dict[str, Any] = {}
        if use_llm:
            obs_text = env_render(sid)
            if obs_text:
                action = llm_action(obs_text, task_id)

        if not action or "action_type" not in action:
            action = heuristic_action(obs_dict, task_id)

        # Execute step against the OpenEnv server
        try:
            obs_dict, r_val, done, _info = env_step(sid, action)
            per_step_rewards.append(r_val)
        except Exception as exc:
            error_str = str(exc).replace("\n", " ")[:120]
            success   = False
            done      = True
            per_step_rewards.append(0.0)

        # ── [STEP] ───────────────────────────────────────────────────────
        action_str = json.dumps(action, separators=(",", ":"))
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={r_val:.2f} done={str(done).lower()} error={error_str}"
        )

        if done:
            break

    # Grade episode
    grader_score, description = env_grade(sid) if sid else (0.0, "")

    # ── [END] ─────────────────────────────────────────────────────────────
    rewards_str = ",".join(f"{r:.2f}" for r in per_step_rewards)
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}")

    return {
        "task_id":      task_id,
        "success":      success,
        "steps":        step_num,
        "rewards":      per_step_rewards,
        "grader_score": grader_score,
        "description":  description,
    }


# ---------------------------------------------------------------------------
# Summary (stderr only — does not pollute stdout log format)
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict]) -> None:
    diff_label = {
        "task1_prioritization":       "EASY  ",
        "task2_resource_allocation":  "MEDIUM",
        "task3_dynamic_coordination": "HARD  ",
    }
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════╗",
        "║     DISASTER RESPONSE — INFERENCE SUMMARY               ║",
        "╠══════════════════════════════════════════════════════════╣",
    ]
    for r in results:
        bar_len = int(r["grader_score"] * 28)
        bar     = "█" * bar_len + "░" * (28 - bar_len)
        diff    = diff_label.get(r["task_id"], "      ")
        lines.append(
            f"║ {diff} │ {r['task_id'][:24]:<24} │ {bar} {r['grader_score']:.4f} ║"
        )
    lines.append("╠══════════════════════════════════════════════════════════╣")
    overall = sum(r["grader_score"] for r in results) / max(len(results), 1)
    bar_len  = int(overall * 28)
    bar      = "█" * bar_len + "░" * (28 - bar_len)
    lines.append(f"║ OVERALL │ {'':24} │ {bar} {overall:.4f} ║")
    lines.append("╚══════════════════════════════════════════════════════════╝")
    print("\n".join(lines), file=sys.stderr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Determine whether to use LLM (only when HF_TOKEN provided)
    use_llm = bool(HF_TOKEN) and HF_TOKEN != "sk-placeholder"

    print(
        f"# Disaster Response Coordination System — Inference",
        file=sys.stderr,
    )
    print(
        f"# env={ENV_NAME}  model={MODEL_NAME}  seed={SEED}  "
        f"llm={use_llm}  server={OPENENV_SERVER}",
        file=sys.stderr,
    )

    # Verify OpenEnv server is reachable
    try:
        resp = requests.get(f"{OPENENV_SERVER}/health", timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"# ERROR: OpenEnv server not reachable at {OPENENV_SERVER}: {exc}",
              file=sys.stderr)
        # Emit [END] for each task so log parsers get valid output
        for task_id in TASKS:
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")
            print(f"[END] success=false steps=0 rewards=")
        sys.exit(1)

    results: List[Dict] = []
    for task_id in TASKS:
        result = run_task(task_id, use_llm)
        results.append(result)
        time.sleep(0.1)   # brief pause between tasks

    _print_summary(results)


if __name__ == "__main__":
    main()
