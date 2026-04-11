#!/usr/bin/env python3
"""
Inference Script v2.2 — Disaster Response Coordination System
=============================================================
MANDATORY environment variables (per hackathon spec):
    API_BASE_URL   The OpenEnv server URL AND LLM base URL
                   (e.g. http://localhost:7860 or https://your-hf-space.hf.space)
    MODEL_NAME     The LLM model identifier (e.g. gpt-4o)
    HF_TOKEN       Your Hugging Face / OpenAI API key

STDOUT FORMAT — strictly per spec (any deviation causes evaluation failure):
    [START] task=<name> env=<env_name> model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Rules (verbatim from spec):
  - One [START] at episode begin.
  - One [STEP] per step, immediately after env.step() returns.
  - One [END] after env closes — ALWAYS emitted, even on exception.
  - reward and rewards formatted to 2 decimal places.
  - done and success are lowercase booleans: true or false.
  - error is the raw error string, or null if none.
  - All fields on a single line, no embedded newlines.
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
# Configuration — read from environment variables per hackathon spec
# ---------------------------------------------------------------------------
# API_BASE_URL is used as BOTH the OpenEnv server URL and the LLM base URL.
# When running locally: http://localhost:7860
# When deployed on HF:  https://your-space.hf.space
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
# OpenAI client — always initialised (uses API_BASE_URL + HF_TOKEN)
# LLM calls go to: {API_BASE_URL}/v1/chat/completions
# ---------------------------------------------------------------------------
_llm_client: Optional[OpenAI] = None
try:
    _llm_client = OpenAI(
        api_key  = HF_TOKEN if HF_TOKEN else "sk-no-key-heuristic-mode",
        base_url = API_BASE_URL + "/v1",
    )
except Exception:
    _llm_client = None

# ---------------------------------------------------------------------------
# Physics lookup (self-contained — no env import required)
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
# Heuristic agent — deterministic strong baseline, no LLM needed
# ---------------------------------------------------------------------------
def heuristic_action(obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Priority-based greedy heuristic.
    Task 1:   sort by urgency_score DESC → reprioritize
    Task 2/3: pick highest-urgency unattended incident,
              assign nearest compatible free resource.
    Fully deterministic via secondary sort on ID.
    """
    incidents = obs.get("incidents", [])
    resources = obs.get("resources", [])
    active    = [i for i in incidents if i.get("status") in ("active", "contained")]
    free      = [r for r in resources  if r.get("available", False)]

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
# LLM agent — uses OpenAI client (API_BASE_URL/v1 + HF_TOKEN)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an AI emergency dispatch coordinator.
Output ONE JSON action per step. No markdown. No explanation.

ACTIONS:
  {"action_type":"assign_resource","resource_id":"RES-01","incident_id":"INC-003"}
  {"action_type":"reprioritize","ordered_incident_ids":["INC-003","INC-001"]}
  {"action_type":"recall_resource","resource_id":"RES-02"}
  {"action_type":"wait"}

RULES:
- Task 1: ALWAYS reprioritize by urgency_score DESC
- Tasks 2/3: assign_resource to highest urgency_score unattended incident
- Match: ambulance→medical/accident, fire_truck→fire/hazmat, rescue_team→earthquake/flood
- NEVER assign available=false resources or resolved/failed incidents

JSON only. Nothing else."""


def llm_action(obs_text: str, task_id: str) -> Dict[str, Any]:
    """Call LLM. Returns action dict or {} on any failure."""
    if _llm_client is None or not HF_TOKEN:
        return {}
    try:
        resp = _llm_client.chat.completions.create(
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
# OpenEnv HTTP client — talks to API_BASE_URL (the env server)
# ---------------------------------------------------------------------------
def _post(path: str, body: Dict, timeout: int = 30) -> Dict:
    r = requests.post(f"{API_BASE_URL}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path: str, timeout: int = 30) -> Dict:
    r = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _wait_for_server(max_wait: int = 120) -> bool:
    """
    Poll /health until the server responds 200.
    Returns True when ready, False on timeout.
    Never raises — all exceptions caught internally.
    """
    deadline = time.time() + max_wait
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"# Server ready after {attempt} poll(s)", file=sys.stderr)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def env_reset(task_id: str) -> Tuple[str, Dict]:
    data = _post("/reset", {"task_id": task_id, "seed": SEED})
    return data["session_id"], data["observation"]


def env_step(sid: str, action: Dict) -> Tuple[Dict, float, bool, Dict]:
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
# Run one episode — emits [START] / [STEP]×N / [END]
# NEVER raises — all exceptions handled internally.
# NEVER calls sys.exit — always returns normally.
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
    except Exception as exc:
        # Server unreachable — emit [END] and return (no sys.exit)
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
        if use_llm:
            try:
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
        action_str = json.dumps(action, separators=(",", ":"))
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
# Summary — written to stderr so it never pollutes stdout log parsing
# ---------------------------------------------------------------------------
def _print_summary(results: List[Dict]) -> None:
    diff_lbl = {
        "task1_prioritization":       "EASY  ",
        "task2_resource_allocation":  "MEDIUM",
        "task3_dynamic_coordination": "HARD  ",
    }
    lines = [
        "",
        "╔═════════════════════════════════════════════════════════╗",
        "║    DISASTER RESPONSE — BASELINE INFERENCE SUMMARY      ║",
        "╠═════════════════════════════════════════════════════════╣",
    ]
    for r in results:
        bar = "█" * int(r["grader_score"] * 28) + "░" * (28 - int(r["grader_score"] * 28))
        d   = diff_lbl.get(r["task_id"], "      ")
        lines.append(f"║ {d} │ {r['task_id'][:23]:<23} │ {bar} {r['grader_score']:.4f} ║")
    lines.append("╠═════════════════════════════════════════════════════════╣")
    overall = sum(r["grader_score"] for r in results) / max(len(results), 1)
    bar     = "█" * int(overall * 28) + "░" * (28 - int(overall * 28))
    lines.append(f"║ OVERALL │ {'':23} │ {bar} {overall:.4f} ║")
    lines.append("╚═════════════════════════════════════════════════════════╝")
    print("\n".join(lines), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Main — NEVER raises, NEVER calls sys.exit
# ---------------------------------------------------------------------------
def main() -> None:
    use_llm = bool(HF_TOKEN) and HF_TOKEN not in ("", "sk-no-key-heuristic-mode")

    print(
        f"# Disaster Response — Inference v2.2  "
        f"model={MODEL_NAME}  seed={SEED}  llm={use_llm}  server={API_BASE_URL}",
        file=sys.stderr, flush=True,
    )

    # Wait for the OpenEnv server to be ready (retry up to 120s)
    server_ready = _wait_for_server(max_wait=120)
    if not server_ready:
        print(
            f"# WARNING: server not ready after 120s — running in degraded mode",
            file=sys.stderr, flush=True,
        )
        # Still emit valid [START]/[END] for every task so parser gets output
        for task_id in TASKS:
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=", flush=True)
        return   # return, not sys.exit

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
    # Normal return — exit code 0


if __name__ == "__main__":
    main()
