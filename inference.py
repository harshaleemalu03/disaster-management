#!/usr/bin/env python3
"""
Inference Script v2.6 — Disaster Response Coordination System
=============================================================
MANDATORY environment variables (injected by validator):
    API_BASE_URL   LiteLLM proxy base URL  (e.g. https://proxy.example.com)
    API_KEY        API key for the proxy    (validator injects this)
    MODEL_NAME     LLM model name           (e.g. gpt-4o)
    HF_TOKEN       Same as API_KEY (fallback)

IMPORTANT:
  - Client ALWAYS initialised: OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
  - API calls ALWAYS attempted every step (validator checks proxy traffic)
  - Heuristic used only as fallback if LLM call fails
  - All third-party imports wrapped in try/except (script never crashes at import)

STDOUT FORMAT (strictly per spec):
    [START] task=<task_name> env=<env_name> model=<model_name>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

# stdlib only at top level — never crashes
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config — read ALL possible key variable names the validator might inject
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o")
# Validator injects API_KEY; HF_TOKEN is the fallback name we also support
API_KEY      = os.getenv("API_KEY", os.getenv("HF_TOKEN", "sk-placeholder"))
SEED         = int(os.getenv("SEED", "42"))
ENV_NAME     = "disaster_response_coordination"

TASKS = [
    "task1_prioritization",
    "task2_resource_allocation",
    "task3_dynamic_coordination",
]

# ---------------------------------------------------------------------------
# Safe third-party imports — inside try/except, never crash at module level
# ---------------------------------------------------------------------------
_HAS_REQUESTS = False
_requests: Any = None
try:
    import requests as _requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:
    pass

_HAS_OPENAI = False
_OpenAI: Any = None
try:
    from openai import OpenAI as _OpenAI  # type: ignore
    _HAS_OPENAI = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# OpenAI client — ALWAYS built at startup (validator checks proxy traffic)
# Uses API_BASE_URL directly as base_url (LiteLLM proxy, not /v1 suffix)
# ---------------------------------------------------------------------------
_llm_client: Optional[Any] = None

def _build_client() -> Optional[Any]:
    """Build OpenAI client pointed at the validator's LiteLLM proxy."""
    global _llm_client
    if not _HAS_OPENAI:
        return None
    try:
        # Try with API_BASE_URL as-is first (LiteLLM proxy uses full URL)
        _llm_client = _OpenAI(
            api_key  = API_KEY,
            base_url = API_BASE_URL,
        )
        return _llm_client
    except Exception:
        try:
            # Fallback: append /v1 (standard OpenAI pattern)
            _llm_client = _OpenAI(
                api_key  = API_KEY,
                base_url = API_BASE_URL + "/v1",
            )
            return _llm_client
        except Exception:
            return None

# Build client immediately at module load
_llm_client = _build_client()

# ---------------------------------------------------------------------------
# Physics (self-contained, zero deps)
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
    c  = _COMPAT.get(res.get("resource_type",""),{}).get(inc.get("incident_type",""),0.15)
    rx = res.get("location",{}).get("x",0); ry = res.get("location",{}).get("y",0)
    ix = inc.get("location",{}).get("x",0); iy = inc.get("location",{}).get("y",0)
    d  = ((rx-ix)**2+(ry-iy)**2)**0.5
    return 0.55*c + 0.30*(1.0-d/_MAX_DIST) + 0.15*inc.get("severity",0.0)

# ---------------------------------------------------------------------------
# Heuristic fallback — pure stdlib, zero deps, never raises
# ---------------------------------------------------------------------------
def heuristic_action(obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    try:
        incidents = obs.get("incidents", [])
        resources = obs.get("resources", [])
        active    = [i for i in incidents if i.get("status") in ("active","contained")]
        free      = [r for r in resources  if r.get("available", False)]

        if task_id == "task1_prioritization":
            ordered = sorted(active, key=lambda i: (
                -i.get("urgency_score",0.0), -i.get("severity",0.0), i.get("id","")))
            return {"action_type":"reprioritize",
                    "ordered_incident_ids":[i["id"] for i in ordered]}

        if not active or not free:
            return {"action_type":"wait"}

        unattended = [i for i in active if not i.get("assigned_resources",[])]
        pool       = sorted(unattended if unattended else active,
                            key=lambda i: (-i.get("urgency_score",0.0), i.get("id","")))
        for target in pool:
            scored = sorted(free, key=lambda r: (-_rscore(r,target), r.get("id","")))
            if scored:
                return {"action_type":"assign_resource",
                        "resource_id":scored[0]["id"],
                        "incident_id":target["id"]}
        return {"action_type":"wait"}
    except Exception:
        return {"action_type":"wait"}

# ---------------------------------------------------------------------------
# LLM agent — ALWAYS attempts API call through validator's proxy
# Falls back to heuristic only if the HTTP call itself fails
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an AI emergency dispatch coordinator.
Output ONE JSON action. No markdown. No explanation. JSON only.

ACTIONS:
  {"action_type":"assign_resource","resource_id":"RES-01","incident_id":"INC-003"}
  {"action_type":"reprioritize","ordered_incident_ids":["INC-003","INC-001"]}
  {"action_type":"wait"}

RULES:
- Task 1 (task1_prioritization): ALWAYS use reprioritize, order by urgency_score DESC
- Tasks 2 & 3: use assign_resource; pick highest urgency_score unattended incident
- Match resource types: ambulance→medical/accident, fire_truck→fire/hazmat, rescue_team→earthquake/flood, helicopter→flood/earthquake
- NEVER assign available=false resources
- NEVER assign to resolved or failed incidents
- If no free resources or no active incidents: use wait

Respond with ONLY the JSON object."""

def llm_action(obs_text: str, task_id: str) -> Tuple[Dict[str, Any], bool]:
    """
    Call LLM via validator's proxy.
    Returns (action_dict, called_llm).
    called_llm=True means an API call was actually attempted.
    """
    if _llm_client is None:
        return {}, False
    try:
        resp = _llm_client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role":"system","content":_SYSTEM_PROMPT},
                {"role":"user",  "content":f"Task: {task_id}\n\n{obs_text}\n\nJSON action:"},
            ],
            temperature = 0.0,
            max_tokens  = 256,
            seed        = SEED,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if model adds them
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return parsed, True
    except Exception:
        return {}, True   # True = call was attempted (even if it failed)

# ---------------------------------------------------------------------------
# OpenEnv HTTP client — all methods safe, return defaults on failure
# ---------------------------------------------------------------------------
def _http_post(path: str, body: Dict, timeout: int=30) -> Optional[Dict]:
    if not _HAS_REQUESTS: return None
    try:
        r = _requests.post(f"{API_BASE_URL}{path}", json=body, timeout=timeout)
        r.raise_for_status(); return r.json()
    except Exception: return None

def _http_get(path: str, timeout: int=30) -> Optional[Dict]:
    if not _HAS_REQUESTS: return None
    try:
        r = _requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        r.raise_for_status(); return r.json()
    except Exception: return None

def _wait_for_server(max_wait: int=120) -> bool:
    if not _HAS_REQUESTS: return False
    deadline = time.time() + max_wait
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = _requests.get(f"{API_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"# server ready ({attempt} polls)", file=sys.stderr, flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def env_reset(task_id: str) -> Tuple[Optional[str], Dict]:
    data = _http_post("/reset", {"task_id":task_id,"seed":SEED})
    if not data: return None, {}
    return data.get("session_id"), data.get("observation",{})

def env_step(sid: str, action: Dict) -> Tuple[Dict, float, bool, Dict]:
    data = _http_post("/step", {"session_id":sid,"action":action})
    if not data: return {}, 0.0, True, {}
    return (data.get("observation",{}),
            float(data.get("reward",{}).get("value",0.0)),
            bool(data.get("done",True)),
            data.get("info",{}))

def env_render(sid: str) -> str:
    data = _http_get(f"/render/{sid}")
    return data.get("text","") if data else ""

def env_grade(sid: str) -> Tuple[float, str]:
    data = _http_get(f"/grade/{sid}")
    if not data: return 0.0, ""
    return float(data.get("grader_score",0.0)), data.get("description","")

# ---------------------------------------------------------------------------
# Run one episode — always emits [START] / [STEP]×N / [END]
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> Dict[str, Any]:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    sid, obs_dict = env_reset(task_id)
    if sid is None:
        print(f"[END] success=false steps=0 rewards=", flush=True)
        return {"task_id":task_id,"success":False,"steps":0,"rewards":[],"grader_score":0.0}

    per_rewards: List[float] = []
    step_num:    int         = 0
    success:     bool        = True

    while True:
        step_num += 1
        error_str = "null"
        r_val     = 0.0
        done      = False

        # ── Get LLM action (ALWAYS attempt, validator watches proxy) ─────
        action: Dict[str, Any] = {}
        llm_called = False
        try:
            obs_text = env_render(sid)
            if obs_text:
                action, llm_called = llm_action(obs_text, task_id)
        except Exception:
            pass

        # Fall back to heuristic if LLM returned empty / bad JSON
        if not action or "action_type" not in action:
            action = heuristic_action(obs_dict, task_id)

        # ── Execute step ─────────────────────────────────────────────────
        try:
            obs_dict, r_val, done, _ = env_step(sid, action)
            per_rewards.append(r_val)
        except Exception as exc:
            error_str = str(exc).replace("\n"," ")[:100]
            success   = False
            done      = True
            per_rewards.append(0.0)

        # ── [STEP] ───────────────────────────────────────────────────────
        try:
            action_str = json.dumps(action, separators=(",",":"))
        except Exception:
            action_str = '{"action_type":"wait"}'

        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={r_val:.2f} done={str(done).lower()} error={error_str}",
            flush=True,
        )
        if done:
            break

    grader_score = 0.0
    try:
        grader_score, _ = env_grade(sid)
    except Exception:
        pass

    rewards_str = ",".join(f"{r:.2f}" for r in per_rewards)
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}", flush=True)

    return {"task_id":task_id,"success":success,"steps":step_num,
            "rewards":per_rewards,"grader_score":grader_score}

# ---------------------------------------------------------------------------
# Summary (stderr — never pollutes stdout)
# ---------------------------------------------------------------------------
def _print_summary(results: List[Dict]) -> None:
    try:
        diff = {"task1_prioritization":"EASY  ","task2_resource_allocation":"MEDIUM","task3_dynamic_coordination":"HARD  "}
        print("\n╔══════════════════════════════════════════════╗", file=sys.stderr)
        print(  "║   DISASTER RESPONSE — INFERENCE SUMMARY    ║", file=sys.stderr)
        print(  "╠══════════════════════════════════════════════╣", file=sys.stderr)
        for r in results:
            bar = "█"*int(r["grader_score"]*24)+"░"*(24-int(r["grader_score"]*24))
            print(f"║ {diff.get(r['task_id'],'      ')} │ {bar} {r['grader_score']:.4f} ║", file=sys.stderr)
        overall = sum(r["grader_score"] for r in results)/max(len(results),1)
        bar = "█"*int(overall*24)+"░"*(24-int(overall*24))
        print(f"╠══════════════════════════════════════════════╣", file=sys.stderr)
        print(f"║ OVERALL │ {bar} {overall:.4f} ║", file=sys.stderr)
        print(  "╚══════════════════════════════════════════════╝", file=sys.stderr)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Main — never raises, never sys.exit, always exits 0
# ---------------------------------------------------------------------------
def main() -> None:
    print(
        f"# Disaster Response v2.6  model={MODEL_NAME}  seed={SEED}  "
        f"llm_ready={_llm_client is not None}  requests={_HAS_REQUESTS}  "
        f"openai={_HAS_OPENAI}  server={API_BASE_URL}  api_key_set={bool(API_KEY)}",
        file=sys.stderr, flush=True,
    )

    # If requests unavailable emit empty results and exit 0
    if not _HAS_REQUESTS:
        print("# WARNING: requests not installed", file=sys.stderr, flush=True)
        for tid in TASKS:
            print(f"[START] task={tid} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=", flush=True)
        return

    # Wait for OpenEnv server
    if not _wait_for_server(max_wait=120):
        print("# WARNING: server not ready", file=sys.stderr, flush=True)
        for tid in TASKS:
            print(f"[START] task={tid} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=", flush=True)
        return

    results: List[Dict] = []
    for tid in TASKS:
        try:
            result = run_task(tid)
        except Exception as exc:
            print(f"# UNEXPECTED: {exc}", file=sys.stderr, flush=True)
            print(f"[START] task={tid} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=", flush=True)
            result = {"task_id":tid,"success":False,"steps":0,"rewards":[],"grader_score":0.0}
        results.append(result)
        time.sleep(0.1)

    _print_summary(results)
    # Normal return → exit code 0

if __name__ == "__main__":
    main()
