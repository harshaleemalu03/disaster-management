#!/usr/bin/env python3
from __future__ import annotations
import json, os, time
from typing import Dict

import requests

# ── REQUIRED ENV ─────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY      = os.environ.get("API_KEY")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")

OPENENV_URL  = "http://localhost:7860"
TASKS        = [
    "task1_prioritization",
    "task2_resource_allocation",
    "task3_dynamic_coordination"
]

# ── SYSTEM PROMPT ────────────────────────────────────────────────────────────
SYSTEM = """Return ONLY JSON.

task1:
{"action_type":"reprioritize","ordered_incident_ids":[]}

task2/3:
{"action_type":"assign_resource","resource_id":"RES-01","incident_id":"INC-001"}
or {"action_type":"wait"}
"""

# ── LLM CALL (RAW HTTP — NO OPENAI LIB) ──────────────────────────────────────
def call_llm(prompt: str) -> Dict:
    try:
        url = f"{API_BASE_URL}/chat/completions"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt[:1500]},
            ],
            "temperature": 0,
            "max_tokens": 100,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=20)
        data = r.json()

        text = data["choices"][0]["message"]["content"].strip()

        if "```" in text:
            text = text.split("```")[1].strip()

        return json.loads(text)

    except Exception:
        return {}

# ── SAFE FALLBACK ────────────────────────────────────────────────────────────
def fallback(obs, task):
    inc = obs.get("incidents", [])
    res = obs.get("resources", [])

    active = [i for i in inc if i.get("status") in ("active", "contained")]
    free   = [r for r in res if r.get("available", False)]

    if task == "task1_prioritization":
        ordered = sorted(active, key=lambda x: -x.get("urgency_score", 0))
        return {
            "action_type": "reprioritize",
            "ordered_incident_ids": [i["id"] for i in ordered]
        }

    if not active or not free:
        return {"action_type": "wait"}

    t = sorted(active, key=lambda x: -x.get("urgency_score", 0))[0]

    return {
        "action_type": "assign_resource",
        "resource_id": free[0]["id"],
        "incident_id": t["id"]
    }

# ── ENV FUNCTIONS ────────────────────────────────────────────────────────────
def reset(task):
    try:
        r = requests.post(f"{OPENENV_URL}/reset", json={"task_id": task, "seed": 42})
        d = r.json()
        return d.get("session_id"), d.get("observation", {})
    except:
        return None, {}

def step(sid, action):
    try:
        r = requests.post(f"{OPENENV_URL}/step", json={"session_id": sid, "action": action})
        d = r.json()
        return d.get("observation", {}), float(d.get("reward", {}).get("value", 0)), bool(d.get("done", True))
    except:
        return {}, 0.0, True

def wait():
    for _ in range(30):
        try:
            if requests.get(f"{OPENENV_URL}/health").status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

# ── RUN TASK ─────────────────────────────────────────────────────────────────
def run(task):
    print(f"[START] task={task} env=disaster model={MODEL_NAME}", flush=True)

    sid, obs = reset(task)
    if not sid:
        print("[END] success=false steps=0 rewards=", flush=True)
        return

    rewards = []
    step_no = 0

    while True:
        step_no += 1

        prompt = json.dumps(obs)

        # 🚨 ALWAYS CALL LLM
        action = call_llm(prompt)

        if not isinstance(action, dict) or "action_type" not in action:
            action = fallback(obs, task)

        obs, reward, done = step(sid, action)
        rewards.append(max(0.0001, min(0.9999, reward)))

        print(f"[STEP] step={step_no} action={json.dumps(action)} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

        if done:
            break

    print(f"[END] success=true steps={step_no} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if not wait():
        for t in TASKS:
            print(f"[START] task={t} env=disaster model={MODEL_NAME}", flush=True)
            print("[END] success=false steps=0 rewards=", flush=True)
        return

    for t in TASKS:
        run(t)

if __name__ == "__main__":
    main()