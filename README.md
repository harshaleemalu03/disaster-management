<div align="center">

# 🚨 Disaster Response Coordination System
### OpenEnv v2.1 — Hackathon Submission

*High-fidelity AI environment for emergency dispatch coordination*
*Real travel delays · Fire cascade spread · Fairness-aware dispatch · Live visualization*

</div>

---

## 🌍 Real-World Motivation

Every 60 seconds of delayed emergency response increases mortality risk. Real dispatch centers must simultaneously manage:

- **Dozens of concurrent incidents** across urban, suburban, and rural areas
- **Heterogeneous fleets** — ambulances, fire trucks, rescue teams, helicopters — with different speeds and specializations
- **Geographic constraints** — urban traffic congestion, rural road conditions, cross-terrain helicopter routes
- **Equity obligations** — rural communities must not be systematically deprioritized

Modern CAD (Computer-Aided Dispatch) systems use heuristics that fail under surge conditions (earthquakes, multi-fire events). **This environment benchmarks AI agents on exactly these failure cases.**

### Why RL is Needed

The optimal dispatch policy requires **multi-step lookahead**:
- Sending the nearest ambulance to a low-severity call may leave a critical patient without coverage 3 steps later
- Allowing a fire to escalate unchecked triggers cascade spread, multiplying the incident load
- Neglecting rural incidents while optimizing for urban response inflates the Gini penalty

This is a **combinatorial optimization problem under stochastic evolution** — tractable for RL, intractable for simple greedy rules.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│               FastAPI OpenEnv Server (port 7860)          │
│  /reset  /step  /grade  /render  /visualize  /view       │
└────────────────────┬─────────────────────────────────────┘
                     │
         DisasterResponseEnv (environment.py)
                     │
        ┌────────────┼────────────┐
        │            │            │
  Task1Env      Task2Env      Task3Env
  (easy)        (medium)      (hard)
        │            │            │
        └────────────┼────────────┘
                     │
              WorldEngine (world.py)
          ┌──────────┴──────────┐
          │  Physics            │  Visualization
          │  travel_steps()     │  build_text_grid()
          │  escalate()         │  build_html_view()
          │  check_cascade()    │
          │  compute_fairness() │
          └─────────────────────┘
```

---

## 🎯 Three Tasks

### Task 1 — Incident Prioritization `[EASY]` — Baseline: 1.00

Six incidents across all three zone types. Each step the agent issues one `reprioritize` action.

**Ground-truth urgency formula:**
```
urgency = 0.45 × severity
         + 0.25 × min(people_affected / 200, 1.0)
         + 0.20 × (1 - steps_to_expiry / 16)
         + 0.10 × population_density
```

**Grader:** `0.80 × Kendall-tau + 0.20 × fairness_score + consistency_bonus`

---

### Task 2 — Resource Allocation with Travel Delays `[MEDIUM]` — Baseline: 0.61

5 typed incidents · 6 typed resources · real travel time.

**Resource state machine:**
```
FREE ──[dispatch]──► IN_TRANSIT ──[arrive]──► ON_SCENE ──[resolve]──► FREE
```

Travel time = `⌈ dist_km / (speed / traffic_factor) ⌉` steps

| Resource | Speed | Urban effective |
|---|---|---|
| Ambulance | 8 km/step | 5.0 km/step |
| Fire Truck | 7 km/step | 4.4 km/step |
| Rescue Team | 5 km/step | 3.1 km/step |
| Police | 9 km/step | 5.6 km/step |
| **Helicopter** | **25 km/step** | **25.0 km/step** (no traffic) |

**Grader weights:** coverage(30%) + accuracy(25%) + efficiency(20%) + time(15%) + fairness(10%)

---

### Task 3 — Dynamic Multi-step Coordination `[HARD]` — Baseline: 0.27

The full simulation. Every step:

| Event | Rate |
|---|---|
| New incident spawns | p = 0.28 |
| Fire escalation | +0.06 severity/step |
| Medical escalation | +0.05 severity/step |
| Fire cascade spread | p = 0.40 (when sev ≥ 0.80, unattended) |
| Medical expiry | 8 steps |
| Fire expiry | 10 steps |

**Grader:** coverage(30%) + severity(25%) + time(20%) + efficiency(15%) + fairness(10%) − cascade_penalty

---

## 🔥 Novel Features

### 1 · Fire Cascade Spread
Unattended fires at severity ≥ 0.80 spread to nearby areas (≤15 km) with p=0.40/step:
```
Primary fire (sev=0.85, unattended) → cascade → Secondary fire spawned at (x±dist, y±dist)
```
Each cascade event: spawns new `INC-XXX`, penalises grader by −0.04 (max −0.20 total).

### 2 · Real Distance-Based Travel
```python
travel_steps = ceil(dist_km / (resource_speed / zone_traffic_factor))
# Example: ambulance 30 km through urban area
# = ceil(30 / (8 / 1.6)) = ceil(30 / 5) = 6 steps = 30 real minutes
```

### 3 · Fairness-Aware Dispatch (Novel)
```python
# Gini coefficient across zone response rates
rates = [urban_resolved/urban_total, suburban_resolved/suburban_total, rural_resolved/rural_total]
gini  = mean_abs_diff(rates) / (2 × n × mean(rates))
fairness_reward = 1.0 − gini   # rewarded in every step
```

### 4 · Live Visualization (Option A + B)
```bash
# Text grid (Option A)
curl http://localhost:7860/visualize/{session_id}

# HTML live map (Option B) — open in browser
open http://localhost:7860/view/{session_id}
```

---

## 💰 Exact Reward Formula

```python
# Dense reward ∈ [0.0, 1.0] per step
value = clip(
    0.35 * life_saving_score          # = severity × min(people/100, 1.0)
  + 0.25 * response_time_score        # = 1 − dist_km / 141.42
  + 0.20 * resource_efficiency_score  # = COMPAT[resource_type][incident_type]
  + 0.10 * fairness_score             # = 1 − gini_coefficient
  − 0.05 * delay_penalty              # = idle_fraction when active incidents exist
  − 0.03 * idle_resource_penalty      # = idle_fraction (always)
  − 0.02 * wrong_assignment_penalty   # = 1.0 for invalid actions
  − 0.05 * cascade_penalty,           # = min(cascade_events × 0.2, 1.0)
  0.0, 1.0
)
```

**Rationale:**
- `life_saving (35%)`: primary objective — severity × human impact
- `response_time (25%)`: speed determines medical/fire outcomes
- `resource_efficiency (20%)`: type match prevents resource waste
- `fairness (10%)`: prevents systematic rural neglect

---

## 📊 Grader Breakdowns

All graders return `{efficiency, accuracy, time_score, coverage, fairness}` in [0,1]:

| Component | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| **coverage** | 1.0 (all seen) | resolution rate | resolution rate |
| **accuracy** | Kendall-tau | severity rescue % | severity rescue % |
| **time_score** | 0 | fast response | response vs expiry |
| **efficiency** | 0 | compat × proximity | compat × proximity |
| **fairness** | 1 − gini | 1 − gini | 1 − gini |

---

## 🤖 Baseline Agent Performance

Heuristic: sort by `urgency_score` DESC → assign nearest compatible free resource.

| Task | Score | Resolved | Lives Saved |
|---|---|---|---|
| task1_prioritization | **1.00** | N/A | N/A |
| task2_resource_allocation | **0.61** | 3/5 | 253 |
| task3_dynamic_coordination | **0.35** | 4/9 | 347 |
| **Overall** | **0.65** | | |

---

## 📺 Visualization

### Text Grid (`/visualize/{session_id}`)
```
╔══ DISASTER MAP: task3_dynamic_coordination │ Step 7/30 ══╗
Active:4  Resolved:2  Failed:1  Lives:347
[ ] [ ] [⚡] [ ] [ ] [ ] [ ] [🌊] [ ] [ ]
[ ] [ ] [ ] [ ] [🚑] [ ] [ ] [ ] [ ] [ ]
[ ] [🔥] [ ] [ ] [ ] [ ] [🚒] [ ] [ ] [ ]
...
Legend: 🔥Fire 🏥Medical 🌊Flood ⚡Quake 💥Accident ☣️Hazmat | 🚑Ambulance ...
```

### HTML Live Map (`/view/{session_id}`)
Full-colour grid rendered in browser. Auto-refreshes every 5 seconds.
Cell colours: incident type colour = active incident. 🟢 green = free resource. 🟠 orange = in-transit. 🔴 red = busy.

---

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start server
uvicorn server:app --host 0.0.0.0 --port 7860

# 3. Run baseline inference
API_BASE_URL=http://localhost:7860 python inference.py

# 4. Watch live in browser (while inference runs)
open http://localhost:7860/view/<session_id>

# 5. With LLM
API_BASE_URL=http://localhost:7860 MODEL_NAME=gpt-4o HF_TOKEN=sk-... python inference.py
```

### Docker
```bash
docker build -t disaster-response .
docker run -p 7860:7860 disaster-response
```

### API Quick Reference
```python
import requests
BASE = "http://localhost:7860"

# Reset
r   = requests.post(f"{BASE}/reset", json={"task_id": "task3_dynamic_coordination", "seed": 42})
sid = r.json()["session_id"]

# Text grid
print(requests.get(f"{BASE}/visualize/{sid}").text)

# HTML map — open in browser
print(f"http://localhost:7860/view/{sid}")

# Step
action = {"action_type": "assign_resource", "resource_id": "RES-05", "incident_id": "INC-001"}
r = requests.post(f"{BASE}/step", json={"session_id": sid, "action": action})
print(f"reward={r.json()['reward']['value']:.4f}")

# Grade
print(requests.get(f"{BASE}/grade/{sid}").json()["grader_score"])
```

---

## 📁 Project Structure

```
disaster_response/
├── env/
│   ├── models.py          Pydantic typed models (all data structures)
│   ├── world.py           Physics engine + visualization builders
│   ├── environment.py     OpenEnv interface dispatcher
│   ├── _pydantic_shim.py  Fallback for pydantic-free environments
│   ├── tasks/
│   │   ├── task1_prioritization.py
│   │   ├── task2_resource_allocation.py
│   │   └── task3_dynamic_coordination.py
│   └── graders/
│       └── task_graders.py
├── server.py              FastAPI REST server (+ /visualize + /view)
├── inference.py           Heuristic + LLM baseline agent
├── openenv.yaml           Full environment specification
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Performance

- Episode runtime: < 50 ms/step on single CPU core
- Full 3-task inference: < 3 seconds (heuristic mode)
- Memory: ~65 MB
- All RNG seeded → fully deterministic and reproducible
- Designed for 2 vCPU / 8 GB RAM (HF Spaces free tier)

---

*MIT License*
