---
title: Disaster Response Coordination System
emoji: "🚨"
colorFrom: blue
colorTo: red
sdk: docker
app_file: server.py
pinned: false
---

<div align="center">

# 🚨 Disaster Response Coordination System

### OpenEnv Hackathon Submission

*High-fidelity AI environment for emergency dispatch coordination*
*Real travel delays · Fire cascade spread · Fairness-aware dispatch · Live visualization*

</div>

---

## 🌍 Real-World Motivation

Every 60 seconds of delayed emergency response increases mortality risk. Real dispatch centers must simultaneously manage:

* **Dozens of concurrent incidents** across urban, suburban, and rural areas
* **Heterogeneous fleets** — ambulances, fire trucks, rescue teams, helicopters — with different speeds and specializations
* **Geographic constraints** — urban traffic congestion, rural road conditions, cross-terrain helicopter routes
* **Equity obligations** — rural communities must not be systematically deprioritized

Modern CAD systems fail under surge conditions.
👉 This environment benchmarks AI agents under real-world failure scenarios.

---

## 🧠 Why RL is Needed

The optimal dispatch policy requires **multi-step reasoning**:

* Greedy dispatch can leave critical patients unattended later
* Fire escalation can cascade into multiple incidents
* Ignoring rural zones increases inequality penalties

👉 This is a **combinatorial optimization problem under uncertainty**

---

## 🏗️ Architecture

```
FastAPI Server (server.py)
   ├── /reset /step /grade /visualize /view
   │
   └── DisasterResponseEnv
          ├── Task1 (easy)
          ├── Task2 (medium)
          └── Task3 (hard)
                 │
                 └── WorldEngine (physics + simulation)
```

---

## 🎯 Tasks

### 🟢 Task 1 — Incident Prioritization

* Rank incidents by urgency
* Grader: Kendall-tau + fairness

---

### 🟡 Task 2 — Resource Allocation

* Assign limited emergency resources
* Includes travel delays + constraints

---

### 🔴 Task 3 — Dynamic Coordination

* Multi-step evolving environment
* Fire spread, new incidents, expiries

---

## 🔥 Key Features

### 🚒 Fire Cascade Spread

* High severity fires spawn new incidents
* Penalizes poor response

---

### 🚑 Real Travel Delays

* Distance + traffic based dispatch timing

---

### ⚖️ Fairness-Aware Dispatch

* Uses **Gini coefficient**
* Prevents rural neglect

---

### 🗺️ Live Visualization

* `/visualize` → text grid
* `/view` → browser map

---

## 💰 Reward Function

* Life saving (35%)
* Response time (25%)
* Resource efficiency (20%)
* Fairness (10%)
* Penalties for delay, idle, wrong actions

👉 Output always ∈ [0.0, 1.0]

---

## 📊 Graders

Each task returns score ∈ [0,1]:

* Coverage
* Accuracy
* Time
* Efficiency
* Fairness

---

## 🤖 Baseline Performance

| Task    | Score |
| ------- | ----- |
| Easy    | 1.00  |
| Medium  | 0.61  |
| Hard    | 0.35  |
| Overall | 0.65  |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

---

## 🐳 Docker

```bash
docker build -t disaster-response .
docker run -p 7860:7860 disaster-response
```

---

## 🔌 API Endpoints

* `/reset`
* `/step`
* `/grade`
* `/visualize/{session_id}`
* `/view/{session_id}`

---

## 📁 Project Structure

```
env/
tasks/
graders/
server.py
inference.py
openenv.yaml
Dockerfile
README.md
```

---

## ⚡ Performance

* < 50 ms per step
* < 3 sec full inference
* deterministic (seeded)

---

## ✅ OpenEnv Compliance

* ✔ step(), reset(), state() implemented
* ✔ Typed models (Pydantic)
* ✔ openenv.yaml included
* ✔ 3 tasks with graders
* ✔ reward in [0,1]
* ✔ reproducible inference

---

## 🏁 Submission Info

* 🔗 Hugging Face Space:
  https://huggingface.co/spaces/harshaleemalu03/disaster-management

* 🔗 GitHub Repo:
  (add your repo link here)

---

## 💡 Summary

This environment simulates **real-world disaster coordination**, combining:

* logistics
* fairness
* dynamic uncertainty

👉 Designed to evaluate **next-generation AI agents** in critical systems.

---
