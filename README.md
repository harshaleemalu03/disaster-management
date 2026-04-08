---
title: Disaster Response Coordination System
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
sdk_version: "1.0"
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

Emergency response systems operate under extreme pressure where **every second impacts human lives**.

This environment simulates real-world disaster coordination challenges:
- Managing multiple simultaneous incidents  
- Allocating heterogeneous emergency resources  
- Handling geographic and traffic constraints  
- Ensuring fairness across urban and rural zones  

Traditional systems rely on heuristics that fail under surge conditions.  
This environment enables AI agents to learn optimal multi-step decision-making.

---

## 🧠 Why This Matters

This is a sequential decision-making problem under uncertainty:
- Immediate actions affect future availability  
- Ignoring incidents leads to escalation or cascade spread  
- Greedy strategies fail → requires planning  

Ideal for Reinforcement Learning and agent evaluation.

---

## 🏗️ Architecture

```
FastAPI Server (OpenEnv API)
        ↓
DisasterResponseEnv
        ↓
Task Environments (3 Levels)
        ↓
World Engine (Simulation + Physics)
```

---

## 🎯 Tasks Overview

### Task 1 — Incident Prioritization (Easy)
- Rank incidents by urgency  
- Graded using Kendall-tau + fairness  
- Baseline Score: 1.00  

### Task 2 — Resource Allocation (Medium)
- Assign resources with travel delays  
- Includes real-world constraints  
- Baseline Score: 0.61  

### Task 3 — Dynamic Coordination (Hard)
- Incident spawning  
- Fire escalation  
- Cascade spread  
- Multi-step planning required  
- Baseline Score: 0.35  

---

## 🔥 Key Innovations

- Fire Cascade Spread → uncontrolled fires create new incidents  
- Real Travel Delays → distance + traffic based  
- Fairness-Aware Dispatch → Gini coefficient based reward  
- Live Visualization → grid + HTML UI  

---

## 💰 Reward Function

Reward ∈ [0,1] combining:
- Life saving (35%)  
- Response time (25%)  
- Resource efficiency (20%)  
- Fairness (10%)  
- Penalties (delay, wrong actions, cascade)  

Encourages realistic and balanced decision-making.

---

## 📊 Evaluation

Each task evaluates:
- Coverage  
- Accuracy  
- Time efficiency  
- Resource efficiency  
- Fairness  

All graders are deterministic and reproducible.

---

## 🤖 Baseline Agent

Heuristic:
- Sort by urgency  
- Assign nearest compatible resource  

Scores:
- Task 1 → 1.00  
- Task 2 → 0.61  
- Task 3 → 0.35  
- Overall → 0.65  

---

## 📺 Live Demo

UI loads automatically on homepage.

Or manually:
```
/view/{session_id}
```

---

## 🚀 Quick Start

```
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

---

## 🐳 Docker

```
docker build -t disaster-response .
docker run -p 7860:7860 disaster-response
```

---

## 🔌 API Overview

- POST /reset  
- POST /step  
- GET /state/{id}  
- GET /grade/{id}  
- GET /view/{id}  

---

## 📁 Project Structure

```
env/
  models.py
  world.py
  environment.py
  tasks/
  graders/

server.py
inference.py
openenv.yaml
Dockerfile
```

---

## ⚡ Performance

- < 50 ms per step  
- < 3 sec full inference  
- Deterministic and reproducible  
- Works on 2 vCPU / 8GB RAM  

---

## 🏁 Submission Highlights

- Real-world disaster management simulation  
- Multi-task RL environment  
- Strong reward design  
- OpenEnv compliant  
- Interactive UI  
- Fully deployed on Hugging Face  