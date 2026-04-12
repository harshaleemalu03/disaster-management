---
title: Disaster Response Coordination System
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_file: server.py
pinned: false
---

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

Modern CAD systems fail under surge conditions.  
**This environment benchmarks AI agents on these failure cases.**

---

## 🏗️ Architecture

```
FastAPI Server (port 7860)
        │
DisasterResponseEnv
        │
 Task1 | Task2 | Task3
        │
   WorldEngine
```

---

## 🎯 Three Tasks

### Task 1 — Incident Prioritization `[EASY]`
- Rank incidents by urgency  
- Baseline: **1.00**

### Task 2 — Resource Allocation `[MEDIUM]`
- Assign resources with travel delays  
- Baseline: **0.61**

### Task 3 — Dynamic Coordination `[HARD]`
- Full simulation with cascading events  
- Baseline: **0.27**

---

## 🔥 Key Features

- 🔥 Fire cascade spread  
- 🚑 Real distance-based travel  
- ⚖️ Fairness-aware dispatch (Gini coefficient)  
- 📊 Dense reward system  
- 🌐 Live visualization dashboard  

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
python inference.py
```

---

## 📺 Endpoints

- `/` → Landing page  
- `/demo` → Live simulation  
- `/docs` → API docs  
- `/view/{session_id}` → Dashboard  
- `/visualize/{session_id}` → Text grid  

---

## 📁 Structure

```
disaster_response/
├── env/
├── server.py
├── inference.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Performance

- < 50 ms / step  
- < 3 sec full run  
- Deterministic & reproducible  

---

## 📜 License
MIT License
