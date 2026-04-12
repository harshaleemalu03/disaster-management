"""
Task 1 (Easy) — Incident Prioritization
========================================
Agent must order incidents by urgency each step.
Ground-truth urgency uses: severity, people_affected, expiry pressure, zone density.
Graded via Kendall-tau correlation.

Key upgrades:
  - Incidents have real geographic zones and population density
  - Urgency formula matches real triage protocols (START triage scoring)
  - Consistency bonus rewards stable, correct ordering across steps
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    ActionType, ActionWrapper, EpisodeResult, GraderBreakdown, Incident,
    IncidentStatus, IncidentType, Location, Observation, Resource, ResourceType,
    Reward, RewardBreakdown, StepLog, TaskDifficulty, ZoneType,
    EXPIRY_STEPS, FairnessMetrics,
)
from env.world import WorldEngine, compute_fairness

TASK_ID       = "task1_prioritization"
MAX_TIMESTEPS = 10


def _make_incidents(rng: random.Random) -> List[Incident]:
    configs = [
        (IncidentType.EARTHQUAKE, 0.95, 200, Location(x=15.0, y=20.0, zone=ZoneType.RURAL),    0.3, ZoneType.RURAL),
        (IncidentType.FIRE,       0.88, 75,  Location(x=55.0, y=60.0, zone=ZoneType.URBAN),    0.9, ZoneType.URBAN),
        (IncidentType.HAZMAT,     0.82, 45,  Location(x=70.0, y=25.0, zone=ZoneType.SUBURBAN), 0.6, ZoneType.SUBURBAN),
        (IncidentType.FLOOD,      0.73, 120, Location(x=30.0, y=75.0, zone=ZoneType.RURAL),    0.2, ZoneType.RURAL),
        (IncidentType.MEDICAL,    0.60, 30,  Location(x=80.0, y=80.0, zone=ZoneType.URBAN),    0.9, ZoneType.URBAN),
        (IncidentType.ACCIDENT,   0.51, 18,  Location(x=45.0, y=40.0, zone=ZoneType.SUBURBAN), 0.5, ZoneType.SUBURBAN),
    ]
    incidents = []
    for idx, (itype, sev, ppl, loc, popd, zone) in enumerate(configs):
        s = round(min(max(sev + rng.uniform(-0.03, 0.03), 0.1), 1.0), 3)
        itype_str = itype.value if hasattr(itype, "value") else itype
        exp = EXPIRY_STEPS.get(itype_str, 12)
        incidents.append(Incident(
            id=f"INC-{idx+1:03d}",
            incident_type=itype,
            severity=s,
            people_affected=max(ppl + rng.randint(-5, 5), 1),
            location=loc,
            time_since_report=round(rng.uniform(1.0, 5.0), 1),
            status=IncidentStatus.ACTIVE,
            required_resource_types=[ResourceType.RESCUE_TEAM],
            base_resolution_steps=5,
            steps_to_expiry=exp,
            zone=zone,
            population_density=round(popd + rng.uniform(-0.05, 0.05), 2),
        ))
    return incidents


def _make_resources(rng: random.Random) -> List[Resource]:
    return [
        Resource(id="RES-01", resource_type=ResourceType.AMBULANCE,
                 location=Location(x=50.0, y=50.0, zone=ZoneType.URBAN)),
        Resource(id="RES-02", resource_type=ResourceType.FIRE_TRUCK,
                 location=Location(x=20.0, y=30.0, zone=ZoneType.SUBURBAN)),
        Resource(id="RES-03", resource_type=ResourceType.RESCUE_TEAM,
                 location=Location(x=70.0, y=60.0, zone=ZoneType.RURAL)),
    ]


def _kendall_tau(agent_order: List[str], optimal_order: List[str]) -> float:
    """Normalized Kendall-tau in [0,1]. 1=perfect, 0.5=random, 0=reversed."""
    n = len(optimal_order)
    if n <= 1:
        return 1.0
    agent_pos = {iid: idx for idx, iid in enumerate(agent_order)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            ia, ib = optimal_order[i], optimal_order[j]
            if ia not in agent_pos or ib not in agent_pos:
                discordant += 1
                continue
            concordant += 1 if agent_pos[ia] < agent_pos[ib] else -1
            if agent_pos[ia] < agent_pos[ib]:
                concordant += 0   # already counted above
    # Redo correctly
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            ia, ib = optimal_order[i], optimal_order[j]
            ap_ia = agent_pos.get(ia, n + 1)
            ap_ib = agent_pos.get(ib, n + 1)
            if ap_ia < ap_ib:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return round((concordant - discordant) / total * 0.5 + 0.5, 4)


class Task1Environment:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng  = random.Random(seed)
        self._incidents: List[Incident]  = []
        self._resources: List[Resource]  = []
        self._timestep:  int             = 0
        self._cumulative_reward: float   = 0.0
        self._step_logs: List[StepLog]   = []
        self._tau_scores: List[float]    = []
        self._done:      bool            = False

    def reset(self) -> Observation:
        self._rng      = random.Random(self._seed)
        self._incidents = _make_incidents(self._rng)
        self._resources = _make_resources(self._rng)
        self._timestep  = 0
        self._cumulative_reward = 0.0
        self._step_logs = []
        self._tau_scores = []
        self._done      = False
        return self._obs()

    def step(self, action: ActionWrapper) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode done; call reset().")
        self._timestep += 1

        # Optimal ordering (ground truth)
        active_ids = [i.id for i in self._incidents if i.is_active]
        optimal    = sorted(active_ids, key=lambda iid: -self._inc(iid).urgency_score())

        reward = self._apply(action, optimal)
        self._cumulative_reward = round(self._cumulative_reward + reward.value, 4)
        self._tau_scores.append(reward.value)

        fairness = compute_fairness(self._incidents)
        self._step_logs.append(StepLog(
            step=self._timestep,
            action_type=action.action_type.value if hasattr(action.action_type,"value") else action.action_type,
            action_detail=f"reprioritize({','.join((action.ordered_incident_ids or [])[:3])}...)",
            reward=reward.value,
            incidents_active=len(active_ids),
            incidents_resolved=sum(1 for i in self._incidents if i.is_resolved),
            resources_busy=sum(1 for r in self._resources if not r.available),
            explanation=reward.explanation,
        ))

        done = self._timestep >= MAX_TIMESTEPS
        self._done = done
        info = {"optimal_order": optimal, "tau_score": reward.value}
        return self._obs(), reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "incidents": [i.model_dump() for i in self._incidents],
            "resources":  [r.model_dump() for r in self._resources],
            "timestep": self._timestep,
            "tau_scores": self._tau_scores,
        }

    def grade(self) -> EpisodeResult:
        scores = self._tau_scores
        avg    = sum(scores) / len(scores) if scores else 0.0
        # Consistency bonus: reward low variance
        if len(scores) > 1:
            var   = sum((s - avg)**2 for s in scores) / len(scores)
            bonus = max(0.0, 0.05 * (1.0 - var * 10))
        else:
            bonus = 0.0
        grader = round(min(avg + bonus, 1.0), 4)

        bd = GraderBreakdown(
            efficiency=0.0,
            accuracy=round(avg, 4),
            time_score=0.0,
            coverage=1.0,
            fairness=round(1.0 - compute_fairness(self._incidents).gini_coefficient, 4),
        )
        return EpisodeResult(
            task_id=TASK_ID, difficulty=TaskDifficulty.EASY,
            total_steps=self._timestep, total_reward=round(self._cumulative_reward, 4),
            grader_score=grader, breakdown=bd,
            resolved_incidents=0, failed_incidents=0,
            resource_utilization=0.0, average_response_time=0.0,
            total_lives_saved=0, cascade_events_triggered=0,
            details={"tau_scores": scores, "consistency_bonus": round(bonus, 4)},
        )

    # -- internals --

    def _inc(self, iid: str) -> Incident:
        return next(i for i in self._incidents if i.id == iid)

    def _obs(self) -> Observation:
        fairness = compute_fairness(self._incidents)
        return Observation(
            incidents=self._incidents, resources=self._resources,
            timestep=self._timestep, task_id=TASK_ID,
            task_difficulty=TaskDifficulty.EASY, max_timesteps=MAX_TIMESTEPS,
            resolved_count=sum(1 for i in self._incidents if i.is_resolved),
            active_count=sum(1 for i in self._incidents if i.is_active),
            cumulative_reward=self._cumulative_reward,
            fairness=fairness, step_log=self._step_logs,
        )

    def _apply(self, action: ActionWrapper, optimal: List[str]) -> Reward:
        at = action.action_type.value if hasattr(action.action_type,"value") else action.action_type
        if at == "reprioritize" and action.ordered_incident_ids:
            agent = [x for x in action.ordered_incident_ids if x in {i.id for i in self._incidents}]
            missing = [x for x in optimal if x not in agent]
            agent = agent + missing
            tau = _kendall_tau(agent, optimal)
            fairness_s = 1.0 - compute_fairness(self._incidents).gini_coefficient
            bd = RewardBreakdown(
                life_saving_score=tau,
                response_time_score=0.0,
                resource_efficiency_score=0.0,
                fairness_score=round(fairness_s, 4),
            )
            # Override formula for task 1: reward = 0.80*tau + 0.20*fairness
            val = round(min(0.80 * tau + 0.20 * fairness_s, 1.0), 4)
            return Reward(value=val, breakdown=bd, explanation=f"kendall_tau={tau:.3f} fairness={fairness_s:.3f}")
        elif at == "wait":
            bd = RewardBreakdown(delay_penalty=0.5)
            return Reward(value=bd.compute_value(), breakdown=bd, explanation="wait penalized in task1")
        else:
            return Reward.invalid(f"invalid action type '{at}' for task1")
