"""
Task 2 (Medium) — Resource Allocation with Travel Delays
=========================================================
Agent assigns 6 typed resources to 5 incidents, accounting for:
  - Type compatibility matrix
  - Geographic distance → real travel time (not instant)
  - Zone traffic factors (urban congestion)
  - Resources arrive after transit, then work on-scene
  - Incident requires correct resource types to resolve

Upgrades:
  - Resources are in-transit before they arrive (realistic delay)
  - Incidents must accumulate treatment steps after arrival
  - Wrong-type assignments waste resources (low compatibility = longer on-scene)
  - Fairness reward penalizes ignoring rural incidents
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

TASK_ID       = "task2_resource_allocation"
MAX_TIMESTEPS = 20


def _make_incidents(rng: random.Random) -> List[Incident]:
    configs = [
        (IncidentType.FIRE,       0.88, 60,  Location(x=12.0, y=10.0, zone=ZoneType.RURAL),    [ResourceType.FIRE_TRUCK],                          0.2, ZoneType.RURAL),
        (IncidentType.MEDICAL,    0.72, 28,  Location(x=75.0, y=72.0, zone=ZoneType.URBAN),    [ResourceType.AMBULANCE],                           0.9, ZoneType.URBAN),
        (IncidentType.EARTHQUAKE, 0.94, 150, Location(x=50.0, y=82.0, zone=ZoneType.SUBURBAN), [ResourceType.RESCUE_TEAM, ResourceType.HELICOPTER], 0.5, ZoneType.SUBURBAN),
        (IncidentType.ACCIDENT,   0.58, 12,  Location(x=30.0, y=48.0, zone=ZoneType.SUBURBAN), [ResourceType.AMBULANCE,   ResourceType.POLICE],     0.6, ZoneType.SUBURBAN),
        (IncidentType.HAZMAT,     0.80, 40,  Location(x=68.0, y=30.0, zone=ZoneType.URBAN),    [ResourceType.FIRE_TRUCK],                          0.8, ZoneType.URBAN),
    ]
    incidents = []
    for idx, (itype, sev, ppl, loc, req, popd, zone) in enumerate(configs):
        s       = round(min(max(sev + rng.uniform(-0.03, 0.03), 0.1), 1.0), 3)
        itype_s = itype.value if hasattr(itype, "value") else itype
        exp     = EXPIRY_STEPS.get(itype_s, 12)
        incidents.append(Incident(
            id=f"INC-{idx+1:03d}",
            incident_type=itype,
            severity=s,
            people_affected=max(ppl + rng.randint(-5, 5), 1),
            location=loc,
            time_since_report=round(rng.uniform(0.5, 3.0), 1),
            status=IncidentStatus.ACTIVE,
            required_resource_types=req,
            base_resolution_steps=4,
            steps_to_expiry=exp,
            zone=zone,
            population_density=round(popd + rng.uniform(-0.05, 0.05), 2),
        ))
    return incidents


def _make_resources(rng: random.Random) -> List[Resource]:
    configs = [
        ("RES-01", ResourceType.AMBULANCE,   Location(x=50.0, y=50.0, zone=ZoneType.URBAN)),
        ("RES-02", ResourceType.FIRE_TRUCK,  Location(x=18.0, y=15.0, zone=ZoneType.SUBURBAN)),
        ("RES-03", ResourceType.RESCUE_TEAM, Location(x=55.0, y=78.0, zone=ZoneType.SUBURBAN)),
        ("RES-04", ResourceType.POLICE,      Location(x=33.0, y=45.0, zone=ZoneType.SUBURBAN)),
        ("RES-05", ResourceType.HELICOPTER,  Location(x=60.0, y=58.0, zone=ZoneType.URBAN)),
        ("RES-06", ResourceType.AMBULANCE,   Location(x=78.0, y=70.0, zone=ZoneType.URBAN)),
    ]
    return [Resource(id=rid, resource_type=rtype, location=loc) for rid, rtype, loc in configs]


class Task2Environment:
    def __init__(self, seed: int = 42):
        self._seed   = seed
        self._rng    = random.Random(seed)
        self._incidents: List[Incident] = []
        self._resources: List[Resource] = []
        self._timestep:  int            = 0
        self._cumulative_reward: float  = 0.0
        self._step_logs: List[StepLog]  = []
        self._assignment_log: List[Dict[str, Any]] = []
        self._resolved_log:   List[Dict[str, Any]] = []
        self._done:      bool           = False

    def reset(self) -> Observation:
        self._rng        = random.Random(self._seed)
        self._incidents  = _make_incidents(self._rng)
        self._resources  = _make_resources(self._rng)
        self._timestep   = 0
        self._cumulative_reward = 0.0
        self._step_logs  = []
        self._assignment_log = []
        self._resolved_log   = []
        self._done       = False
        return self._obs()

    def step(self, action: ActionWrapper) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode done; call reset().")
        self._timestep += 1

        # World dynamics
        WorldEngine.age_incidents(self._incidents, self._timestep)
        WorldEngine.escalate_incidents(self._incidents)
        WorldEngine.advance_resources(self._resources)
        newly_resolved = WorldEngine.check_resolutions(self._incidents, self._resources, self._timestep)
        newly_failed   = WorldEngine.expire_incidents(self._incidents, self._timestep)

        for iid in newly_resolved:
            inc = self._inc(iid)
            self._resolved_log.append({
                "incident_id": iid, "severity": inc.severity,
                "people_affected": inc.people_affected,
                "resolution_step": self._timestep,
                "time_since_report": inc.time_since_report,
                "zone": inc.zone.value if hasattr(inc.zone, "value") else inc.zone,
            })

        fairness = compute_fairness(self._incidents)
        reward   = self._apply(action, fairness)
        self._cumulative_reward = round(self._cumulative_reward + reward.value, 4)

        at_str = action.action_type.value if hasattr(action.action_type,"value") else action.action_type
        detail = f"{at_str}({action.resource_id}→{action.incident_id})" if action.resource_id else at_str
        self._step_logs.append(StepLog(
            step=self._timestep, action_type=at_str, action_detail=detail,
            reward=reward.value,
            incidents_active=sum(1 for i in self._incidents if i.is_active),
            incidents_resolved=sum(1 for i in self._incidents if i.is_resolved),
            resources_busy=sum(1 for r in self._resources if not r.available),
            explanation=reward.explanation,
        ))

        all_resolved = all(not i.is_active for i in self._incidents)
        done = self._timestep >= MAX_TIMESTEPS or all_resolved
        self._done = done
        info = {"newly_resolved": newly_resolved, "newly_failed": newly_failed}
        return self._obs(), reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "incidents": [i.model_dump() for i in self._incidents],
            "resources":  [r.model_dump() for r in self._resources],
            "timestep": self._timestep, "resolved_log": self._resolved_log,
            "assignment_log": self._assignment_log,
        }

    def grade(self) -> EpisodeResult:
        total      = len(self._incidents)
        resolved   = sum(1 for i in self._incidents if i.is_resolved)
        failed     = sum(1 for i in self._incidents if
                         (i.status.value if hasattr(i.status,"value") else i.status) == "failed")

        resolution_rate = resolved / total if total > 0 else 0.0

        max_sev = sum(i.severity for i in self._incidents)
        rescued_sev = sum(log["severity"] for log in self._resolved_log)
        severity_score = rescued_sev / max_sev if max_sev > 0 else 0.0

        if self._assignment_log:
            avg_compat = sum(a["compat"] for a in self._assignment_log) / len(self._assignment_log)
            avg_dist   = sum(a["dist_km"] for a in self._assignment_log) / len(self._assignment_log)
            dist_score = max(0.0, 1.0 - avg_dist / 100.0)
            efficiency = 0.6 * avg_compat + 0.4 * dist_score
        else:
            efficiency = 0.0

        total_assign = sum(r.total_assignments for r in self._resources)
        max_possible = len(self._resources) * 3
        utilization  = min(total_assign / max_possible, 1.0) if max_possible > 0 else 0.0

        fairness = compute_fairness(self._incidents)
        fairness_score = max(0.0, 1.0 - fairness.gini_coefficient)

        if self._resolved_log:
            avg_rt = sum(log["time_since_report"] for log in self._resolved_log) / len(self._resolved_log)
            time_score = max(0.0, 1.0 - avg_rt / 12.0)
        else:
            avg_rt, time_score = float(MAX_TIMESTEPS), 0.0

        total_lives = sum(log["people_affected"] for log in self._resolved_log)

        bd = GraderBreakdown(
            efficiency=round(efficiency, 4),
            accuracy=round(severity_score, 4),
            time_score=round(time_score, 4),
            coverage=round(resolution_rate, 4),
            fairness=round(fairness_score, 4),
        )
        grader = round(
            0.30 * resolution_rate
            + 0.25 * severity_score
            + 0.20 * efficiency
            + 0.15 * time_score
            + 0.10 * fairness_score,
            4,
        )
        return EpisodeResult(
            task_id=TASK_ID, difficulty=TaskDifficulty.MEDIUM,
            total_steps=self._timestep, total_reward=round(self._cumulative_reward, 4),
            grader_score=grader, breakdown=bd,
            resolved_incidents=resolved, failed_incidents=failed,
            resource_utilization=round(utilization, 4),
            average_response_time=round(avg_rt, 2),
            total_lives_saved=total_lives, cascade_events_triggered=0,
            details={
                "resolution_rate": round(resolution_rate, 4),
                "severity_score": round(severity_score, 4),
                "efficiency": round(efficiency, 4),
                "fairness_score": round(fairness_score, 4),
                "assignment_log": self._assignment_log,
            },
        )

    # -- internals --

    def _inc(self, iid: str) -> Incident:
        return next(i for i in self._incidents if i.id == iid)

    def _res(self, rid: str) -> Resource:
        return next(r for r in self._resources if r.id == rid)

    def _obs(self) -> Observation:
        fairness = compute_fairness(self._incidents)
        return Observation(
            incidents=self._incidents, resources=self._resources,
            timestep=self._timestep, task_id=TASK_ID,
            task_difficulty=TaskDifficulty.MEDIUM, max_timesteps=MAX_TIMESTEPS,
            resolved_count=sum(1 for i in self._incidents if i.is_resolved),
            failed_count=sum(1 for i in self._incidents
                             if (i.status.value if hasattr(i.status,"value") else i.status) == "failed"),
            active_count=sum(1 for i in self._incidents if i.is_active),
            cumulative_reward=self._cumulative_reward,
            total_lives_saved=sum(log["people_affected"] for log in self._resolved_log),
            fairness=fairness, step_log=self._step_logs,
        )

    def _apply(self, action: ActionWrapper, fairness: FairnessMetrics) -> Reward:
        at = action.action_type.value if hasattr(action.action_type,"value") else action.action_type

        if at == "assign_resource":
            if not action.resource_id or not action.incident_id:
                return Reward.invalid("assign_resource requires resource_id and incident_id")
            try:
                res = self._res(action.resource_id)
                inc = self._inc(action.incident_id)
            except StopIteration:
                return Reward.invalid("unknown resource_id or incident_id")
            if not res.available:
                return Reward.invalid(f"{res.id} is not available (busy/transit)")
            if not inc.is_active:
                return Reward.invalid(f"{inc.id} is not active (resolved/failed)")

            cascade_count = 0  # Task 2 has no cascades
            compat, dist_km, t_steps = WorldEngine.apply_assignment(res, inc, self._timestep)
            self._assignment_log.append({
                "step": self._timestep, "resource": res.id, "incident": inc.id,
                "compat": round(compat, 3), "dist_km": round(dist_km, 1),
                "transit_steps": int(t_steps),
            })
            return WorldEngine.compute_assignment_reward(compat, dist_km, inc, fairness, cascade_count)

        elif at in ("wait", "reprioritize"):
            return WorldEngine.compute_wait_reward(self._incidents, self._resources, fairness)

        elif at == "recall_resource":
            if not action.resource_id:
                return Reward.invalid("recall_resource requires resource_id")
            try:
                res = self._res(action.resource_id)
            except StopIteration:
                return Reward.invalid("unknown resource_id")
            if res.available:
                return Reward.invalid(f"{res.id} is already free — recall unnecessary")
            # Free the resource immediately (recall penalty: wasted work)
            if res.current_assignment:
                try:
                    inc = self._inc(res.current_assignment)
                    if res.id in inc.assigned_resources:
                        inc.assigned_resources.remove(res.id)
                except StopIteration:
                    pass
            res.available         = True
            res.current_assignment = None
            res.steps_until_free  = 0
            res.in_transit        = False
            bd = RewardBreakdown(delay_penalty=0.6, wrong_assignment_penalty=0.3)
            return Reward(value=bd.compute_value(), breakdown=bd, explanation="recall: resource freed but work wasted")

        else:
            return Reward.invalid(f"unknown action_type '{at}'")
