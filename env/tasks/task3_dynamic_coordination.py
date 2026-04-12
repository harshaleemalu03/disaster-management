"""
Task 3 (Hard) — Dynamic Multi-step Coordination
================================================
Full simulation: incidents evolve, escalate, cascade, and expire.
New incidents spawn stochastically (seeded). Resources travel with delay.
Agent must continuously re-optimize under time pressure.

Novel features:
  - Fire cascade: high-severity unattended fires spread to adjacent areas
  - Fairness-aware dispatch: rural incidents penalized when neglected
  - Partial observability: agent sees all incidents but not guaranteed resource ETA accuracy
  - Multi-step dependency: wrong assignment now blocks future moves

Grader: resolution_rate(0.30) + severity_weighted(0.25) + time_efficiency(0.20)
       + resource_efficiency(0.15) + fairness(0.10) - cascade_penalty
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    ActionType, ActionWrapper, CascadeEvent, EpisodeResult, GraderBreakdown,
    Incident, IncidentStatus, IncidentType, Location, Observation, Resource,
    ResourceType, Reward, RewardBreakdown, StepLog, TaskDifficulty, ZoneType,
    EXPIRY_STEPS, FairnessMetrics,
)
from env.world import WorldEngine, compute_fairness

TASK_ID       = "task3_dynamic_coordination"
MAX_TIMESTEPS = 30
NEW_INC_PROB  = 0.28   # probability of spawning per step

SPAWN_TEMPLATES = [
    (IncidentType.FIRE,       0.70, 50,  [ResourceType.FIRE_TRUCK],                          ZoneType.RURAL,    0.2),
    (IncidentType.MEDICAL,    0.55, 22,  [ResourceType.AMBULANCE],                           ZoneType.URBAN,    0.9),
    (IncidentType.FLOOD,      0.65, 90,  [ResourceType.RESCUE_TEAM],                         ZoneType.RURAL,    0.3),
    (IncidentType.EARTHQUAKE, 0.82, 110, [ResourceType.RESCUE_TEAM, ResourceType.HELICOPTER],ZoneType.SUBURBAN, 0.5),
    (IncidentType.ACCIDENT,   0.58, 14,  [ResourceType.AMBULANCE,   ResourceType.POLICE],    ZoneType.URBAN,    0.8),
    (IncidentType.HAZMAT,     0.75, 35,  [ResourceType.FIRE_TRUCK],                          ZoneType.SUBURBAN, 0.6),
    (IncidentType.MEDICAL,    0.90, 5,   [ResourceType.AMBULANCE],                           ZoneType.URBAN,    0.9),
    (IncidentType.FIRE,       0.68, 28,  [ResourceType.FIRE_TRUCK],                          ZoneType.RURAL,    0.2),
]


def _make_initial_incidents(rng: random.Random) -> List[Incident]:
    templates = [
        (IncidentType.FIRE,       0.85, 55,  Location(x=12.0, y=15.0, zone=ZoneType.RURAL),    [ResourceType.FIRE_TRUCK],                          ZoneType.RURAL,    0.2),
        (IncidentType.MEDICAL,    0.70, 30,  Location(x=72.0, y=68.0, zone=ZoneType.URBAN),    [ResourceType.AMBULANCE],                           ZoneType.URBAN,    0.9),
        (IncidentType.EARTHQUAKE, 0.92, 140, Location(x=48.0, y=80.0, zone=ZoneType.SUBURBAN), [ResourceType.RESCUE_TEAM, ResourceType.HELICOPTER],ZoneType.SUBURBAN, 0.5),
        (IncidentType.ACCIDENT,   0.60, 16,  Location(x=32.0, y=44.0, zone=ZoneType.SUBURBAN), [ResourceType.AMBULANCE,   ResourceType.POLICE],    ZoneType.SUBURBAN, 0.6),
    ]
    incidents = []
    for idx, (itype, sev, ppl, loc, req, zone, popd) in enumerate(templates):
        s       = round(min(max(sev + rng.uniform(-0.04, 0.04), 0.1), 1.0), 3)
        itype_s = itype.value if hasattr(itype,"value") else itype
        exp     = EXPIRY_STEPS.get(itype_s, 12)
        incidents.append(Incident(
            id=f"INC-{idx+1:03d}",
            incident_type=itype, severity=s,
            people_affected=max(ppl + rng.randint(-8, 8), 1),
            location=loc, time_since_report=round(rng.uniform(0.5, 3.0), 1),
            status=IncidentStatus.ACTIVE, required_resource_types=req,
            base_resolution_steps=5, steps_to_expiry=exp,
            zone=zone, population_density=round(popd + rng.uniform(-0.05, 0.05), 2),
        ))
    return incidents


def _make_resources(rng: random.Random) -> List[Resource]:
    configs = [
        ("RES-01", ResourceType.AMBULANCE,   Location(x=50.0, y=50.0, zone=ZoneType.URBAN)),
        ("RES-02", ResourceType.FIRE_TRUCK,  Location(x=18.0, y=12.0, zone=ZoneType.SUBURBAN)),
        ("RES-03", ResourceType.RESCUE_TEAM, Location(x=55.0, y=78.0, zone=ZoneType.SUBURBAN)),
        ("RES-04", ResourceType.POLICE,      Location(x=35.0, y=42.0, zone=ZoneType.SUBURBAN)),
        ("RES-05", ResourceType.HELICOPTER,  Location(x=58.0, y=55.0, zone=ZoneType.URBAN)),
        ("RES-06", ResourceType.AMBULANCE,   Location(x=80.0, y=70.0, zone=ZoneType.URBAN)),
    ]
    return [Resource(id=rid, resource_type=rt, location=loc) for rid, rt, loc in configs]


class Task3Environment:
    def __init__(self, seed: int = 42):
        self._seed     = seed
        self._rng      = random.Random(seed)
        self._incidents: List[Incident]     = []
        self._resources: List[Resource]     = []
        self._cascades:  List[CascadeEvent] = []
        self._timestep:  int                = 0
        self._cumulative_reward: float      = 0.0
        self._step_logs: List[StepLog]      = []
        self._assignment_log: List[Dict[str, Any]] = []
        self._resolved_log:   List[Dict[str, Any]] = []
        self._failed_log:     List[Dict[str, Any]] = []
        self._incident_counter: List[int]   = [0]   # mutable for WorldEngine
        self._done:      bool               = False

    def reset(self) -> Observation:
        self._rng       = random.Random(self._seed)
        self._incidents = _make_initial_incidents(self._rng)
        self._resources = _make_resources(self._rng)
        self._cascades  = []
        self._timestep  = 0
        self._cumulative_reward = 0.0
        self._step_logs = []
        self._assignment_log = []
        self._resolved_log   = []
        self._failed_log     = []
        self._incident_counter = [len(self._incidents)]
        self._done      = False
        return self._obs()

    def step(self, action: ActionWrapper) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode done; call reset().")
        self._timestep += 1

        # -- World dynamics (ordered) --
        WorldEngine.age_incidents(self._incidents, self._timestep)
        WorldEngine.escalate_incidents(self._incidents)
        WorldEngine.advance_resources(self._resources)
        newly_resolved = WorldEngine.check_resolutions(self._incidents, self._resources, self._timestep)
        newly_failed   = WorldEngine.expire_incidents(self._incidents, self._timestep)
        new_cascades   = WorldEngine.check_cascade(
            self._incidents, self._rng, self._timestep, self._incident_counter
        )
        self._cascades.extend(new_cascades)

        for iid in newly_resolved:
            inc = self._inc(iid)
            self._resolved_log.append({
                "incident_id": iid, "severity": inc.severity,
                "people_affected": inc.people_affected,
                "resolution_step": self._timestep,
                "time_since_report": inc.time_since_report,
                "zone": inc.zone.value if hasattr(inc.zone,"value") else inc.zone,
            })
        for iid in newly_failed:
            inc = self._inc(iid)
            self._failed_log.append({
                "incident_id": iid, "severity": inc.severity,
                "people_affected": inc.people_affected,
                "step": self._timestep,
                "zone": inc.zone.value if hasattr(inc.zone,"value") else inc.zone,
            })

        # Spawn new incidents (seeded)
        self._maybe_spawn()

        fairness = compute_fairness(self._incidents)
        reward   = self._apply(action, fairness, len(new_cascades))
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

        done = self._timestep >= MAX_TIMESTEPS
        self._done = done
        info = {
            "newly_resolved": newly_resolved,
            "newly_failed": newly_failed,
            "new_cascades": [c.model_dump() for c in new_cascades],
            "new_incidents_spawned": [
                i.id for i in self._incidents
                if i.time_since_report < 1.0 and not i.is_resolved
            ],
        }
        return self._obs(), reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "incidents": [i.model_dump() for i in self._incidents],
            "resources":  [r.model_dump() for r in self._resources],
            "timestep": self._timestep,
            "resolved_log": self._resolved_log,
            "failed_log": self._failed_log,
            "cascade_events": [c.model_dump() for c in self._cascades],
            "assignment_log": self._assignment_log,
        }

    def grade(self) -> EpisodeResult:
        total_spawned = self._incident_counter[0]
        resolved      = len(self._resolved_log)
        failed        = len(self._failed_log)

        resolution_rate = resolved / total_spawned if total_spawned > 0 else 0.0

        # Severity-weighted rescue score
        all_inc_sev = sum(i.severity for i in self._incidents)
        rescued_sev = sum(log["severity"] for log in self._resolved_log)
        severity_score = rescued_sev / all_inc_sev if all_inc_sev > 0 else 0.0

        # Time efficiency
        if self._resolved_log:
            avg_rt = sum(log["time_since_report"] for log in self._resolved_log) / len(self._resolved_log)
            max_exp = max(EXPIRY_STEPS.values())
            time_score = max(0.0, 1.0 - avg_rt / max_exp)
        else:
            avg_rt, time_score = float(MAX_TIMESTEPS), 0.0

        # Resource efficiency
        if self._assignment_log:
            avg_compat = sum(a["compat"] for a in self._assignment_log) / len(self._assignment_log)
            avg_dist   = sum(a["dist_km"] for a in self._assignment_log) / len(self._assignment_log)
            eff        = 0.6 * avg_compat + 0.4 * max(0.0, 1.0 - avg_dist / 80.0)
        else:
            eff = 0.0

        # Fairness
        fairness = compute_fairness(self._incidents)
        fairness_score = max(0.0, 1.0 - fairness.gini_coefficient)

        # Cascade penalty
        cascade_count  = len(self._cascades)
        cascade_penalty = min(cascade_count * 0.04, 0.20)

        # Resource utilization
        total_assign = sum(r.total_assignments for r in self._resources)
        max_possible = len(self._resources) * (MAX_TIMESTEPS // 5)
        utilization  = min(total_assign / max_possible, 1.0) if max_possible > 0 else 0.0

        total_lives = sum(log["people_affected"] for log in self._resolved_log)

        bd = GraderBreakdown(
            efficiency=round(eff, 4),
            accuracy=round(severity_score, 4),
            time_score=round(time_score, 4),
            coverage=round(resolution_rate, 4),
            fairness=round(fairness_score, 4),
        )
        grader = round(max(0.0001, min(0.9999, (
            0.30 * resolution_rate
            + 0.25 * severity_score
            + 0.20 * time_score
            + 0.15 * eff
            + 0.10 * fairness_score
            - cascade_penalty
        ))), 4)

        return EpisodeResult(
            task_id=TASK_ID, difficulty=TaskDifficulty.HARD,
            total_steps=self._timestep, total_reward=round(self._cumulative_reward, 4),
            grader_score=grader, breakdown=bd,
            resolved_incidents=resolved, failed_incidents=failed,
            resource_utilization=round(utilization, 4),
            average_response_time=round(avg_rt, 2),
            total_lives_saved=total_lives, cascade_events_triggered=cascade_count,
            details={
                "total_spawned": total_spawned,
                "resolution_rate": round(resolution_rate, 4),
                "severity_score": round(severity_score, 4),
                "time_score": round(time_score, 4),
                "resource_efficiency": round(eff, 4),
                "fairness_score": round(fairness_score, 4),
                "cascade_penalty": round(cascade_penalty, 4),
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
            task_difficulty=TaskDifficulty.HARD, max_timesteps=MAX_TIMESTEPS,
            resolved_count=sum(1 for i in self._incidents if i.is_resolved),
            failed_count=len(self._failed_log),
            active_count=sum(1 for i in self._incidents if i.is_active),
            cumulative_reward=self._cumulative_reward,
            total_lives_saved=sum(log["people_affected"] for log in self._resolved_log),
            fairness=fairness, step_log=self._step_logs,
            cascade_events=self._cascades,
        )

    def _maybe_spawn(self) -> None:
        active = sum(1 for i in self._incidents if i.is_active)
        if active >= 12:
            return
        if self._rng.random() > NEW_INC_PROB:
            return
        tmpl = self._rng.choice(SPAWN_TEMPLATES)
        itype, sev, ppl, req, zone, popd = tmpl
        itype_s = itype.value if hasattr(itype,"value") else itype
        exp     = EXPIRY_STEPS.get(itype_s, 12)
        s       = round(min(max(sev + self._rng.uniform(-0.08, 0.08), 0.1), 1.0), 3)
        self._incident_counter[0] += 1
        new_id  = f"INC-{self._incident_counter[0]:03d}"
        loc     = Location(
            x=round(self._rng.uniform(5.0, 95.0), 1),
            y=round(self._rng.uniform(5.0, 95.0), 1),
            zone=zone,
        )
        self._incidents.append(Incident(
            id=new_id, incident_type=itype, severity=s,
            people_affected=max(ppl + self._rng.randint(-10, 10), 1),
            location=loc, time_since_report=0.0,
            status=IncidentStatus.ACTIVE, required_resource_types=req,
            base_resolution_steps=5, steps_to_expiry=exp,
            zone=zone, population_density=round(popd + self._rng.uniform(-0.05, 0.05), 2),
        ))

    def _apply(self, action: ActionWrapper, fairness: FairnessMetrics, cascade_count: int) -> Reward:
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
                return Reward.invalid(f"{res.id} not available")
            if not inc.is_active:
                return Reward.invalid(f"{inc.id} not active")

            compat, dist_km, t_steps = WorldEngine.apply_assignment(res, inc, self._timestep)
            self._assignment_log.append({
                "step": self._timestep, "resource": res.id, "incident": inc.id,
                "compat": round(compat, 3), "dist_km": round(dist_km, 1),
                "transit_steps": int(t_steps),
            })
            return WorldEngine.compute_assignment_reward(compat, dist_km, inc, fairness, cascade_count)

        elif at == "reprioritize":
            # Reprioritize: reward if ordering matches urgency
            if not action.ordered_incident_ids:
                return WorldEngine.compute_wait_reward(self._incidents, self._resources, fairness)
            active = [i for i in self._incidents if i.is_active]
            optimal = sorted(active, key=lambda i: -i.urgency_score())
            opt_ids = [i.id for i in optimal]
            agent   = [x for x in action.ordered_incident_ids if x in {i.id for i in active}]
            missing = [x for x in opt_ids if x not in agent]
            agent   = agent + missing
            n = len(opt_ids)
            agent_pos = {iid: idx for idx, iid in enumerate(agent)}
            concordant = discordant = 0
            for i in range(n):
                for j in range(i+1, n):
                    ia, ib = opt_ids[i], opt_ids[j]
                    if agent_pos.get(ia, n) < agent_pos.get(ib, n):
                        concordant += 1
                    else:
                        discordant += 1
            total_p = concordant + discordant
            tau = (concordant - discordant) / total_p if total_p > 0 else 0.0
            tau_norm = round((tau + 1.0) / 2.0, 4)
            fairness_s = max(0.0, 1.0 - fairness.gini_coefficient)
            val = round(min(0.6 * tau_norm + 0.20 * fairness_s, 1.0), 4)
            bd = RewardBreakdown(life_saving_score=tau_norm, fairness_score=fairness_s)
            return Reward(value=val, breakdown=bd, explanation=f"reprioritize tau={tau_norm:.3f}")

        elif at == "wait":
            return WorldEngine.compute_wait_reward(self._incidents, self._resources, fairness)

        elif at == "recall_resource":
            if not action.resource_id:
                return Reward.invalid("recall_resource requires resource_id")
            try:
                res = self._res(action.resource_id)
            except StopIteration:
                return Reward.invalid("unknown resource_id")
            if res.available:
                return Reward.invalid(f"{res.id} already free")
            if res.current_assignment:
                try:
                    inc = self._inc(res.current_assignment)
                    if res.id in inc.assigned_resources:
                        inc.assigned_resources.remove(res.id)
                except StopIteration:
                    pass
            res.available = True
            res.current_assignment = None
            res.steps_until_free = 0
            res.in_transit = False
            bd = RewardBreakdown(delay_penalty=0.5, wrong_assignment_penalty=0.2)
            return Reward(value=bd.compute_value(), breakdown=bd, explanation="recall: freed resource early")

        else:
            return Reward.invalid(f"unknown action_type '{at}'")
