"""
Disaster Response Coordination System — Typed Data Models
All Pydantic models. Falls back to lightweight shim when pydantic absent.
"""
from __future__ import annotations

try:
    from pydantic import BaseModel, Field
except ImportError:
    from env._pydantic_shim import BaseModel, Field  # type: ignore

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import math


# ── Enums ─────────────────────────────────────────────────────────────────────

class ResourceType(str, Enum):
    AMBULANCE   = "ambulance"
    FIRE_TRUCK  = "fire_truck"
    RESCUE_TEAM = "rescue_team"
    POLICE      = "police"
    HELICOPTER  = "helicopter"

class IncidentType(str, Enum):
    FIRE       = "fire"
    MEDICAL    = "medical"
    FLOOD      = "flood"
    EARTHQUAKE = "earthquake"
    ACCIDENT   = "accident"
    HAZMAT     = "hazmat"

class ActionType(str, Enum):
    ASSIGN_RESOURCE = "assign_resource"
    REPRIORITIZE    = "reprioritize"
    RECALL_RESOURCE = "recall_resource"
    WAIT            = "wait"

class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"

class IncidentStatus(str, Enum):
    ACTIVE    = "active"
    CONTAINED = "contained"
    RESOLVED  = "resolved"
    FAILED    = "failed"

class ZoneType(str, Enum):
    URBAN    = "urban"
    SUBURBAN = "suburban"
    RURAL    = "rural"


# ── Physics constants ──────────────────────────────────────────────────────────

GRID_KM     = 100.0
MAX_DIST_KM = GRID_KM * math.sqrt(2)

RESOURCE_SPEED: Dict[str, float] = {
    "ambulance": 8.0, "fire_truck": 7.0, "rescue_team": 5.0,
    "police": 9.0, "helicopter": 25.0,
}

ZONE_TRAFFIC: Dict[str, float] = {
    "urban": 1.6, "suburban": 1.1, "rural": 0.9,
}

COMPATIBILITY: Dict[str, Dict[str, float]] = {
    "ambulance":   {"medical":1.00,"accident":0.90,"fire":0.45,"flood":0.40,"earthquake":0.55,"hazmat":0.25},
    "fire_truck":  {"fire":1.00,"hazmat":0.85,"accident":0.35,"medical":0.10,"flood":0.20,"earthquake":0.30},
    "rescue_team": {"earthquake":1.00,"flood":0.95,"fire":0.50,"accident":0.60,"hazmat":0.40,"medical":0.30},
    "police":      {"accident":1.00,"hazmat":0.60,"fire":0.30,"medical":0.15,"flood":0.25,"earthquake":0.35},
    "helicopter":  {"flood":1.00,"earthquake":0.90,"fire":0.70,"medical":0.80,"accident":0.50,"hazmat":0.40},
}

ESCALATION_RATE: Dict[str, float] = {
    "fire":0.06,"hazmat":0.04,"flood":0.03,"earthquake":0.02,"medical":0.05,"accident":0.03,
}

EXPIRY_STEPS: Dict[str, int] = {
    "fire":10,"medical":8,"flood":14,"earthquake":16,"accident":9,"hazmat":12,
}

PEOPLE_GROWTH: Dict[str, float] = {
    "fire":1.12,"flood":1.08,"earthquake":1.05,"hazmat":1.06,"medical":1.0,"accident":1.0,
}

INCIDENT_EMOJI: Dict[str, str] = {
    "fire":"🔥","medical":"🏥","flood":"🌊","earthquake":"⚡","accident":"💥","hazmat":"☣️",
}
RESOURCE_EMOJI: Dict[str, str] = {
    "ambulance":"🚑","fire_truck":"🚒","rescue_team":"🛟","police":"🚓","helicopter":"🚁",
}


def _v(val: Any) -> str:
    return val.value if hasattr(val, "value") else val

def travel_steps(resource_type: str, dist_km: float, zone: str) -> int:
    rt = _v(resource_type); z = _v(zone)
    speed = RESOURCE_SPEED.get(rt, 6.0)
    traffic = ZONE_TRAFFIC.get(z, 1.0)
    return max(1, math.ceil(dist_km / (speed / traffic)))

def get_compatibility(resource_type: str, incident_type: str) -> float:
    return COMPATIBILITY.get(_v(resource_type), {}).get(_v(incident_type), 0.15)


# ── Sub-models ────────────────────────────────────────────────────────────────

class Location(BaseModel):
    x:    float    = Field(default=0.0)
    y:    float    = Field(default=0.0)
    zone: ZoneType = Field(default=ZoneType.SUBURBAN)

    def distance_to(self, other: "Location") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def model_dump(self) -> Dict[str, Any]:
        return {"x": round(self.x,2), "y": round(self.y,2), "zone": _v(self.zone)}

    def grid_cell(self, cols:int=10, rows:int=10):
        return (min(int(self.x/GRID_KM*cols), cols-1),
                min(int(self.y/GRID_KM*rows), rows-1))


class CascadeEvent(BaseModel):
    source_incident_id: str
    trigger_step:       int
    incident_type:      IncidentType
    severity:           float
    location:           Location

    def model_dump(self) -> Dict[str, Any]:
        return {"source_incident_id":self.source_incident_id,
                "trigger_step":self.trigger_step,
                "incident_type":_v(self.incident_type),
                "severity":self.severity,
                "location":self.location.model_dump()}


class Incident(BaseModel):
    id:                      str
    incident_type:           IncidentType
    severity:                float
    people_affected:         int
    location:                Location
    time_since_report:       float
    status:                  IncidentStatus        = Field(default=IncidentStatus.ACTIVE)
    assigned_resources:      List[str]             = Field(default_factory=list)
    required_resource_types: List[ResourceType]   = Field(default_factory=list)
    base_resolution_steps:   int                  = Field(default=5)
    steps_to_expiry:         int                  = Field(default=12)
    steps_being_treated:     int                  = Field(default=0)
    resolution_step:         Optional[int]        = Field(default=None)
    cascade_triggered:       bool                 = Field(default=False)
    zone:                    ZoneType             = Field(default=ZoneType.SUBURBAN)
    population_density:      float                = Field(default=0.5)

    def __init__(self, **data):
        data.setdefault("assigned_resources", [])
        data.setdefault("required_resource_types", [])
        data.setdefault("status", IncidentStatus.ACTIVE)
        data.setdefault("steps_being_treated", 0)
        data.setdefault("cascade_triggered", False)
        data.setdefault("zone", ZoneType.SUBURBAN)
        data.setdefault("population_density", 0.5)
        super().__init__(**data)

    @property
    def is_active(self) -> bool:
        return _v(self.status) in ("active","contained")

    @property
    def is_resolved(self) -> bool:
        return _v(self.status) == "resolved"

    def itype_str(self) -> str:
        return _v(self.incident_type)

    def urgency_score(self) -> float:
        expiry_pressure = max(0.0, 1.0 - self.steps_to_expiry / 16.0)
        return round(0.45*self.severity + 0.25*min(self.people_affected/200.0,1.0)
                     + 0.20*expiry_pressure + 0.10*self.population_density, 4)

    def emoji(self) -> str:
        return INCIDENT_EMOJI.get(self.itype_str(), "❓")

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "incident_type": _v(self.incident_type),
            "severity": round(self.severity, 4),
            "people_affected": self.people_affected,
            "location": self.location.model_dump(),
            "time_since_report": round(self.time_since_report, 2),
            "status": _v(self.status),
            "assigned_resources": list(self.assigned_resources),
            "required_resource_types": [_v(rt) for rt in self.required_resource_types],
            "base_resolution_steps": self.base_resolution_steps,
            "steps_to_expiry": self.steps_to_expiry,
            "steps_being_treated": self.steps_being_treated,
            "resolution_step": self.resolution_step,
            "cascade_triggered": self.cascade_triggered,
            "zone": _v(self.zone),
            "population_density": self.population_density,
            "urgency_score": self.urgency_score(),
        }


class Resource(BaseModel):
    id:                     str
    resource_type:          ResourceType
    available:              bool               = Field(default=True)
    location:               Location
    home_location:          Optional[Location] = Field(default=None)
    current_assignment:     Optional[str]      = Field(default=None)
    steps_until_free:       int                = Field(default=0)
    steps_in_transit:       int                = Field(default=0)
    in_transit:             bool               = Field(default=False)
    total_assignments:      int                = Field(default=0)
    successful_assignments: int                = Field(default=0)
    total_km_traveled:      float              = Field(default=0.0)

    def __init__(self, **data):
        if "home_location" not in data or data.get("home_location") is None:
            data["home_location"] = data.get("location", Location())
        super().__init__(**data)

    def rtype_str(self) -> str:
        return _v(self.resource_type)

    def emoji(self) -> str:
        return RESOURCE_EMOJI.get(self.rtype_str(), "🔧")

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "resource_type": _v(self.resource_type),
            "available": self.available,
            "location": self.location.model_dump(),
            "home_location": self.home_location.model_dump() if self.home_location else None,
            "current_assignment": self.current_assignment,
            "steps_until_free": self.steps_until_free,
            "steps_in_transit": self.steps_in_transit,
            "in_transit": self.in_transit,
            "total_assignments": self.total_assignments,
            "successful_assignments": self.successful_assignments,
            "total_km_traveled": round(self.total_km_traveled, 2),
        }


class FairnessMetrics(BaseModel):
    urban_response_rate:    float = Field(default=0.0)
    suburban_response_rate: float = Field(default=0.0)
    rural_response_rate:    float = Field(default=0.0)
    gini_coefficient:       float = Field(default=0.0)
    neglected_rural_steps:  int   = Field(default=0)

    def __init__(self, **data):
        for k in ["urban_response_rate","suburban_response_rate","rural_response_rate","gini_coefficient"]:
            data.setdefault(k, 0.0)
        data.setdefault("neglected_rural_steps", 0)
        super().__init__(**data)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "urban_response_rate":    round(self.urban_response_rate,    4),
            "suburban_response_rate": round(self.suburban_response_rate, 4),
            "rural_response_rate":    round(self.rural_response_rate,    4),
            "gini_coefficient":       round(self.gini_coefficient,       4),
            "neglected_rural_steps":  self.neglected_rural_steps,
        }


class StepLog(BaseModel):
    step:               int
    action_type:        str
    action_detail:      str
    reward:             float
    incidents_active:   int
    incidents_resolved: int
    resources_busy:     int
    explanation:        str

    def model_dump(self) -> Dict[str, Any]:
        return {"step":self.step,"action_type":self.action_type,
                "action_detail":self.action_detail,"reward":round(self.reward,4),
                "incidents_active":self.incidents_active,
                "incidents_resolved":self.incidents_resolved,
                "resources_busy":self.resources_busy,"explanation":self.explanation}


class Observation(BaseModel):
    incidents:         List[Incident]     = Field(default_factory=list)
    resources:         List[Resource]     = Field(default_factory=list)
    timestep:          int                = Field(default=0)
    task_id:           str
    task_difficulty:   TaskDifficulty
    max_timesteps:     int
    resolved_count:    int                = Field(default=0)
    failed_count:      int                = Field(default=0)
    active_count:      int                = Field(default=0)
    cumulative_reward: float              = Field(default=0.0)
    total_lives_saved: int                = Field(default=0)
    fairness:          FairnessMetrics    = Field(default_factory=FairnessMetrics)
    step_log:          List[StepLog]      = Field(default_factory=list)
    cascade_events:    List[CascadeEvent] = Field(default_factory=list)

    def __init__(self, **data):
        for k,v in [("incidents",[]),("resources",[]),("resolved_count",0),("failed_count",0),
                    ("active_count",0),("cumulative_reward",0.0),("total_lives_saved",0),
                    ("step_log",[]),("cascade_events",[])]:
            data.setdefault(k,v)
        data.setdefault("fairness", FairnessMetrics())
        super().__init__(**data)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "incidents":         [i.model_dump() for i in self.incidents],
            "resources":         [r.model_dump() for r in self.resources],
            "timestep":          self.timestep,
            "task_id":           self.task_id,
            "task_difficulty":   _v(self.task_difficulty),
            "max_timesteps":     self.max_timesteps,
            "resolved_count":    self.resolved_count,
            "failed_count":      self.failed_count,
            "active_count":      self.active_count,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "total_lives_saved": self.total_lives_saved,
            "fairness":          self.fairness.model_dump(),
            "step_log":          [s.model_dump() for s in self.step_log[-5:]],
            "cascade_events":    [c.model_dump() for c in self.cascade_events[-3:]],
        }

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        return {"type":"object","title":"Observation"}

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_text(self) -> str:
        td = _v(self.task_difficulty)
        lines = [
            f"╔═══ DISASTER CONTROL: {self.task_id} ({td.upper()}) ═══╗",
            f"  Step {self.timestep}/{self.max_timesteps} | Resolved:{self.resolved_count} "
            f"Failed:{self.failed_count} Active:{self.active_count} "
            f"Lives:{self.total_lives_saved} Reward:{self.cumulative_reward:.3f}",
            f"  Fairness Gini:{self.fairness.gini_coefficient:.3f} "
            f"Urban:{self.fairness.urban_response_rate:.0%} "
            f"Rural:{self.fairness.rural_response_rate:.0%}",
            "",
            "┌─ ACTIVE INCIDENTS ─────────────────────────────────────────┐",
        ]
        active = [i for i in self.incidents if i.is_active]
        for inc in sorted(active, key=lambda x: -x.urgency_score()):
            zone     = _v(inc.zone)
            assigned = ", ".join(inc.assigned_resources) if inc.assigned_resources else "⚠ NONE"
            lines.append(
                f"  [{inc.id}] {inc.emoji()} {inc.itype_str():10} "
                f"sev={inc.severity:.2f} ppl={inc.people_affected:4d} "
                f"expiry={inc.steps_to_expiry:2d} zone={zone:8} "
                f"urgency={inc.urgency_score():.3f} [{assigned}]"
            )
        if not active:
            lines.append("  (no active incidents)")
        lines += ["└────────────────────────────────────────────────────────────┘",
                  "┌─ RESOURCES ─────────────────────────────────────────────────┐"]
        for res in self.resources:
            zone = _v(res.location.zone)
            if res.available:
                st = "✓ FREE"
            elif res.in_transit:
                st = f"→ TRANSIT({res.steps_in_transit}s)→{res.current_assignment}"
            else:
                st = f"● BUSY({res.steps_until_free}s)→{res.current_assignment}"
            lines.append(
                f"  [{res.id}] {res.emoji()} {res.rtype_str():12} "
                f"@({res.location.x:5.1f},{res.location.y:5.1f}) zone={zone:8} | {st}"
            )
        lines.append("└────────────────────────────────────────────────────────────┘")
        if self.step_log:
            lines += ["","┌─ RECENT ACTIONS ────────────────────────────────────────────┐"]
            for e in self.step_log[-3:]:
                lines.append(f"  Step {e.step:2d} | {e.action_detail:42} | r={e.reward:.3f}")
            lines.append("└────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


class ActionWrapper(BaseModel):
    action_type:           ActionType
    resource_id:           Optional[str]       = Field(default=None)
    incident_id:           Optional[str]       = Field(default=None)
    ordered_incident_ids:  Optional[List[str]] = Field(default=None)

    def __init__(self, **data):
        if "action_type" in data and isinstance(data["action_type"], str):
            data["action_type"] = ActionType(data["action_type"])
        data.setdefault("resource_id", None)
        data.setdefault("incident_id", None)
        data.setdefault("ordered_incident_ids", None)
        super().__init__(**data)

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        return {"type":"object","title":"ActionWrapper"}


class RewardBreakdown(BaseModel):
    life_saving_score:         float = Field(default=0.0)
    response_time_score:       float = Field(default=0.0)
    resource_efficiency_score: float = Field(default=0.0)
    fairness_score:            float = Field(default=0.0)
    delay_penalty:             float = Field(default=0.0)
    idle_resource_penalty:     float = Field(default=0.0)
    wrong_assignment_penalty:  float = Field(default=0.0)
    cascade_penalty:           float = Field(default=0.0)

    def __init__(self, **data):
        for k in ["life_saving_score","response_time_score","resource_efficiency_score",
                  "fairness_score","delay_penalty","idle_resource_penalty",
                  "wrong_assignment_penalty","cascade_penalty"]:
            data.setdefault(k, 0.0)
        super().__init__(**data)

    def model_dump(self) -> Dict[str, Any]:
        return {k: round(getattr(self,k),4) for k in [
            "life_saving_score","response_time_score","resource_efficiency_score",
            "fairness_score","delay_penalty","idle_resource_penalty",
            "wrong_assignment_penalty","cascade_penalty"]}

    def compute_value(self) -> float:
        raw = (0.35*self.life_saving_score + 0.25*self.response_time_score
               + 0.20*self.resource_efficiency_score + 0.10*self.fairness_score
               - 0.05*self.delay_penalty - 0.03*self.idle_resource_penalty
               - 0.02*self.wrong_assignment_penalty - 0.05*self.cascade_penalty)
        return round(min(max(raw, 0.0), 1.0), 4)


class Reward(BaseModel):
    value:       float
    breakdown:   RewardBreakdown = Field(default_factory=RewardBreakdown)
    explanation: str             = Field(default="")

    def __init__(self, **data):
        data.setdefault("breakdown", RewardBreakdown())
        data.setdefault("explanation", "")
        super().__init__(**data)

    def model_dump(self) -> Dict[str, Any]:
        return {"value":self.value,"breakdown":self.breakdown.model_dump(),"explanation":self.explanation}

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        return {"type":"object","title":"Reward"}

    @staticmethod
    def invalid(reason: str) -> "Reward":
        bd = RewardBreakdown(wrong_assignment_penalty=1.0)
        return Reward(value=bd.compute_value(), breakdown=bd, explanation=reason)


class GraderBreakdown(BaseModel):
    efficiency: float = Field(default=0.0)
    accuracy:   float = Field(default=0.0)
    time_score: float = Field(default=0.0)
    coverage:   float = Field(default=0.0)
    fairness:   float = Field(default=0.0)

    def __init__(self, **data):
        for k in ["efficiency","accuracy","time_score","coverage","fairness"]:
            data.setdefault(k, 0.0)
        super().__init__(**data)

    def model_dump(self) -> Dict[str, Any]:
        return {k: round(getattr(self,k),4) for k in ["efficiency","accuracy","time_score","coverage","fairness"]}


class EpisodeResult(BaseModel):
    task_id:                  str
    difficulty:               TaskDifficulty
    total_steps:              int
    total_reward:             float
    grader_score:             float
    breakdown:                GraderBreakdown = Field(default_factory=GraderBreakdown)
    resolved_incidents:       int
    failed_incidents:         int
    resource_utilization:     float
    average_response_time:    float
    total_lives_saved:        int
    cascade_events_triggered: int
    details:                  Dict[str, Any]  = Field(default_factory=dict)

    def __init__(self, **data):
        data.setdefault("breakdown", GraderBreakdown())
        data.setdefault("details", {})
        data.setdefault("cascade_events_triggered", 0)
        super().__init__(**data)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "task_id":self.task_id,"difficulty":_v(self.difficulty),
            "total_steps":self.total_steps,"total_reward":round(self.total_reward,4),
            "grader_score":round(self.grader_score,4),"breakdown":self.breakdown.model_dump(),
            "resolved_incidents":self.resolved_incidents,"failed_incidents":self.failed_incidents,
            "resource_utilization":round(self.resource_utilization,4),
            "average_response_time":round(self.average_response_time,2),
            "total_lives_saved":self.total_lives_saved,
            "cascade_events_triggered":self.cascade_events_triggered,
            "details":self.details,
        }
