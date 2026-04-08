"""
env/world.py — Shared deterministic world simulation engine v2.1
================================================================
Physics, escalation, cascade, resolution, fairness — used by all 3 tasks.

Step ordering (every task calls in this exact sequence):
  1. age_incidents          — increment time_since_report, decrement steps_to_expiry
  2. escalate_incidents     — severity/people grow for unattended incidents
  3. advance_resources      — tick transit & busy counters; free resources when done
  4. check_resolutions      — mark resolved when resource coverage + treatment met
  5. expire_incidents       — mark failed if steps_to_expiry <= 0
  6. check_cascade          — seeded fire-spread (Task 3 only)
  7. apply_action           — agent action → reward
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    CascadeEvent, ESCALATION_RATE, EXPIRY_STEPS, FairnessMetrics,
    GraderBreakdown, Incident, IncidentStatus, IncidentType, Location,
    MAX_DIST_KM, PEOPLE_GROWTH, Reward, RewardBreakdown, Resource,
    ResourceType, StepLog, ZoneType, _v, get_compatibility, travel_steps,
)


CASCADE_SPREAD_DIST_KM     = 15.0
CASCADE_SEVERITY_THRESHOLD = 0.80
CASCADE_PROB               = 0.40
GRID_COLS                  = 10
GRID_ROWS                  = 10


# ── Fairness ──────────────────────────────────────────────────────────────────

def compute_fairness(incidents: List[Incident]) -> FairnessMetrics:
    """
    Compute Gini coefficient across urban/suburban/rural response rates.
    Gini = mean_abs_diff / (2 × n × mean_rate).
    """
    zone_total:    Dict[str, int] = {"urban": 0, "suburban": 0, "rural": 0}
    zone_resolved: Dict[str, int] = {"urban": 0, "suburban": 0, "rural": 0}
    neglected = 0

    for inc in incidents:
        z = _v(inc.zone)
        if z not in zone_total:
            z = "suburban"
        zone_total[z] += 1
        if _v(inc.status) == "resolved":
            zone_resolved[z] += 1
        if z == "rural" and inc.is_active and not inc.assigned_resources and inc.time_since_report > 3:
            neglected += 1

    rates = [
        zone_resolved[z] / zone_total[z] if zone_total[z] > 0 else 0.0
        for z in ("urban", "suburban", "rural")
    ]
    n     = len(rates)
    total = sum(rates)
    if total == 0:
        gini = 0.0
    else:
        mad  = sum(abs(rates[i] - rates[j]) for i in range(n) for j in range(n))
        gini = mad / (2 * n * total)

    return FairnessMetrics(
        urban_response_rate=round(rates[0], 4),
        suburban_response_rate=round(rates[1], 4),
        rural_response_rate=round(rates[2], 4),
        gini_coefficient=round(min(gini, 1.0), 4),
        neglected_rural_steps=neglected,
    )


# ── Grid visualization ────────────────────────────────────────────────────────

def build_text_grid(
    incidents: List[Incident],
    resources: List[Resource],
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
) -> str:
    """
    Build a 2-D text map showing incident and resource positions.
    Priority: incidents > resources > empty cell.

    Returns a multiline string, e.g.:
       [ ] [🔥] [ ]
       [🚑] [ ] [🚒]
       [ ] [🛟] [ ]
    """
    from env.models import GRID_KM

    # cell[row][col] = emoji string
    cell: Dict[Tuple[int,int], str] = {}

    # Place incidents (active ones take priority)
    for inc in incidents:
        if not inc.is_active:
            continue
        c, r = inc.location.grid_cell(cols, rows)
        key  = (r, c)
        existing = cell.get(key, "")
        # Stack multiple icons with + separator (max 2)
        if not existing:
            cell[key] = inc.emoji()
        elif "+" not in existing:
            cell[key] = existing + "+" + inc.emoji()

    # Place resources (only if cell not already occupied by incident)
    for res in resources:
        c, r = res.location.grid_cell(cols, rows)
        key  = (r, c)
        if key not in cell:
            cell[key] = res.emoji()

    # Build grid string
    lines = []
    for row in range(rows - 1, -1, -1):   # top = high y
        row_parts = []
        for col in range(cols):
            icon = cell.get((row, col), " ")
            # Pad to 2 chars for alignment
            row_parts.append(f"[{icon:<2}]")
        lines.append(" ".join(row_parts))
    return "\n".join(lines)


def build_html_view(
    incidents: List[Incident],
    resources: List[Resource],
    timestep: int,
    task_id: str,
    cols: int = 10,
    rows: int = 10,
) -> str:
    """
    Generate a self-contained HTML page rendering the grid visually.
    Cells use emoji, color-coded by content type.
    """
    from env.models import GRID_KM

    # Build cell map: (row, col) → list of (emoji, label, color)
    cell_data: Dict[Tuple[int,int], List[tuple]] = {}

    zone_colors = {"urban": "#ffd700", "suburban": "#90ee90", "rural": "#87ceeb"}
    inc_colors  = {"fire":"#ff4500","medical":"#ff69b4","flood":"#1e90ff",
                   "earthquake":"#9932cc","accident":"#ff8c00","hazmat":"#32cd32"}

    for inc in incidents:
        if not inc.is_active:
            continue
        c, r = inc.location.grid_cell(cols, rows)
        key  = (r, c)
        cell_data.setdefault(key, [])
        color = inc_colors.get(inc.itype_str(), "#888")
        cell_data[key].append((inc.emoji(), f"{inc.id} sev={inc.severity:.2f}", color, "inc"))

    for res in resources:
        c, r = res.location.grid_cell(cols, rows)
        key  = (r, c)
        cell_data.setdefault(key, [])
        color = "#4CAF50" if res.available else ("#FF9800" if res.in_transit else "#F44336")
        status = "free" if res.available else ("transit" if res.in_transit else "busy")
        cell_data[key].append((res.emoji(), f"{res.id} ({status})", color, "res"))

    # Build HTML table rows (top = high y)
    table_rows = []
    cell_size  = 64  # px

    for row in range(rows - 1, -1, -1):
        cells_html = []
        for col in range(cols):
            key    = (row, col)
            items  = cell_data.get(key, [])
            bg     = "#1a1a2e"
            border = "#2a2a4e"
            if items:
                # Dominant color = first item
                bg = items[0][2]
                border = "rgba(255,255,255,0.3)"

            content = ""
            tooltip = ""
            for emoji, label, color, kind in items[:2]:
                content += f'<span style="font-size:20px;line-height:1.1">{emoji}</span>'
                tooltip += label + "&#10;"

            title_attr = f'title="{tooltip.strip()}"' if tooltip else ""
            cells_html.append(
                f'<td {title_attr} style="width:{cell_size}px;height:{cell_size}px;'
                f'background:{bg};border:1px solid {border};text-align:center;'
                f'vertical-align:middle;cursor:default;border-radius:4px">'
                f'{content or "<span style=\'opacity:0.15\'>·</span>"}</td>'
            )
        table_rows.append(f"<tr>{''.join(cells_html)}</tr>")

    legend_items = [
        ("🔥","Fire"),("🏥","Medical"),("🌊","Flood"),("⚡","Earthquake"),
        ("💥","Accident"),("☣️","Hazmat"),
        ("🚑","Ambulance"),("🚒","Fire Truck"),("🛟","Rescue"),("🚓","Police"),("🚁","Helicopter"),
    ]
    legend_html = "".join(
        f'<span style="margin-right:12px;font-size:15px">{e} {l}</span>'
        for e, l in legend_items
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">
<title>Disaster Response — Step {timestep}</title>
<style>
  body{{margin:0;padding:20px;background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif}}
  h1{{color:#58a6ff;margin-bottom:4px;font-size:1.4rem}}
  .subtitle{{color:#8b949e;font-size:0.85rem;margin-bottom:16px}}
  table{{border-collapse:separate;border-spacing:3px}}
  .legend{{margin-top:14px;padding:10px;background:#161b22;border-radius:8px;font-size:14px}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;margin-right:6px}}
</style>
</head>
<body>
<h1>🚨 Disaster Response Coordination</h1>
<div class="subtitle">Task: <strong>{task_id}</strong> &nbsp;|&nbsp; Step: <strong>{timestep}</strong> &nbsp;|&nbsp; Auto-refreshes every 5 seconds</div>
<table>
{''.join(table_rows)}
</table>
<div class="legend">
  <strong>Legend:</strong>&nbsp;&nbsp;{legend_html}
  <br><br>
  <span style="color:#8b949e;font-size:12px">
  Cell color: incident type color = active incident &nbsp;|&nbsp;
  🟢 green resource = free &nbsp;|&nbsp; 🟠 orange = in transit &nbsp;|&nbsp; 🔴 red = busy
  </span>
</div>
</body>
</html>"""


# ── World engine ──────────────────────────────────────────────────────────────

class WorldEngine:
    """Stateless helper called by each task environment each step."""

    @staticmethod
    def age_incidents(incidents: List[Incident], timestep: int) -> None:
        """Increment time_since_report; decrement steps_to_expiry for all active incidents."""
        for inc in incidents:
            if inc.is_active:
                inc.time_since_report += 1.0
                inc.steps_to_expiry   -= 1

    @staticmethod
    def escalate_incidents(incidents: List[Incident]) -> None:
        """
        Severity and people_affected grow for unattended active incidents.
        Incidents with resources assigned are 'contained' (slow escalation).
        """
        for inc in incidents:
            if not inc.is_active:
                continue
            itype = inc.itype_str()
            if not inc.assigned_resources:
                # Fully unattended — escalate at full rate
                rate   = ESCALATION_RATE.get(itype, 0.03)
                growth = PEOPLE_GROWTH.get(itype, 1.0)
                inc.severity       = round(min(inc.severity + rate, 1.0), 4)
                inc.people_affected = int(math.ceil(inc.people_affected * growth))
                inc.status          = IncidentStatus.ACTIVE
            else:
                # Being treated — mark contained, accumulate treatment steps
                inc.steps_being_treated += 1
                inc.status = IncidentStatus.CONTAINED

    @staticmethod
    def advance_resources(resources: List[Resource]) -> None:
        """Tick transit counter; then tick busy counter; free resource when done."""
        for res in resources:
            if res.in_transit:
                res.steps_in_transit -= 1
                if res.steps_in_transit <= 0:
                    res.steps_in_transit = 0
                    res.in_transit       = False
                    # Now on-scene: steps_until_free still counts down treatment time
            if not res.available and not res.in_transit:
                res.steps_until_free -= 1
                if res.steps_until_free <= 0:
                    res.available          = True
                    res.current_assignment = None
                    res.steps_until_free   = 0

    @staticmethod
    def check_resolutions(
        incidents: List[Incident],
        resources: List[Resource],
        timestep: int,
    ) -> List[str]:
        """
        Resolve incidents that have:
          (a) all required resource types present and not in-transit, AND
          (b) accumulated steps_being_treated >= base_resolution_steps.
        Returns list of newly-resolved incident IDs.
        """
        res_map         = {r.id: r for r in resources}
        newly_resolved: List[str] = []

        for inc in incidents:
            if not inc.is_active or not inc.assigned_resources:
                continue

            # Only count resources that have arrived (not in transit)
            on_scene = [
                rid for rid in inc.assigned_resources
                if rid in res_map and not res_map[rid].in_transit
            ]
            if not on_scene:
                continue

            assigned_types = {_v(res_map[rid].resource_type) for rid in on_scene}
            required_types = {_v(rt) for rt in inc.required_resource_types}
            coverage       = required_types.issubset(assigned_types) if required_types else True

            if coverage and inc.steps_being_treated >= inc.base_resolution_steps:
                inc.status          = IncidentStatus.RESOLVED
                inc.resolution_step = timestep
                newly_resolved.append(inc.id)
                # Free all assigned resources
                for rid in inc.assigned_resources:
                    if rid in res_map:
                        r                      = res_map[rid]
                        r.available            = True
                        r.steps_until_free     = 0
                        r.in_transit           = False
                        r.steps_in_transit     = 0
                        r.current_assignment   = None
                        r.successful_assignments += 1

        return newly_resolved

    @staticmethod
    def expire_incidents(incidents: List[Incident], timestep: int) -> List[str]:
        """Mark incidents as failed when steps_to_expiry <= 0."""
        failed: List[str] = []
        for inc in incidents:
            if inc.is_active and inc.steps_to_expiry <= 0:
                # Free any assigned resources before failing
                inc.status = IncidentStatus.FAILED
                failed.append(inc.id)
        return failed

    @staticmethod
    def check_cascade(
        incidents: List[Incident],
        rng:       random.Random,
        timestep:  int,
        counter:   List[int],
    ) -> List[CascadeEvent]:
        """
        Fire cascade: unattended fire at severity ≥ threshold can spread within
        CASCADE_SPREAD_DIST_KM with probability CASCADE_PROB per step (seeded).
        """
        new_cascades: List[CascadeEvent] = []

        for inc in list(incidents):   # snapshot to avoid mutation-during-iteration
            if inc.cascade_triggered:
                continue
            if inc.itype_str() != "fire":
                continue
            if inc.severity < CASCADE_SEVERITY_THRESHOLD:
                continue
            if inc.assigned_resources:
                continue
            if rng.random() > CASCADE_PROB:
                continue

            # Spawn secondary fire at random bearing
            angle = rng.uniform(0.0, 2.0 * math.pi)
            dist  = rng.uniform(3.0, CASCADE_SPREAD_DIST_KM)
            nx    = min(max(inc.location.x + dist * math.cos(angle), 2.0), 98.0)
            ny    = min(max(inc.location.y + dist * math.sin(angle), 2.0), 98.0)

            counter[0] += 1
            new_id = f"INC-{counter[0]:03d}"
            zone   = inc.zone

            new_inc = Incident(
                id                    = new_id,
                incident_type         = IncidentType.FIRE,
                severity              = round(rng.uniform(0.40, 0.65), 3),
                people_affected       = rng.randint(5, 40),
                location              = Location(x=round(nx, 1), y=round(ny, 1), zone=zone),
                time_since_report     = 0.0,
                status                = IncidentStatus.ACTIVE,
                required_resource_types = [ResourceType.FIRE_TRUCK],
                base_resolution_steps = 4,
                steps_to_expiry       = EXPIRY_STEPS["fire"],
                zone                  = zone,
                population_density    = round(rng.uniform(0.2, 0.8), 2),
            )
            incidents.append(new_inc)

            cascade = CascadeEvent(
                source_incident_id = inc.id,
                trigger_step       = timestep,
                incident_type      = IncidentType.FIRE,
                severity           = new_inc.severity,
                location           = new_inc.location,
            )
            new_cascades.append(cascade)
            inc.cascade_triggered = True

        return new_cascades

    # ── Assignment helper ──────────────────────────────────────────────────

    @staticmethod
    def apply_assignment(
        resource:  Resource,
        incident:  Incident,
        timestep:  int,
    ) -> Tuple[float, float, int]:
        """
        Assign resource to incident with realistic travel delay.
        Returns (compatibility, dist_km, transit_steps).
        Mutates resource and incident in-place.
        """
        dist_km   = resource.location.distance_to(incident.location)
        zone      = _v(incident.location.zone)
        t_steps   = travel_steps(resource.rtype_str(), dist_km, zone)
        compat    = get_compatibility(resource.rtype_str(), incident.itype_str())

        # On-scene treatment time (better compatibility = faster resolution)
        on_scene  = max(1, int(incident.base_resolution_steps * (1.1 - compat * 0.5)))

        # Mutate resource
        resource.available          = False
        resource.current_assignment = incident.id
        resource.steps_in_transit   = t_steps
        resource.in_transit         = True
        resource.steps_until_free   = t_steps + on_scene
        resource.total_assignments += 1
        resource.total_km_traveled  = round(resource.total_km_traveled + dist_km, 2)

        # Update resource location to incident location (will be there after transit)
        # Do NOT teleport yet — location updates when transit completes
        # We track destination implicitly via current_assignment

        # Mutate incident
        if resource.id not in incident.assigned_resources:
            incident.assigned_resources.append(resource.id)

        return compat, dist_km, t_steps

    # ── Reward computation ─────────────────────────────────────────────────

    @staticmethod
    def compute_assignment_reward(
        compat:           float,
        dist_km:          float,
        incident:         Incident,
        fairness:         FairnessMetrics,
        cascade_count:    int = 0,
    ) -> Reward:
        """
        Reward for an assign_resource action.

        life_saving  = severity × min(people/100, 1)  — direct value of rescue
        response_time = 1 − dist/MAX_DIST             — closer = faster
        efficiency   = compatibility score            — right tool matters
        fairness     = 1 − gini_coefficient           — equitable dispatch
        """
        life_saving  = min(incident.severity * min(incident.people_affected / 100.0, 1.0), 1.0)
        rt_score     = 1.0 - min(dist_km / MAX_DIST_KM, 1.0)
        fairness_s   = max(0.0, 1.0 - fairness.gini_coefficient)
        cascade_pen  = min(cascade_count * 0.2, 1.0)

        bd = RewardBreakdown(
            life_saving_score=round(life_saving, 4),
            response_time_score=round(rt_score, 4),
            resource_efficiency_score=round(compat, 4),
            fairness_score=round(fairness_s, 4),
            cascade_penalty=round(cascade_pen, 4),
        )
        return Reward(
            value       = bd.compute_value(),
            breakdown   = bd,
            explanation = (
                f"assign: compat={compat:.2f} dist={dist_km:.1f}km "
                f"ls={life_saving:.2f} fairness={fairness_s:.2f} casc_pen={cascade_pen:.2f}"
            ),
        )

    @staticmethod
    def compute_wait_reward(
        incidents: List[Incident],
        resources: List[Resource],
        fairness:  FairnessMetrics,
    ) -> Reward:
        """Penalty-heavy reward for waiting when actionable work exists."""
        idle   = sum(1 for r in resources if r.available)
        active = sum(1 for i in incidents if i.is_active)
        frac   = idle / max(len(resources), 1)
        fs     = max(0.0, 1.0 - fairness.gini_coefficient)

        delay_pen = min(frac * 0.8, 1.0) if (idle > 0 and active > 0) else 0.0
        idle_pen  = frac                  if (idle > 0 and active > 0) else 0.0

        bd = RewardBreakdown(
            fairness_score=round(fs, 4),
            delay_penalty=round(delay_pen, 4),
            idle_resource_penalty=round(idle_pen, 4),
        )
        return Reward(
            value       = bd.compute_value(),
            breakdown   = bd,
            explanation = f"wait: idle={idle} active={active}",
        )
