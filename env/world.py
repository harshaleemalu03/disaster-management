"""
env/world.py — Shared deterministic world simulation engine v2.2
================================================================
Physics, escalation, cascade, resolution, fairness — used by all 3 tasks.
Also contains the UI: build_text_grid() and build_html_view().
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
        urban_response_rate    = round(rates[0], 4),
        suburban_response_rate = round(rates[1], 4),
        rural_response_rate    = round(rates[2], 4),
        gini_coefficient       = round(min(gini, 1.0), 4),
        neglected_rural_steps  = neglected,
    )


# ── Text grid ─────────────────────────────────────────────────────────────────

def build_text_grid(
    incidents: List[Incident],
    resources: List[Resource],
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
) -> str:
    from env.models import GRID_KM
    cell: Dict[Tuple[int,int], str] = {}

    for inc in incidents:
        if not inc.is_active:
            continue
        c = min(int(inc.location.x / GRID_KM * cols), cols - 1)
        r = min(int(inc.location.y / GRID_KM * rows), rows - 1)
        key = (r, c)
        existing = cell.get(key, "")
        cell[key] = existing + inc.emoji() if existing else inc.emoji()

    for res in resources:
        c = min(int(res.location.x / GRID_KM * cols), cols - 1)
        r = min(int(res.location.y / GRID_KM * rows), rows - 1)
        key = (r, c)
        if key not in cell:
            cell[key] = res.emoji()

    lines = []
    for row in range(rows - 1, -1, -1):
        parts = []
        for col in range(cols):
            icon = cell.get((row, col), " ")
            parts.append(f"[{icon:<2}]")
        lines.append(" ".join(parts))
    return "\n".join(lines)


# ── Beautiful HTML dashboard ──────────────────────────────────────────────────

def build_html_view(
    incidents: List[Incident],
    resources: List[Resource],
    timestep:  int,
    task_id:   str,
    max_timesteps: int = 30,
    fairness:  Optional[FairnessMetrics] = None,
    step_logs: Optional[List[StepLog]]   = None,
    cascade_events: Optional[List[CascadeEvent]] = None,
    resolved_count: int = 0,
    failed_count:   int = 0,
    active_count:   int = 0,
    lives_saved:    int = 0,
    cumulative_reward: float = 0.0,
) -> str:
    from env.models import GRID_KM

    COLS, ROWS = 12, 12

    # ── Build cell data ──────────────────────────────────────────────────
    INC_COLORS = {
        "fire":       "#ff4500",
        "medical":    "#e91e8c",
        "flood":      "#1565c0",
        "earthquake": "#7b1fa2",
        "accident":   "#f57c00",
        "hazmat":     "#2e7d32",
    }
    RES_COLORS = {
        "available": "#00c853",
        "transit":   "#ff9100",
        "busy":      "#d50000",
    }

    # cell_map: (row,col) → list of dicts
    cell_map: Dict[Tuple[int,int], List[Dict]] = {}

    for inc in incidents:
        if not inc.is_active:
            continue
        c   = min(int(inc.location.x / GRID_KM * COLS), COLS - 1)
        r   = min(int(inc.location.y / GRID_KM * ROWS), ROWS - 1)
        key = (r, c)
        cell_map.setdefault(key, [])
        cell_map[key].append({
            "emoji":   inc.emoji(),
            "color":   INC_COLORS.get(inc.itype_str(), "#888"),
            "label":   f"{inc.id} {inc.itype_str()} sev={inc.severity:.2f} ppl={inc.people_affected}",
            "kind":    "incident",
            "severity": inc.severity,
            "expiry":  inc.steps_to_expiry,
        })

    for res in resources:
        c   = min(int(res.location.x / GRID_KM * COLS), COLS - 1)
        r   = min(int(res.location.y / GRID_KM * ROWS), ROWS - 1)
        key = (r, c)
        if key in cell_map:
            continue    # incidents take priority
        cell_map.setdefault(key, [])
        if res.available:
            color  = RES_COLORS["available"]
            status = "FREE"
        elif res.in_transit:
            color  = RES_COLORS["transit"]
            status = f"TRANSIT({res.steps_in_transit}s)"
        else:
            color  = RES_COLORS["busy"]
            status = f"BUSY({res.steps_until_free}s)"
        cell_map[key].append({
            "emoji":  res.emoji(),
            "color":  color,
            "label":  f"{res.id} {res.rtype_str()} {status}",
            "kind":   "resource",
        })

    # ── Build grid HTML ──────────────────────────────────────────────────
    CELL_PX = 60
    grid_rows_html = []
    for row in range(ROWS - 1, -1, -1):
        cells_html = []
        for col in range(COLS):
            items = cell_map.get((row, col), [])
            if not items:
                cells_html.append(
                    f'<td style="width:{CELL_PX}px;height:{CELL_PX}px;'
                    f'background:#0d1117;border:1px solid #21262d;'
                    f'text-align:center;vertical-align:middle;">'
                    f'<span style="opacity:0.08;font-size:10px">·</span></td>'
                )
            else:
                item   = items[0]
                color  = item["color"]
                label  = item["label"].replace('"', "'")
                emojis = "".join(i["emoji"] for i in items[:2])
                # Pulse animation for high-severity incidents
                pulse  = ""
                if item.get("kind") == "incident" and item.get("severity", 0) > 0.7:
                    pulse = "animation:pulse 1.5s infinite;"

                cells_html.append(
                    f'<td title="{label}" style="width:{CELL_PX}px;height:{CELL_PX}px;'
                    f'background:{color}22;border:2px solid {color}88;'
                    f'text-align:center;vertical-align:middle;cursor:default;'
                    f'border-radius:6px;{pulse}">'
                    f'<span style="font-size:22px;line-height:1">{emojis}</span></td>'
                )
        grid_rows_html.append(f"<tr>{''.join(cells_html)}</tr>")

    grid_html = "\n".join(grid_rows_html)

    # ── Incidents table ──────────────────────────────────────────────────
    active_incs = [i for i in incidents if i.is_active]
    active_incs.sort(key=lambda i: -i.urgency_score())

    inc_rows = ""
    for inc in active_incs:
        zone    = _v(inc.zone)
        status  = _v(inc.status)
        assigned = ", ".join(inc.assigned_resources) if inc.assigned_resources else "—"
        sev_pct  = int(inc.severity * 100)
        sev_color = "#ff4500" if inc.severity > 0.7 else ("#f57c00" if inc.severity > 0.4 else "#2e7d32")
        expiry_color = "#ff4500" if inc.steps_to_expiry <= 3 else ("#f57c00" if inc.steps_to_expiry <= 6 else "#8b949e")
        inc_rows += f"""
        <tr style="border-bottom:1px solid #21262d">
          <td style="padding:6px 8px">{inc.emoji()} <strong>{inc.id}</strong></td>
          <td style="padding:6px 8px;color:{sev_color}">
            <div style="background:#21262d;border-radius:4px;height:8px;width:80px;display:inline-block;vertical-align:middle;margin-right:4px">
              <div style="background:{sev_color};height:8px;width:{sev_pct}px;border-radius:4px"></div>
            </div>
            {inc.severity:.2f}
          </td>
          <td style="padding:6px 8px">{inc.people_affected:,}</td>
          <td style="padding:6px 8px;color:{expiry_color};font-weight:bold">{inc.steps_to_expiry}s</td>
          <td style="padding:6px 8px;opacity:0.7">{zone}</td>
          <td style="padding:6px 8px;opacity:0.7;font-size:11px">{assigned}</td>
        </tr>"""

    # ── Resources table ──────────────────────────────────────────────────
    res_rows = ""
    for res in resources:
        if res.available:
            status_html = '<span style="color:#00c853;font-weight:bold">✓ FREE</span>'
        elif res.in_transit:
            status_html = f'<span style="color:#ff9100">→ TRANSIT ({res.steps_in_transit}s)</span>'
        else:
            status_html = f'<span style="color:#d50000">● BUSY ({res.steps_until_free}s) → {res.current_assignment or "?"}</span>'
        zone = _v(res.location.zone)
        res_rows += f"""
        <tr style="border-bottom:1px solid #21262d">
          <td style="padding:6px 8px">{res.emoji()} <strong>{res.id}</strong></td>
          <td style="padding:6px 8px;opacity:0.8">{res.rtype_str()}</td>
          <td style="padding:6px 8px">{status_html}</td>
          <td style="padding:6px 8px;opacity:0.7">({res.location.x:.0f}, {res.location.y:.0f})</td>
          <td style="padding:6px 8px;opacity:0.7">{zone}</td>
          <td style="padding:6px 8px;opacity:0.7">{res.total_km_traveled:.0f} km</td>
        </tr>"""

    # ── Step log ─────────────────────────────────────────────────────────
    log_rows = ""
    if step_logs:
        for entry in reversed(step_logs[-8:]):
            r_color = "#00c853" if entry.reward > 0.5 else ("#f57c00" if entry.reward > 0.2 else "#d50000")
            log_rows += f"""
            <tr style="border-bottom:1px solid #21262d">
              <td style="padding:4px 8px;opacity:0.6">#{entry.step}</td>
              <td style="padding:4px 8px;font-size:11px;font-family:monospace">{entry.action_detail[:40]}</td>
              <td style="padding:4px 8px;color:{r_color};font-weight:bold">{entry.reward:.3f}</td>
              <td style="padding:4px 8px;opacity:0.6;font-size:11px">{entry.explanation[:40]}</td>
            </tr>"""

    # ── Cascade log ───────────────────────────────────────────────────────
    cascade_html = ""
    if cascade_events:
        cascade_html = """
        <div style="background:#1a0000;border:1px solid #ff4500;border-radius:8px;padding:12px;margin-top:16px">
          <h3 style="color:#ff4500;margin:0 0 8px">🔥 Cascade Fire Events</h3>
          <table style="width:100%;border-collapse:collapse;font-size:13px">"""
        for ce in cascade_events[-5:]:
            cascade_html += f"""
            <tr>
              <td style="padding:3px 8px;opacity:0.7">Step {ce.trigger_step}</td>
              <td style="padding:3px 8px">🔥 from {ce.source_incident_id}</td>
              <td style="padding:3px 8px;color:#ff4500">sev={ce.severity:.2f}</td>
              <td style="padding:3px 8px;opacity:0.6">@({ce.location.x:.0f},{ce.location.y:.0f})</td>
            </tr>"""
        cascade_html += "</table></div>"

    # ── Fairness bars ─────────────────────────────────────────────────────
    def zone_bar(rate: float, label: str, color: str) -> str:
        pct = int(rate * 100)
        return f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span style="font-size:12px;opacity:0.7">{label}</span>
            <span style="font-size:12px;color:{color};font-weight:bold">{pct}%</span>
          </div>
          <div style="background:#21262d;border-radius:4px;height:8px">
            <div style="background:{color};height:8px;width:{pct}%;border-radius:4px;transition:width 0.5s"></div>
          </div>
        </div>"""

    fairness_html = ""
    if fairness:
        gini_color = "#00c853" if fairness.gini_coefficient < 0.2 else ("#f57c00" if fairness.gini_coefficient < 0.5 else "#d50000")
        fairness_html = f"""
        <div style="background:#161b22;border-radius:8px;padding:14px;margin-top:16px">
          <h3 style="margin:0 0 12px;color:#e6edf3;font-size:14px">⚖️ Dispatch Fairness</h3>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
            <span style="opacity:0.7;font-size:13px">Gini Coefficient</span>
            <span style="font-size:18px;font-weight:bold;color:{gini_color}">{fairness.gini_coefficient:.3f}</span>
          </div>
          {zone_bar(fairness.urban_response_rate, "🏙️ Urban", "#58a6ff")}
          {zone_bar(fairness.suburban_response_rate, "🏘️ Suburban", "#a371f7")}
          {zone_bar(fairness.rural_response_rate, "🌲 Rural", "#3fb950")}
          {"<p style='color:#f57c00;font-size:12px;margin:8px 0 0'>⚠️ Rural incidents being neglected!</p>" if fairness.neglected_rural_steps > 2 else ""}
        </div>"""

    # ── Progress bar ──────────────────────────────────────────────────────
    progress_pct = int((timestep / max(max_timesteps, 1)) * 100)
    progress_color = "#00c853" if progress_pct < 50 else ("#f57c00" if progress_pct < 80 else "#d50000")

    # ── Difficulty badge ──────────────────────────────────────────────────
    diff_map = {
        "task1_prioritization":       ("EASY",   "#00c853"),
        "task2_resource_allocation":  ("MEDIUM", "#f57c00"),
        "task3_dynamic_coordination": ("HARD",   "#d50000"),
    }
    diff_label, diff_color = diff_map.get(task_id, ("?", "#888"))

    # ── Full HTML ─────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="3">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🚨 Disaster Control Room — Step {timestep}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    min-height: 100vh;
    padding: 16px;
  }}
  h2 {{ font-size: 13px; color: #8b949e; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
  }}
  .stat-value {{ font-size: 28px; font-weight: 700; line-height: 1.1; }}
  .stat-label {{ font-size: 11px; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.5px; }}
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }}
  @keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(255,69,0,0.6); }}
    70%  {{ box-shadow: 0 0 0 8px rgba(255,69,0,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(255,69,0,0); }}
  }}
  @keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.4; }}
  }}
  .live-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    background: #00c853;
    border-radius: 50%;
    animation: blink 1.5s infinite;
    margin-right: 6px;
  }}
  tr:hover {{ background: #1c2128; }}
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: #0d1117; }}
  ::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
</style>
</head>
<body>

<!-- ── HEADER ─────────────────────────────────────────────────────── -->
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px">
  <div>
    <h1 style="font-size:1.5rem;color:#58a6ff;display:flex;align-items:center;gap:8px">
      🚨 Disaster Response Control Room
    </h1>
    <p style="opacity:0.6;font-size:13px;margin-top:2px">{task_id.replace("_"," ").title()}</p>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <span class="badge" style="background:{diff_color}22;color:{diff_color};border:1px solid {diff_color}55">{diff_label}</span>
    <span style="font-size:13px;opacity:0.7"><span class="live-dot"></span>LIVE · auto-refresh 3s</span>
  </div>
</div>

<!-- ── STEP PROGRESS ──────────────────────────────────────────────── -->
<div class="card" style="padding:12px 16px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <span style="font-size:13px;opacity:0.7">Episode Progress</span>
    <span style="font-weight:700;color:{progress_color}">Step {timestep} / {max_timesteps}</span>
  </div>
  <div style="background:#21262d;border-radius:6px;height:10px">
    <div style="background:linear-gradient(90deg,{progress_color},{progress_color}88);height:10px;
                width:{progress_pct}%;border-radius:6px;transition:width 0.5s"></div>
  </div>
</div>

<!-- ── STAT CARDS ──────────────────────────────────────────────────── -->
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:16px">
  <div class="card" style="text-align:center">
    <div class="stat-value" style="color:#f57c00">{active_count}</div>
    <div class="stat-label">Active Incidents</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="stat-value" style="color:#00c853">{resolved_count}</div>
    <div class="stat-label">Resolved</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="stat-value" style="color:#d50000">{failed_count}</div>
    <div class="stat-label">Failed</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="stat-value" style="color:#58a6ff">{lives_saved:,}</div>
    <div class="stat-label">Lives Saved</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="stat-value" style="color:#a371f7">{cumulative_reward:.2f}</div>
    <div class="stat-label">Cum. Reward</div>
  </div>
  <div class="card" style="text-align:center">
    <div class="stat-value" style="color:#ff9100">{len(cascade_events or [])}</div>
    <div class="stat-label">Cascades</div>
  </div>
</div>

<!-- ── TWO COLUMN LAYOUT ───────────────────────────────────────────── -->
<div style="display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:start">

  <!-- LEFT: Grid Map -->
  <div>
    <div class="card" style="padding:12px">
      <h2 style="margin-bottom:12px">🗺️ Disaster Scene Map</h2>
      <div style="overflow:auto">
        <table style="border-collapse:separate;border-spacing:3px;table-layout:fixed">
          {grid_html}
        </table>
      </div>
      <div style="margin-top:10px;font-size:11px;opacity:0.5;line-height:1.8">
        <strong>Incidents:</strong>
        🔥 Fire &nbsp; 🏥 Medical &nbsp; 🌊 Flood &nbsp; ⚡ Earthquake &nbsp; 💥 Accident &nbsp; ☣️ Hazmat<br>
        <strong>Resources:</strong>
        🚑 Ambulance &nbsp; 🚒 Fire Truck &nbsp; 🛟 Rescue &nbsp; 🚓 Police &nbsp; 🚁 Helicopter<br>
        <span style="color:#00c853">■</span> Free &nbsp;
        <span style="color:#ff9100">■</span> In Transit &nbsp;
        <span style="color:#d50000">■</span> Busy
      </div>
    </div>
    {fairness_html}
    {cascade_html}
  </div>

  <!-- RIGHT: Tables -->
  <div>

    <!-- Incidents Table -->
    <div class="card">
      <h2>⚠️ Active Incidents ({active_count})</h2>
      {"<p style='opacity:0.4;font-size:13px;padding-top:4px'>No active incidents</p>" if not active_incs else f'''
      <table>
        <thead>
          <tr style="border-bottom:1px solid #30363d;color:#8b949e">
            <th style="padding:6px 8px;text-align:left;font-weight:500">ID</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Severity</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">People</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Expiry</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Zone</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Assigned</th>
          </tr>
        </thead>
        <tbody>{inc_rows}</tbody>
      </table>'''}
    </div>

    <!-- Resources Table -->
    <div class="card">
      <h2>🚒 Resources ({len(resources)})</h2>
      <table>
        <thead>
          <tr style="border-bottom:1px solid #30363d;color:#8b949e">
            <th style="padding:6px 8px;text-align:left;font-weight:500">ID</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Type</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Status</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Location</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Zone</th>
            <th style="padding:6px 8px;text-align:left;font-weight:500">Traveled</th>
          </tr>
        </thead>
        <tbody>{res_rows}</tbody>
      </table>
    </div>

    <!-- Step Log -->
    {"" if not step_logs else f'''
    <div class="card">
      <h2>📋 Recent Actions</h2>
      <table>
        <thead>
          <tr style="border-bottom:1px solid #30363d;color:#8b949e">
            <th style="padding:4px 8px;text-align:left;font-weight:500">Step</th>
            <th style="padding:4px 8px;text-align:left;font-weight:500">Action</th>
            <th style="padding:4px 8px;text-align:left;font-weight:500">Reward</th>
            <th style="padding:4px 8px;text-align:left;font-weight:500">Detail</th>
          </tr>
        </thead>
        <tbody>{log_rows}</tbody>
      </table>
    </div>'''}

  </div>
</div>

<p style="text-align:center;opacity:0.3;font-size:11px;margin-top:16px;padding-bottom:8px">
  Disaster Response Coordination System v2.2 · OpenEnv · Auto-refreshes every 3 seconds
</p>

</body>
</html>"""


# ── WorldEngine ───────────────────────────────────────────────────────────────

class WorldEngine:

    @staticmethod
    def age_incidents(incidents: List[Incident], timestep: int) -> None:
        for inc in incidents:
            if inc.is_active:
                inc.time_since_report += 1.0
                inc.steps_to_expiry   -= 1

    @staticmethod
    def escalate_incidents(incidents: List[Incident]) -> None:
        for inc in incidents:
            if not inc.is_active:
                continue
            itype = inc.itype_str()
            if not inc.assigned_resources:
                rate   = ESCALATION_RATE.get(itype, 0.03)
                growth = PEOPLE_GROWTH.get(itype, 1.0)
                inc.severity        = round(min(inc.severity + rate, 1.0), 4)
                inc.people_affected = int(math.ceil(inc.people_affected * growth))
                inc.status          = IncidentStatus.ACTIVE
            else:
                inc.steps_being_treated += 1
                inc.status = IncidentStatus.CONTAINED

    @staticmethod
    def advance_resources(resources: List[Resource]) -> None:
        for res in resources:
            if res.in_transit:
                res.steps_in_transit -= 1
                if res.steps_in_transit <= 0:
                    res.steps_in_transit = 0
                    res.in_transit       = False
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
        timestep:  int,
    ) -> List[str]:
        res_map         = {r.id: r for r in resources}
        newly_resolved: List[str] = []

        for inc in incidents:
            if not inc.is_active or not inc.assigned_resources:
                continue
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
                for rid in inc.assigned_resources:
                    if rid in res_map:
                        r                        = res_map[rid]
                        r.available              = True
                        r.steps_until_free       = 0
                        r.in_transit             = False
                        r.steps_in_transit       = 0
                        r.current_assignment     = None
                        r.successful_assignments += 1

        return newly_resolved

    @staticmethod
    def expire_incidents(incidents: List[Incident], timestep: int) -> List[str]:
        failed: List[str] = []
        for inc in incidents:
            if inc.is_active and inc.steps_to_expiry <= 0:
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
        new_cascades: List[CascadeEvent] = []

        for inc in list(incidents):
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

            angle = rng.uniform(0.0, 2.0 * math.pi)
            dist  = rng.uniform(3.0, CASCADE_SPREAD_DIST_KM)
            nx    = min(max(inc.location.x + dist * math.cos(angle), 2.0), 98.0)
            ny    = min(max(inc.location.y + dist * math.sin(angle), 2.0), 98.0)

            counter[0] += 1
            new_id = f"INC-{counter[0]:03d}"
            zone   = inc.zone

            new_inc = Incident(
                id                     = new_id,
                incident_type          = IncidentType.FIRE,
                severity               = round(rng.uniform(0.40, 0.65), 3),
                people_affected        = rng.randint(5, 40),
                location               = Location(x=round(nx, 1), y=round(ny, 1), zone=zone),
                time_since_report      = 0.0,
                status                 = IncidentStatus.ACTIVE,
                required_resource_types = [ResourceType.FIRE_TRUCK],
                base_resolution_steps  = 4,
                steps_to_expiry        = EXPIRY_STEPS["fire"],
                zone                   = zone,
                population_density     = round(rng.uniform(0.2, 0.8), 2),
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

    @staticmethod
    def apply_assignment(
        resource:  Resource,
        incident:  Incident,
        timestep:  int,
    ) -> Tuple[float, float, int]:
        dist_km   = resource.location.distance_to(incident.location)
        zone      = _v(incident.location.zone)
        t_steps   = travel_steps(resource.rtype_str(), dist_km, zone)
        compat    = get_compatibility(resource.rtype_str(), incident.itype_str())
        on_scene  = max(1, int(incident.base_resolution_steps * (1.1 - compat * 0.5)))

        resource.available          = False
        resource.current_assignment = incident.id
        resource.steps_in_transit   = t_steps
        resource.in_transit         = True
        resource.steps_until_free   = t_steps + on_scene
        resource.total_assignments += 1
        resource.total_km_traveled  = round(resource.total_km_traveled + dist_km, 2)

        if resource.id not in incident.assigned_resources:
            incident.assigned_resources.append(resource.id)

        return compat, dist_km, t_steps

    @staticmethod
    def compute_assignment_reward(
        compat:        float,
        dist_km:       float,
        incident:      Incident,
        fairness:      FairnessMetrics,
        cascade_count: int = 0,
    ) -> Reward:
        life_saving  = min(incident.severity * min(incident.people_affected / 100.0, 1.0), 1.0)
        rt_score     = 1.0 - min(dist_km / MAX_DIST_KM, 1.0)
        fairness_s   = max(0.0, 1.0 - fairness.gini_coefficient)
        cascade_pen  = min(cascade_count * 0.2, 1.0)

        bd = RewardBreakdown(
            life_saving_score          = round(life_saving, 4),
            response_time_score        = round(rt_score,    4),
            resource_efficiency_score  = round(compat,      4),
            fairness_score             = round(fairness_s,  4),
            cascade_penalty            = round(cascade_pen, 4),
        )
        return Reward(
            value       = bd.compute_value(),
            breakdown   = bd,
            explanation = (
                f"assign: compat={compat:.2f} dist={dist_km:.1f}km "
                f"ls={life_saving:.2f} fairness={fairness_s:.2f}"
            ),
        )

    @staticmethod
    def compute_wait_reward(
        incidents: List[Incident],
        resources: List[Resource],
        fairness:  FairnessMetrics,
    ) -> Reward:
        idle   = sum(1 for r in resources if r.available)
        active = sum(1 for i in incidents if i.is_active)
        frac   = idle / max(len(resources), 1)
        fs     = max(0.0, 1.0 - fairness.gini_coefficient)
        delay_pen = min(frac * 0.8, 1.0) if (idle > 0 and active > 0) else 0.0
        idle_pen  = frac                  if (idle > 0 and active > 0) else 0.0

        bd = RewardBreakdown(
            fairness_score        = round(fs,        4),
            delay_penalty         = round(delay_pen, 4),
            idle_resource_penalty = round(idle_pen,  4),
        )
        return Reward(
            value       = bd.compute_value(),
            breakdown   = bd,
            explanation = f"wait: idle={idle} active={active}",
        )
