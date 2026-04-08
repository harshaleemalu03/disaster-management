"""
Graders for all three tasks.
Each returns score in [0.0, 1.0] with a GraderBreakdown.
All graders are fully deterministic given fixed episode results.
"""
from __future__ import annotations
from env.models import EpisodeResult, GraderBreakdown


class Task1Grader:
    """
    Grader: Task 1 — Incident Prioritization (Easy)

    Score = 0.80 * mean_kendall_tau + 0.20 * fairness_score
          + consistency_bonus (up to 0.05)

    breakdown:
      accuracy   = mean Kendall-tau across steps
      fairness   = 1 - gini from last observation
      coverage   = 1.0 (all incidents seen every step)
      efficiency = 0 (no resource allocation in task 1)
      time_score = 0 (no resolution time)
    """

    def grade(self, result: EpisodeResult) -> float:
        scores = result.details.get("tau_scores", [])
        bonus  = result.details.get("consistency_bonus", 0.0)
        avg    = sum(scores) / len(scores) if scores else 0.0
        fairness = result.breakdown.fairness
        return round(min(0.80 * avg + 0.20 * fairness + bonus, 1.0), 4)

    def describe(self, result: EpisodeResult) -> str:
        scores = result.details.get("tau_scores", [])
        avg    = sum(scores) / len(scores) if scores else 0.0
        return (
            f"Task1Grader | steps={result.total_steps} | mean_tau={avg:.4f} | "
            f"fairness={result.breakdown.fairness:.4f} | "
            f"final={self.grade(result):.4f}"
        )


class Task2Grader:
    """
    Grader: Task 2 — Resource Allocation (Medium)

    Score = 0.30 * coverage       (resolution rate)
          + 0.25 * accuracy       (severity-weighted rescues)
          + 0.20 * efficiency     (compat × proximity)
          + 0.15 * time_score     (fast response)
          + 0.10 * fairness       (equitable zone dispatch)

    All components in [0,1].
    """

    WEIGHTS = {"coverage": 0.30, "accuracy": 0.25, "efficiency": 0.20,
               "time_score": 0.15, "fairness": 0.10}

    def grade(self, result: EpisodeResult) -> float:
        bd = result.breakdown
        raw = (
            self.WEIGHTS["coverage"]   * bd.coverage
            + self.WEIGHTS["accuracy"] * bd.accuracy
            + self.WEIGHTS["efficiency"] * bd.efficiency
            + self.WEIGHTS["time_score"] * bd.time_score
            + self.WEIGHTS["fairness"]   * bd.fairness
        )
        return round(min(max(raw, 0.0), 1.0), 4)

    def describe(self, result: EpisodeResult) -> str:
        bd = result.breakdown
        return (
            f"Task2Grader | coverage={bd.coverage:.3f} accuracy={bd.accuracy:.3f} "
            f"efficiency={bd.efficiency:.3f} time={bd.time_score:.3f} "
            f"fairness={bd.fairness:.3f} | final={self.grade(result):.4f}"
        )


class Task3Grader:
    """
    Grader: Task 3 — Dynamic Coordination (Hard)

    Score = 0.30 * coverage       (fraction of all spawned incidents resolved)
          + 0.25 * accuracy       (severity-weighted rescues)
          + 0.20 * time_score     (speed of resolution vs expiry window)
          + 0.15 * efficiency     (resource type match × distance)
          + 0.10 * fairness       (Gini across zones)
          - cascade_penalty       (up to 0.20 for uncontrolled fire spread)

    Clamped to [0, 1].
    """

    def grade(self, result: EpisodeResult) -> float:
        bd  = result.breakdown
        cap = result.details.get("cascade_penalty", 0.0)
        raw = (
            0.30 * bd.coverage
            + 0.25 * bd.accuracy
            + 0.20 * bd.time_score
            + 0.15 * bd.efficiency
            + 0.10 * bd.fairness
            - cap
        )
        return round(min(max(raw, 0.0), 1.0), 4)

    def describe(self, result: EpisodeResult) -> str:
        bd  = result.breakdown
        cap = result.details.get("cascade_penalty", 0.0)
        return (
            f"Task3Grader | coverage={bd.coverage:.3f} severity={bd.accuracy:.3f} "
            f"time={bd.time_score:.3f} efficiency={bd.efficiency:.3f} "
            f"fairness={bd.fairness:.3f} cascade_pen={cap:.3f} | "
            f"final={self.grade(result):.4f}"
        )
