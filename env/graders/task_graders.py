"""
Graders for all three tasks.
Each returns score strictly in (0.0, 1.0).
"""
from __future__ import annotations
from env.models import EpisodeResult, GraderBreakdown


# ── STRICT CLAMP FUNCTION ────────────────────────────────────────────────────
def clamp_score(x: float) -> float:
    return round(max(0.0001, min(0.9999, float(x))), 4)


# ── TASK 1 ───────────────────────────────────────────────────────────────────
class Task1Grader:
    def grade(self, result: EpisodeResult) -> float:
        scores = result.details.get("tau_scores", [])
        bonus  = result.details.get("consistency_bonus", 0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        fairness = result.breakdown.fairness

        raw = 0.80 * avg + 0.20 * fairness + bonus
        return clamp_score(raw)

    def describe(self, result: EpisodeResult) -> str:
        scores = result.details.get("tau_scores", [])
        avg = sum(scores) / len(scores) if scores else 0.0

        return (
            f"Task1Grader | steps={result.total_steps} | mean_tau={avg:.4f} | "
            f"fairness={result.breakdown.fairness:.4f} | "
            f"final={self.grade(result):.4f}"
        )


# ── TASK 2 ───────────────────────────────────────────────────────────────────
class Task2Grader:
    WEIGHTS = {
        "coverage": 0.30,
        "accuracy": 0.25,
        "efficiency": 0.20,
        "time_score": 0.15,
        "fairness": 0.10
    }

    def grade(self, result: EpisodeResult) -> float:
        bd = result.breakdown

        raw = (
            self.WEIGHTS["coverage"]   * bd.coverage +
            self.WEIGHTS["accuracy"]   * bd.accuracy +
            self.WEIGHTS["efficiency"] * bd.efficiency +
            self.WEIGHTS["time_score"] * bd.time_score +
            self.WEIGHTS["fairness"]   * bd.fairness
        )

        return clamp_score(raw)

    def describe(self, result: EpisodeResult) -> str:
        bd = result.breakdown
        return (
            f"Task2Grader | coverage={bd.coverage:.3f} accuracy={bd.accuracy:.3f} "
            f"efficiency={bd.efficiency:.3f} time={bd.time_score:.3f} "
            f"fairness={bd.fairness:.3f} | final={self.grade(result):.4f}"
        )


# ── TASK 3 ───────────────────────────────────────────────────────────────────
class Task3Grader:
    def grade(self, result: EpisodeResult) -> float:
        bd  = result.breakdown
        cap = result.details.get("cascade_penalty", 0.0)

        raw = (
            0.30 * bd.coverage +
            0.25 * bd.accuracy +
            0.20 * bd.time_score +
            0.15 * bd.efficiency +
            0.10 * bd.fairness -
            cap
        )

        return clamp_score(raw)

    def describe(self, result: EpisodeResult) -> str:
        bd  = result.breakdown
        cap = result.details.get("cascade_penalty", 0.0)

        return (
            f"Task3Grader | coverage={bd.coverage:.3f} severity={bd.accuracy:.3f} "
            f"time={bd.time_score:.3f} efficiency={bd.efficiency:.3f} "
            f"fairness={bd.fairness:.3f} cascade_pen={cap:.3f} | "
            f"final={self.grade(result):.4f}"
        )
