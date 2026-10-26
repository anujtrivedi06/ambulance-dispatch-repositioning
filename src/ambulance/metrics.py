"""Evaluation metrics for a set of simulated calls."""

from __future__ import annotations

import numpy as np

from .city import City
from .sim import CallRecord

TARGET_MINUTES = 8.0


def summarise(
    records: list[CallRecord], city: City, target: float = TARGET_MINUTES
) -> dict[str, float]:
    """Headline metrics. `coverage` is the primary objective; the rest diagnose it."""
    rts = np.array([r.response_time for r in records if r.response_time is not None])
    waits = np.array([r.wait_to_assign for r in records if r.wait_to_assign is not None])
    served = len(rts)

    out = {
        "n_calls": float(len(records)),
        "served": float(served),
        "coverage": float((rts <= target).mean()) if served else 0.0,
        "mean_response": float(rts.mean()) if served else float("nan"),
        "p90_response": float(np.percentile(rts, 90)) if served else float("nan"),
        "max_response": float(rts.max()) if served else float("nan"),
        "mean_queue_wait": float(waits.mean()) if len(waits) else 0.0,
        "queued_frac": float((waits > 1e-9).mean()) if len(waits) else 0.0,
    }
    out["worst_zone_coverage"] = _worst_zone_coverage(records, city, target)
    out["zone_coverage_gap"] = _zone_coverage_gap(records, city, target)
    out.update(_by_priority(records, target))
    return out


def _by_priority(records: list[CallRecord], target: float) -> dict[str, float]:
    """Coverage and response time split by urgency class.

    The city-wide average hides the question selective dispatch is really asking: holding
    a unit back from a routine call is only defensible if the urgent calls gain more than
    the routine ones lose, and a single pooled number cannot show that.
    """
    out: dict[str, float] = {}
    for priority in (1, 2):
        rts = np.array(
            [
                r.response_time
                for r in records
                if r.response_time is not None and r.call.priority == priority
            ]
        )
        out[f"coverage_p{priority}"] = float((rts <= target).mean()) if len(rts) else 0.0
        out[f"mean_response_p{priority}"] = (
            float(rts.mean()) if len(rts) else float("nan")
        )
        out[f"p90_response_p{priority}"] = (
            float(np.percentile(rts, 90)) if len(rts) else float("nan")
        )
        out[f"n_calls_p{priority}"] = float(len(rts))
    return out


def _zone_coverage_gap(records: list[CallRecord], city: City, target: float) -> float:
    """Best zone's coverage minus the worst zone's: how unequal the service is.

    Reported alongside the worst zone because the two can move apart. A policy that drags
    every zone down equally improves the gap while helping nobody.
    """
    per_zone = _zone_coverages(records, city, target)
    return max(per_zone) - min(per_zone) if per_zone else 0.0


def _zone_coverages(
    records: list[CallRecord], city: City, target: float
) -> list[float]:
    by_zone: dict[int, list[float]] = {}
    for r in records:
        if r.response_time is None:
            continue
        by_zone.setdefault(city.zone_of(r.call.node), []).append(r.response_time)
    return [float(np.mean(np.array(v) <= target)) for v in by_zone.values()]


def _worst_zone_coverage(records: list[CallRecord], city: City, target: float) -> float:
    """Equity metric: coverage in the worst-served zone, not the city average.

    A policy can look good on the mean while effectively abandoning a peripheral zone,
    so the project reports both.
    """
    per_zone = _zone_coverages(records, city, target)
    return min(per_zone) if per_zone else 0.0


def format_table(rows: dict[str, dict[str, float]]) -> str:
    """Render {policy_name: metrics} as a fixed-width table."""
    cols = [
        "coverage",
        "mean_response",
        "p90_response",
        "worst_zone_coverage",
        "queued_frac",
        "served",
    ]
    width = max(len(k) for k in rows) + 2
    head = "policy".ljust(width) + "".join(c.rjust(20) for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(name.ljust(width) + "".join(f"{m[c]:>20.3f}" for c in cols))
    return "\n".join(lines)
