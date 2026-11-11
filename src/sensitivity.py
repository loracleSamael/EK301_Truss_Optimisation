# sensitivity.py
# How much does the buckling-fit uncertainty (+/- config.BUCKLING_UNCERTAINTY oz per member) matter?
#
# The optimiser works with the nominal fit. This module re-scores the finalists with the lower-bound
# fit (capacity - uncertainty) and answers two questions:
#   1. Does the winner, or the order of the top few designs, change?
#   2. Regret: how much better could the winner be, in lower-bound terms, if the geometry were
#      re-optimised against the lower-bound fit? (a local re-optimisation from the winner)

from dataclasses import dataclass, field

import numpy as np

import config
from src import geometry_opt as go
from src import packing, scoring


@dataclass
class SensitivityReport:
    objective: str
    rows: list = field(default_factory=list)       # one dict per design, in nominal order
    winner_changed: bool = False
    order_changed: bool = False
    regret: float = None                           # relative loss of the winner (>= 0), or None
    regret_design: object = None                   # the re-optimised design, if it was better

    def summary_lines(self) -> list:
        lines = [f"Sensitivity to the +/-{config.BUCKLING_UNCERTAINTY:g} oz buckling-fit "
                 f"uncertainty ({self.objective}):"]
        for r in self.rows:
            lines.append(f"  nominal #{r['nominal_rank']} -> lower-bound #{r['lower_rank']}: "
                         f"P_max {r['p_max']:.2f} -> {r['p_max_lower']:.2f} oz "
                         f"({100 * r['drop']:.1f}% lower)")
        if self.winner_changed:
            lines.append("  WARNING: the winner changes under the lower-bound fit.")
        elif self.order_changed:
            lines.append("  Winner unchanged; the order of the other designs changes.")
        else:
            lines.append("  Winner and order unchanged.")
        if self.regret is not None:
            lines.append(f"  Regret of the nominal winner vs re-optimising against the lower "
                         f"bound: {100 * self.regret:.2f}%")
        return lines


def lower_bound_value(design, objective: str) -> float:
    """The quantity being maximised, evaluated with the lower-bound P_max."""
    low = design.evaluation.analysis.p_max_lower
    if objective == scoring.OBJ_MAX_LOAD:
        return low
    if objective == scoring.OBJ_MAX_LOAD_PER_COST:
        return low / design.evaluation.cost
    raise ValueError(f"unknown objective {objective!r}")


def reoptimise_against_lower_bound(design):
    """Best design found by re-running SLSQP from `design` with the capacity offset applied.

    Returns (design, lower-bound value) or (None, None). A local search only: it starts at the
    nominal winner, tries the top stick assignments and keeps the best legal, rounded result.
    """
    cand, objective = design.candidate, design.objective
    lengths = go._lengths(cand, design.coords)
    best, best_value = None, -np.inf
    for pr in packing.top_assignments(lengths, config.PACKING_ASSIGNMENTS_PER_START):
        start, assignment = design.coords, pr.assignment
        for _ in range(go.REPACK_ROUNDS + 1):
            try:
                coords, _ = go.optimise_geometry(cand, start, objective, assignment=assignment,
                                                 offset=config.BUCKLING_UNCERTAINTY)
            except (ValueError, np.linalg.LinAlgError):
                break
            found = go.finalise(cand, coords, objective, polish=False)
            if found is not None:
                value = lower_bound_value(found, objective)
                if value > best_value:
                    best, best_value = found, value
            start = found.coords if found is not None else coords
            nxt = packing.min_overflow(go._lengths(cand, start)).assignment
            if np.array_equal(nxt, assignment):
                break
            assignment = nxt
    return (best, best_value) if best is not None else (None, None)


def sensitivity_check(designs, objective: str, n: int = config.N_FINALISTS,
                      with_regret: bool = True) -> SensitivityReport:
    """Re-score the best `n` designs (best first, nominal order) with the lower-bound fit."""
    top = list(designs[:n])
    report = SensitivityReport(objective)
    if not top:
        return report
    lower = [lower_bound_value(d, objective) for d in top]
    lower_rank = {int(i): r + 1 for r, i in enumerate(np.argsort(-np.asarray(lower), kind="stable"))}
    for i, d in enumerate(top):
        a = d.evaluation.analysis
        report.rows.append({
            "nominal_rank": i + 1, "lower_rank": lower_rank[i], "p_max": a.p_max,
            "p_max_lower": a.p_max_lower, "drop": 1.0 - a.p_max_lower / a.p_max,
            "value_lower": lower[i],
        })
    report.winner_changed = lower_rank[0] != 1
    report.order_changed = any(r["nominal_rank"] != r["lower_rank"] for r in report.rows)
    if with_regret:
        better, value = reoptimise_against_lower_bound(top[0])
        base = lower[0]
        report.regret = max(0.0, (value - base) / value) if better is not None else None
        if better is not None and value > base:
            report.regret_design = better
    return report
