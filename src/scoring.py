# scoring.py
# Turns a design into numbers the optimiser can minimise.
#
# Every design is evaluated once into an Evaluation that keeps P_max, cost and the constraint
# violations separately. score() then turns an Evaluation into the number to minimise for a given
# objective, so the metric can be changed without re-running the analysis.
#
# Score ordering (lower is better):
#   feasible, valid design    : -(objective value)            (negative, small magnitude)
#   geometry-infeasible       : PENALTY_INFEASIBLE + violation (graded, so the search is guided)
#   singular / not determinate: PENALTY_SINGULAR   + violation

from dataclasses import dataclass, field

import numpy as np

import config
from src import analysis, geometry, packing

OBJ_MAX_LOAD = "max_load"                  # objective 1: maximise P_max, cost ignored
OBJ_MAX_LOAD_PER_COST = "max_load_per_cost"  # objective 2: maximise P_max / cost
OBJECTIVES = (OBJ_MAX_LOAD, OBJ_MAX_LOAD_PER_COST)


@dataclass
class Evaluation:
    violations: dict                  # rule name -> graded violation (0 when satisfied)
    violation: float                  # sum of violations
    feasible: bool                    # all geometric rules satisfied
    analysis: analysis.AnalysisResult
    p_max: float = 0.0                # 0 if the design could not be analysed
    cost: float = 0.0
    p_over_cost: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """Feasible and analysable: a legal design."""
        return self.feasible and self.analysis.valid


# --------------------------------------------------------------------------------------
# Constraint violations
# --------------------------------------------------------------------------------------
def _excess(value: float, lo: float, hi: float) -> float:
    return max(0.0, lo - value) + max(0.0, value - hi)


def constraint_violations(coords, members) -> dict:
    """Graded violation of each rule in inches (crossings: count). All zeros means feasible."""
    c = np.asarray(coords, dtype=float)
    lengths = geometry.member_lengths(c, members)

    strip = 0.0
    xl = c[config.LOAD_INDEX, 0]
    for j in geometry.joints_in_strip(c):
        strip += config.STRIP_VIOLATION_FLOOR + (config.STRIP_TOLERANCE - abs(c[j, 0] - xl))

    # Supports are on the same level by construction: the pin is at the origin and the roller has
    # y = 0 whenever the design comes from geometry.params_to_coords. Check anyway for safety.
    support_level = (abs(c[config.PIN_INDEX, 1]) + abs(c[config.ROLLER_INDEX, 1])
                     + abs(c[config.PIN_INDEX, 0]))

    return {
        "supports": float(support_level),
        "span": _excess(geometry.span(c), config.SPAN_MIN, config.SPAN_MAX),
        "load_height": _excess(c[config.LOAD_INDEX, 1] - c[config.PIN_INDEX, 1],
                               config.LOAD_HEIGHT_MIN, config.LOAD_HEIGHT_MAX),
        "below_supports": float(max(0.0, -c[:, 1].min())),
        "member_length": float(sum(_excess(L, config.MEMBER_LENGTH_MIN, config.MEMBER_LENGTH_MAX)
                                   for L in lengths)),
        "material": packing.material_violation(lengths),     # stock sticks: 48 + 48 + 24 in
        "strip": float(strip),
        "crossings": config.CROSSING_VIOLATION * geometry.count_crossings(c, members),
    }


# --------------------------------------------------------------------------------------
# Evaluation and scoring
# --------------------------------------------------------------------------------------
def evaluate(coords, members) -> Evaluation:
    """Check every rule and run the analysis once; P_max and cost are logged separately."""
    viol = constraint_violations(coords, members)
    total = float(sum(viol.values()))
    res = analysis.analyse(coords, members)
    ev = Evaluation(violations=viol, violation=total, feasible=total <= config.FEASIBILITY_TOL,
                    analysis=res)
    if res.valid:
        ev.p_max, ev.cost, ev.p_over_cost = res.p_max, res.cost, res.p_over_cost
    return ev


def objective_value(ev: Evaluation, objective: str) -> float:
    """The quantity being maximised for a valid design."""
    if objective == OBJ_MAX_LOAD:
        return ev.p_max - config.COST_TIEBREAK * ev.cost      # cost only breaks exact ties
    if objective == OBJ_MAX_LOAD_PER_COST:
        return ev.p_over_cost
    raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")


def score(ev: Evaluation, objective: str) -> float:
    """Number to minimise. Legal designs always score below any infeasible one."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    if not ev.analysis.valid:
        return config.PENALTY_SINGULAR + ev.violation
    if not ev.feasible:
        return config.PENALTY_INFEASIBLE + ev.violation
    return -objective_value(ev, objective)


def score_design(coords, members, objective: str) -> float:
    """Convenience wrapper: evaluate then score."""
    return score(evaluate(coords, members), objective)
