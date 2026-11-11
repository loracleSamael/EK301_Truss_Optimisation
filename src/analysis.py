# analysis.py
# Fast Python equilibrium solver used inside the optimisation loop.
#
# Method of joints as one linear system. Unknowns (2J of them for a determinate truss, M = 2J - 3):
#   [F_0 ... F_{M-1}, Rx_pin, Ry_pin, Ry_roller]   (member forces: + tension, - compression)
# Forces scale linearly with the applied load, so one solve at TEST_LOAD gives P_max directly.

from dataclasses import dataclass, field

import numpy as np

import config
from src import geometry

COMPRESSION_TOL = 1e-9      # member forces below -tol (at unit load) count as compression
MAX_CONDITION = 1e10        # above this condition number the system is treated as singular


@dataclass
class AnalysisResult:
    valid: bool                       # False if singular / not square
    reason: str = ""                  # why it is invalid (empty if valid)
    forces: np.ndarray = None         # (M,) member forces at TEST_LOAD (+ tension, - compression)
    reactions: np.ndarray = None      # [Rx_pin, Ry_pin, Ry_roller] at TEST_LOAD
    capacities: np.ndarray = None     # (M,) buckling capacity of each member (oz)
    p_max: float = 0.0                # load at which the first member buckles (oz)
    governing: int = -1               # index of the member that buckles first
    utilisation: np.ndarray = None    # (M,) at P_max: |F| / capacity for compression, 0 for tension
    total_length: float = 0.0
    cost: float = 0.0
    p_over_cost: float = 0.0
    # Buckling-fit uncertainty (config.BUCKLING_UNCERTAINTY, +/- oz on each member's capacity):
    p_max_lower: float = 0.0          # P_max using (capacity - uncertainty): the conservative value
    p_max_uncertainty: float = 0.0    # +/- on P_max from the governing member: uncertainty / f_gov
    governing_lower: int = -1         # member that limits p_max_lower (can differ from `governing`)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
def buckling_capacity(lengths) -> np.ndarray:
    """Buckling load (oz) of a member of the given length (in): F = A * L^B."""
    return config.BUCKLING_A * np.asarray(lengths, dtype=float) ** config.BUCKLING_B


def total_cost(n_joints: int, total_length: float) -> float:
    """Cost = joints x COST_PER_JOINT + total member length x COST_PER_LENGTH."""
    return n_joints * config.COST_PER_JOINT + total_length * config.COST_PER_LENGTH


def equilibrium_matrix(coords, members) -> np.ndarray:
    """(2J, M + 3) joint equilibrium matrix.

    Row 2j is the x balance at joint j and row 2j + 1 the y balance. A member in tension pulls each
    of its end joints towards the other end. The last three columns are the pin Rx, pin Ry and
    roller Ry reactions (each acting in the positive axis direction).
    """
    c = np.asarray(coords, dtype=float)
    m = np.asarray(members, dtype=int)
    n_j, n_m = len(c), len(m)
    A = np.zeros((2 * n_j, n_m + 3))
    vec = c[m[:, 1]] - c[m[:, 0]]
    unit = vec / np.maximum(np.hypot(vec[:, 0], vec[:, 1]), 1e-12)[:, None]   # zero-length: no force
    for k, (i, j) in enumerate(m):
        A[2 * i, k] += unit[k, 0]
        A[2 * i + 1, k] += unit[k, 1]
        A[2 * j, k] -= unit[k, 0]
        A[2 * j + 1, k] -= unit[k, 1]
    A[2 * config.PIN_INDEX, n_m] = 1.0
    A[2 * config.PIN_INDEX + 1, n_m + 1] = 1.0
    A[2 * config.ROLLER_INDEX + 1, n_m + 2] = 1.0
    return A


def _invalid(reason: str) -> AnalysisResult:
    return AnalysisResult(valid=False, reason=reason)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
def analyse(coords, members) -> AnalysisResult:
    """Solve for member forces under TEST_LOAD at the load joint and derive P_max and cost."""
    c = np.asarray(coords, dtype=float)
    m = np.asarray(members, dtype=int).reshape(-1, 2)
    n_j, n_m = len(c), len(m)

    if n_m + 3 != 2 * n_j:
        return _invalid(f"not determinate: M + 3 = {n_m + 3} but 2J = {2 * n_j}")
    lengths = geometry.member_lengths(c, m)
    if np.any(lengths <= 1e-12):
        return _invalid("zero-length member")

    A = equilibrium_matrix(c, m)
    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > MAX_CONDITION:
        return _invalid(f"singular equilibrium matrix (condition number {cond:.3g})")

    b = np.zeros(2 * n_j)
    b[2 * config.LOAD_INDEX + 1] = config.TEST_LOAD      # A x = -(external load), load acts down
    x = np.linalg.solve(A, b)
    forces, reactions = x[:n_m], x[n_m:]

    caps = buckling_capacity(lengths)
    compressed = forces < -COMPRESSION_TOL
    if compressed.any():
        ratios = np.full(n_m, np.inf)
        ratios[compressed] = caps[compressed] / (-forces[compressed])
        governing = int(np.argmin(ratios))
        p_unit = float(ratios[governing])                 # P_max for TEST_LOAD = 1 oz
        p_max = p_unit * config.TEST_LOAD
        # lower bound: every compression member's capacity reduced by the fit uncertainty
        low_caps = np.maximum(caps[compressed] - config.BUCKLING_UNCERTAINTY, 0.0)
        low_ratios = low_caps / (-forces[compressed])
        governing_lower = int(np.flatnonzero(compressed)[np.argmin(low_ratios)])
        p_max_lower = float(low_ratios.min()) * config.TEST_LOAD
        p_max_unc = config.BUCKLING_UNCERTAINTY * config.TEST_LOAD / (-forces[governing])
    else:                                                 # cannot happen for a real bridge
        governing, governing_lower, p_max = -1, -1, float("inf")
        p_max_lower = p_max_unc = float("inf")

    util = np.zeros(n_m)
    if np.isfinite(p_max):
        scale = p_max / config.TEST_LOAD
        util[compressed] = scale * (-forces[compressed]) / caps[compressed]

    total_len = float(lengths.sum())
    cost = total_cost(n_j, total_len)
    return AnalysisResult(
        valid=True, forces=forces, reactions=reactions, capacities=caps, p_max=p_max,
        governing=governing, utilisation=util, total_length=total_len, cost=cost,
        p_over_cost=p_max / cost, p_max_lower=p_max_lower, p_max_uncertainty=float(p_max_unc),
        governing_lower=governing_lower,
        extra={"condition": float(cond)},
    )
