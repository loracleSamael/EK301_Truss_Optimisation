# packing.py
# Stock-length constraint: every member is cut from ONE stick (config.STOCK_LENGTHS, i.e. two 48 in
# sticks and one 24 in stick) and uses exactly its joint-to-joint length.
#
# The question "can these member lengths be cut from the sticks?" is a small bin-packing problem.
# With at most 12 members and 3 sticks it is solved exactly by trying every assignment of members
# to sticks (3^12 = 531k at most, halved because the two 48 in sticks are interchangeable). The
# table of assignments is built once per member count and reused; a lengths vector is then scored
# against all assignments with one matrix product.
#
# Overflow of an assignment = sum over sticks of max(0, load - capacity). The minimum over all
# assignments is 0 exactly when the members can be cut from the sticks. It is also the graded
# violation used by scoring.py.

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import config

TOL = config.FEASIBILITY_TOL
STICK_LETTERS = "ABCDEFGH"


@dataclass
class PackResult:
    overflow: float             # total overflow over all sticks (0 = packable)
    assignment: np.ndarray      # stick index of each member
    loads: np.ndarray           # total member length assigned to each stick

    @property
    def packable(self) -> bool:
        return self.overflow <= TOL


def capacities() -> np.ndarray:
    return np.asarray(config.STOCK_LENGTHS, dtype=float)


# --------------------------------------------------------------------------------------
# Exact search
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _table(n_members: int, caps: tuple):
    """All distinct assignments of n_members to sticks: (digits (n, M) int8, masks (k, n, M))."""
    k = len(caps)
    n = k ** n_members
    digits = (np.arange(n, dtype=np.int64)[:, None] // (k ** np.arange(n_members))) % k
    keep = np.ones(n, dtype=bool)
    for a in range(k):
        for b in range(a + 1, k):
            if caps[a] != caps[b]:
                continue
            # interchangeable sticks: the one holding the lowest-numbered member is the lower index
            in_a, in_b = digits == a, digits == b
            first_a = np.where(in_a.any(axis=1), in_a.argmax(axis=1), n_members)
            first_b = np.where(in_b.any(axis=1), in_b.argmax(axis=1), n_members)
            keep &= first_a <= first_b
    digits = digits[keep]
    masks = np.stack([(digits == b) for b in range(k)]).astype(np.float64)
    return digits.astype(np.int8), masks


def _scan(lengths: np.ndarray):
    """Overflow and worst slack of every assignment (exact search)."""
    caps = capacities()
    digits, masks = _table(len(lengths), tuple(caps))
    loads = masks @ lengths                                  # (k, n)
    slack = loads - caps[:, None]
    return digits, loads, np.maximum(slack, 0.0).sum(axis=0), slack.max(axis=0)


def _first_fit_decreasing(lengths: np.ndarray) -> PackResult:
    """Fallback for member counts beyond the exact search: a valid (not always minimal) packing."""
    caps = capacities()
    remaining = caps.copy()
    assignment = np.zeros(len(lengths), dtype=int)
    for i in np.argsort(-lengths, kind="stable"):
        fits = np.flatnonzero(remaining >= lengths[i] - 1e-12)
        s = int(fits[np.argmin(remaining[fits])]) if len(fits) else int(np.argmax(remaining))
        assignment[i] = s
        remaining[s] -= lengths[i]
    loads = np.array([lengths[assignment == s].sum() for s in range(len(caps))])
    return PackResult(float(np.maximum(loads - caps, 0.0).sum()), assignment, loads)


def min_overflow(lengths) -> PackResult:
    """The assignment with the least total overflow (ties: the one with the most spare room)."""
    lengths = np.asarray(lengths, dtype=float).ravel()
    if len(lengths) == 0:
        return PackResult(0.0, np.zeros(0, dtype=int), np.zeros(len(config.STOCK_LENGTHS)))
    if len(lengths) > config.MAX_EXACT_PACKING_MEMBERS:
        return _first_fit_decreasing(lengths)
    digits, loads, over, worst = _scan(lengths)
    best = np.flatnonzero(over <= over.min() + 1e-12)
    i = best[np.argmin(worst[best])]
    return PackResult(float(over[i]), digits[i].astype(int), loads[:, i])


def top_assignments(lengths, k: int) -> list:
    """The k best distinct assignments (least overflow, then most spare room)."""
    lengths = np.asarray(lengths, dtype=float).ravel()
    if len(lengths) == 0 or len(lengths) > config.MAX_EXACT_PACKING_MEMBERS:
        return [min_overflow(lengths)]
    digits, loads, over, worst = _scan(lengths)
    order = np.lexsort((worst, over))[:k]
    return [PackResult(float(over[i]), digits[i].astype(int), loads[:, i]) for i in order]


def is_packable(lengths) -> bool:
    return min_overflow(lengths).packable


def material_violation(lengths) -> float:
    """Graded violation for scoring: 0 if packable, otherwise the overflow in inches.

    When the total length already exceeds all the stock, the excess is returned directly (it is a
    lower bound on the overflow and avoids the search for clearly illegal designs).
    """
    lengths = np.asarray(lengths, dtype=float).ravel()
    excess = float(lengths.sum()) - config.MATERIAL_BUDGET
    if excess > TOL:
        return excess
    if len(lengths) and _first_fit_decreasing(lengths).overflow <= TOL:
        return 0.0                      # a quick greedy packing already works: certainly packable
    ov = min_overflow(lengths).overflow
    return ov if ov > TOL else 0.0


# --------------------------------------------------------------------------------------
# Cutting plan
# --------------------------------------------------------------------------------------
def stick_name(s: int) -> str:
    return f"{STICK_LETTERS[s]} ({config.STOCK_LENGTHS[s]:g} in)"


def cutting_plan(lengths, assignment=None) -> list:
    """Per stick: its name, capacity, the members cut from it (0-based), length used and left over."""
    lengths = np.asarray(lengths, dtype=float).ravel()
    if assignment is None:
        assignment = min_overflow(lengths).assignment
    plan = []
    for s, cap in enumerate(config.STOCK_LENGTHS):
        members = [int(m) for m in np.flatnonzero(np.asarray(assignment) == s)]
        used = float(lengths[members].sum()) if members else 0.0
        plan.append({"stick": stick_name(s), "capacity": float(cap), "members": members,
                     "used": used, "left": float(cap) - used})
    return plan
