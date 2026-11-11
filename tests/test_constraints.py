# Known good and bad geometries for each rule:
# span range, member length range, load joint height, strip under the load joint,
# nothing below the supports, material budget, member crossings, plus scoring behaviour.

import numpy as np
import pytest

import config
from src import geometry as g
from src import scoring as sc
from tests.test_packing import brute_force_overflow

# Seven-joint Warren-style truss that satisfies every rule (total length ~105 in).
GOOD_PARAMS = [30.0, 20.0, 1.0, 10.0, 0.0, 5.0, 8.0, 15.0, 8.0, 25.0, 8.0]
GOOD_MEMBERS = [(0, 3), (3, 2), (2, 1), (4, 5), (5, 6),
                (0, 4), (4, 3), (3, 5), (5, 2), (2, 6), (6, 1)]


def good():
    return g.params_to_coords(GOOD_PARAMS), list(GOOD_MEMBERS)


def violated(coords, members):
    return {k for k, v in sc.constraint_violations(coords, members).items() if v > 0}


def test_good_design_is_legal():
    coords, members = good()
    ev = sc.evaluate(coords, members)
    assert ev.feasible and ev.valid, ev.violations
    assert ev.p_max > 0 and ev.cost > 0
    assert np.isclose(ev.p_over_cost, ev.p_max / ev.cost)
    assert g.count_crossings(coords, members) == 0


@pytest.mark.parametrize("span, ok", [(29.9, False), (30.0, True), (33.0, True), (33.1, False)])
def test_span_range(span, ok):
    coords, members = good()
    coords[config.ROLLER_INDEX, 0] = span
    assert ("span" not in violated(coords, members)) == ok


@pytest.mark.parametrize("h, ok", [(0.4, False), (0.5, True), (1.5, True), (1.6, False)])
def test_load_height_range(h, ok):
    coords, members = good()
    coords[config.LOAD_INDEX, 1] = h
    assert ("load_height" not in violated(coords, members)) == ok


def test_member_length_limits():
    coords, members = good()
    assert "member_length" not in violated(coords, members)
    # too short: joint 3 moves to within ~5 in of the load joint (member 3-2)
    short = coords.copy()
    short[3] = (15.0, 0.0)
    assert "member_length" in violated(short, members)
    # too long: joint 4 moves up so that member 0-4 exceeds 14 in
    long = coords.copy()
    long[4] = (5.0, 14.0)
    assert "member_length" in violated(long, members)
    # violation is graded by how far outside the limit
    v1 = sc.constraint_violations(long, members)["member_length"]
    long[4] = (5.0, 20.0)
    assert sc.constraint_violations(long, members)["member_length"] > v1 > 0


def test_material_budget():
    # Stock rule: the members must be cut from sticks of 48, 48 and 24 in (one stick per member).
    coords, members = good()
    assert sc.constraint_violations(coords, members)["material"] == 0
    # Top chord at 8.5 in: 107.6 in of members, packable as 4 + 5 + 2 members.
    ok = coords.copy()
    ok[4:, 1] = 8.5
    assert violated(ok, members) == set()
    # Top chord at 9 in: only 110.2 in of members, well under 120 in, but they cannot be cut from
    # the sticks (the best split overflows by ~0.87 in). Every member is still in 8.5-14 in.
    stock_bound = coords.copy()
    stock_bound[4:, 1] = 9.0
    lengths = g.member_lengths(stock_bound, members)
    assert lengths.sum() < config.MATERIAL_BUDGET
    assert violated(stock_bound, members) == {"material"}
    assert np.isclose(sc.constraint_violations(stock_bound, members)["material"],
                      brute_force_overflow(lengths.tolist()), atol=1e-9)
    # Top chord at 11 in: the total length itself (~120.79 in) exceeds all the stock.
    tall = coords.copy()
    tall[4:, 1] = 11.0
    assert violated(tall, members) == {"material"}
    assert np.isclose(sc.constraint_violations(tall, members)["material"],
                      g.member_lengths(tall, members).sum() - config.MATERIAL_BUDGET)


def test_nothing_below_supports():
    coords, members = good()
    coords[3, 1] = -0.2
    assert "below_supports" in violated(coords, members)
    coords[3, 1] = 0.0
    assert "below_supports" not in violated(coords, members)


def test_supports_on_same_level():
    coords, members = good()
    coords[config.ROLLER_INDEX, 1] = 0.3
    assert "supports" in violated(coords, members)


def test_strip_rule_joint_only():
    coords, members = good()
    xl = coords[config.LOAD_INDEX, 0]
    coords[3] = (xl + 0.2, 0.0)                  # joint inside the strip, below the load joint
    assert "strip" in violated(coords, members)
    coords[3] = (xl + 0.6, 0.0)                  # outside the strip
    assert "strip" not in violated(coords, members)
    # joint above the load joint at the same x is fine
    coords2, members2 = good()
    coords2[5] = (coords2[config.LOAD_INDEX, 0], 8.0)
    assert "strip" not in violated(coords2, members2)


def test_member_under_load_joint_is_fine():
    coords = np.array([[0, 0], [30, 0], [15, 1.0]], dtype=float)
    assert "strip" not in violated(coords, [(0, 2), (2, 1), (0, 1)])


def test_crossing_is_detected():
    coords, members = good()
    assert sc.constraint_violations(coords, members)["crossings"] == 0
    members = members + [(3, 6)]                 # 3-6 cuts across diagonals 3-5/5-2 and 2-6
    assert sc.constraint_violations(coords, members)["crossings"] >= config.CROSSING_VIOLATION


def test_violations_are_graded():
    coords, members = good()
    coords[config.ROLLER_INDEX, 0] = 34.0
    small = sc.constraint_violations(coords, members)["span"]
    coords[config.ROLLER_INDEX, 0] = 36.0
    big = sc.constraint_violations(coords, members)["span"]
    assert 0 < small < big


def test_score_ordering():
    coords, members = good()
    legal = sc.score_design(coords, members, sc.OBJ_MAX_LOAD)
    assert legal < 0
    bad = coords.copy()
    bad[config.ROLLER_INDEX, 0] = 36.0
    infeasible = sc.score_design(bad, members, sc.OBJ_MAX_LOAD)
    assert infeasible >= config.PENALTY_INFEASIBLE
    # the more infeasible design scores worse
    worse = coords.copy()
    worse[config.ROLLER_INDEX, 0] = 40.0
    assert sc.score_design(worse, members, sc.OBJ_MAX_LOAD) > infeasible
    # singular designs sit above merely infeasible ones
    flat = np.array([[0, 0], [30, 0], [15, 0]], dtype=float)
    singular = sc.score_design(flat, [(0, 2), (2, 1), (0, 1)], sc.OBJ_MAX_LOAD)
    assert singular >= config.PENALTY_SINGULAR > infeasible


def test_objectives_differ_and_log_separately():
    coords, members = good()
    ev = sc.evaluate(coords, members)
    s1 = sc.score(ev, sc.OBJ_MAX_LOAD)
    s2 = sc.score(ev, sc.OBJ_MAX_LOAD_PER_COST)
    assert np.isclose(s1, -(ev.p_max - config.COST_TIEBREAK * ev.cost))
    assert np.isclose(s2, -ev.p_over_cost)
    assert ev.p_max > 0 and ev.cost > 0              # stored separately on the Evaluation


def test_cost_tiebreak_prefers_cheaper_for_equal_load():
    coords, members = good()
    ev = sc.evaluate(coords, members)
    cheaper = sc.Evaluation(violations=ev.violations, violation=0.0, feasible=True,
                            analysis=ev.analysis, p_max=ev.p_max, cost=ev.cost - 5.0,
                            p_over_cost=ev.p_max / (ev.cost - 5.0))
    assert sc.score(cheaper, sc.OBJ_MAX_LOAD) < sc.score(ev, sc.OBJ_MAX_LOAD)


def test_unknown_objective():
    coords, members = good()
    with pytest.raises(ValueError):
        sc.score_design(coords, members, "nope")
