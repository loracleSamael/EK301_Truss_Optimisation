# Stock-length packing: 48 + 48 + 24 in sticks, every member from a single stick.

import itertools
import time

import numpy as np
import pytest

import config
from src import packing as pk

CAPS = config.STOCK_LENGTHS


def brute_force_overflow(lengths):
    """Independent reference: plain Python over every assignment, no symmetry tricks."""
    best = float("inf")
    for combo in itertools.product(range(len(CAPS)), repeat=len(lengths)):
        loads = [0.0] * len(CAPS)
        for member, stick in enumerate(combo):
            loads[stick] += lengths[member]
        best = min(best, sum(max(0.0, loads[s] - CAPS[s]) for s in range(len(CAPS))))
    return best


def test_stock_configuration():
    assert CAPS == (48.0, 48.0, 24.0)
    assert config.MATERIAL_BUDGET == 120.0
    assert config.STOCK_MEMBER_LIMIT == 12          # 5 + 5 + 2 members of at least 8.5 in
    assert config.J_MAX == 7 and config.M_MAX == 11


def test_single_member_fits_the_small_stick():
    r = pk.min_overflow([24.0])
    assert r.packable and r.overflow == 0.0


def test_member_longer_than_every_stick_overflows():
    r = pk.min_overflow([50.0])
    assert not r.packable and np.isclose(r.overflow, 2.0)       # best case: a 48 in stick


def test_twelve_minimum_members_fit_as_five_five_two():
    r = pk.min_overflow([8.5] * 12)
    assert r.packable
    assert sorted(np.bincount(r.assignment, minlength=3).tolist()) == [2, 5, 5]


def test_thirteen_minimum_members_are_infeasible():
    # 13 x 8.5 = 110.5 in is under 120 in, but the sticks only hold 5 + 5 + 2 members.
    r = pk.min_overflow([8.5] * 13)
    # best case: 5 + 5 members on the 48 in sticks (42.5 in each) and 3 on the 24 in stick (25.5 in)
    assert not r.packable and np.isclose(r.overflow, 1.5)
    assert np.isclose(brute_force_overflow([8.5] * 9 + [8.5] * 3), 0.0)    # 12 members are fine


def test_sixth_member_in_a_48_inch_stick_does_not_fit():
    # six members of 8.5 in need 51 in; with only two 48 in sticks and a 24 in stick the
    # total of 13 members is what overflows, not 12.
    assert pk.is_packable([8.5] * 12)
    assert not pk.is_packable([8.5] * 12 + [8.5])


def test_total_under_120_can_still_be_unpackable():
    # 11 in + 11 in + 11 in + 11 in + 11 in cannot share a 48 in stick (55 in) and the 24 in stick
    # only holds two; 4+... members of 13 in: total is 117 in but 9 x 13 = 117 splits 3/3/2 -> 39,39,26
    lengths = [13.0] * 9
    assert sum(lengths) < 120
    r = pk.min_overflow(lengths)
    assert not r.packable                      # 24 in stick: at most one 13 in member (26 > 24)
    assert r.overflow > 0


def test_total_over_120_returns_the_excess_cheaply():
    lengths = [14.0] * 10                       # 140 in
    assert np.isclose(pk.material_violation(lengths), 20.0)


def test_material_violation_zero_when_packable():
    assert pk.material_violation([9.0] * 11) == 0.0             # 5 + 5 + 1 members: 45, 45, 9 in
    assert pk.material_violation([]) == 0.0


def test_eleven_members_of_ten_inches_do_not_pack():
    # 110 in is under 120 in, but a 48 in stick holds only four 10 in members (five need 50 in)
    # and the 24 in stick two, so only 10 of the 11 members fit.
    assert np.isclose(pk.material_violation([10.0] * 11), 2.0)


@pytest.mark.parametrize("seed", range(8))
def test_matches_brute_force_on_random_small_cases(seed):
    rng = np.random.default_rng(seed)
    lengths = rng.uniform(8.5, 14.0, size=int(rng.integers(3, 8)))
    r = pk.min_overflow(lengths)
    assert np.isclose(r.overflow, brute_force_overflow(lengths.tolist()), atol=1e-9)
    # the returned assignment really has that overflow
    loads = np.array([lengths[r.assignment == s].sum() for s in range(3)])
    assert np.allclose(loads, r.loads)
    assert np.isclose(np.maximum(loads - np.array(CAPS), 0).sum(), r.overflow, atol=1e-9)


def test_matches_brute_force_with_overflowing_cases():
    rng = np.random.default_rng(99)
    for _ in range(5):
        lengths = rng.uniform(12.0, 20.0, size=6)               # forced to overflow
        assert np.isclose(pk.min_overflow(lengths).overflow,
                          brute_force_overflow(lengths.tolist()), atol=1e-9)


def test_order_of_members_does_not_matter():
    rng = np.random.default_rng(5)
    lengths = rng.uniform(8.5, 12.0, size=9)
    a = pk.min_overflow(lengths).overflow
    b = pk.min_overflow(lengths[::-1]).overflow
    c = pk.min_overflow(rng.permutation(lengths)).overflow
    assert np.isclose(a, b) and np.isclose(a, c)


def test_deterministic():
    lengths = np.linspace(8.6, 13.0, 11)
    a, b = pk.min_overflow(lengths), pk.min_overflow(lengths)
    assert np.array_equal(a.assignment, b.assignment) and a.overflow == b.overflow


def test_top_assignments_are_sorted_and_distinct():
    lengths = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 9.2, 9.7, 10.2, 10.7])
    tops = pk.top_assignments(lengths, 4)
    assert len(tops) == 4
    keys = [(t.overflow, (t.loads - np.array(CAPS)).max()) for t in tops]
    assert keys == sorted(keys)
    assert len({tuple(t.assignment) for t in tops}) == 4
    assert tops[0].overflow == pk.min_overflow(lengths).overflow


def test_exact_search_marks_two_48_sticks_as_interchangeable():
    digits, masks = pk._table(6, tuple(map(float, CAPS)))
    assert len(digits) < 3 ** 6                  # symmetry removed some assignments
    assert len({tuple(r) for r in digits.tolist()}) == len(digits)


def test_cutting_plan_accounts_for_every_member():
    lengths = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 9.2, 9.7, 10.2, 10.7])
    plan = pk.cutting_plan(lengths)
    assert [p["capacity"] for p in plan] == [48.0, 48.0, 24.0]
    members = sorted(m for p in plan for m in p["members"])
    assert members == list(range(len(lengths)))
    assert np.isclose(sum(p["used"] for p in plan), lengths.sum())
    for p in plan:
        assert np.isclose(p["left"], p["capacity"] - p["used"])
        assert p["left"] >= -1e-9                                # this case is packable
    assert plan[0]["stick"].startswith("A") and "24" in plan[2]["stick"]


def test_quick_path_never_disagrees_with_the_exact_search():
    rng = np.random.default_rng(11)
    for _ in range(60):
        lengths = rng.uniform(8.5, 14.0, size=int(rng.integers(5, 12)))
        exact = pk.min_overflow(lengths).overflow
        quick = pk.material_violation(lengths)
        if lengths.sum() <= config.MATERIAL_BUDGET:
            assert np.isclose(quick, exact if exact > pk.TOL else 0.0, atol=1e-9)
        else:
            assert quick > 0 and exact > 0                    # excess is a lower bound on overflow
            assert quick <= exact + 1e-9


def test_first_fit_fallback_above_exact_limit():
    r = pk.min_overflow([8.5] * 14)             # beyond MAX_EXACT_PACKING_MEMBERS
    assert not r.packable and len(r.assignment) == 14
    ok = pk.min_overflow([1.0] * 13)
    assert ok.packable


def test_exact_search_is_fast_enough():
    lengths = np.linspace(8.6, 11.0, 11)
    pk.min_overflow(lengths)                     # builds the table once
    t = time.perf_counter()
    for _ in range(20):
        pk.min_overflow(lengths)
    assert (time.perf_counter() - t) / 20 < 0.05          # well under 50 ms per call
