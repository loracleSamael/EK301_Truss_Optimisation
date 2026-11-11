import numpy as np
import pytest

import config
from src import analysis as an
from src import geometry as g


def test_buckling_capacity_matches_fit():
    assert np.isclose(an.buckling_capacity(10.0), 3391.349 * 10.0 ** -2.012)
    caps = an.buckling_capacity([8.5, 14.0])
    assert caps[0] > caps[1]


def test_cost():
    assert an.total_cost(4, 50.0) == 4 * 10 + 50


def test_triangle_hand_calculation():
    # pin (0,0), roller (8,0), load joint at apex (4,3): diagonals of length 5.
    coords = np.array([[0, 0], [8, 0], [4, 3]], dtype=float)
    members = [(0, 2), (2, 1), (0, 1)]
    r = an.analyse(coords, members)
    assert r.valid
    assert np.allclose(r.reactions, [0.0, 0.5, 0.5], atol=1e-12)
    assert np.allclose(r.forces, [-5 / 6, -5 / 6, 2 / 3], atol=1e-12)
    cap = an.buckling_capacity(5.0)
    assert np.isclose(r.p_max, cap / (5 / 6))
    assert r.governing in (0, 1)
    assert np.isclose(r.utilisation[0], 1.0) and r.utilisation[2] == 0.0


def test_hanging_load_truss_hand_calculation():
    # Load joint (index 2) sits on the bottom chord, apex above it (joint 3).
    coords = np.array([[0, 0], [6, 0], [3, 0.0], [3, 4]], dtype=float)
    members = [(0, 2), (2, 1), (0, 3), (3, 1), (2, 3)]
    r = an.analyse(coords, members)
    assert r.valid
    # F(2-3) = +1 (tension), diagonals -0.625, chords +0.375, reactions 0.5 each
    assert np.allclose(r.forces, [0.375, 0.375, -0.625, -0.625, 1.0], atol=1e-12)
    assert np.allclose(r.reactions, [0.0, 0.5, 0.5], atol=1e-12)
    assert np.isclose(r.p_max, an.buckling_capacity(5.0) / 0.625)
    assert np.isclose(r.total_length, 3 + 3 + 5 + 5 + 4)
    assert np.isclose(r.cost, 4 * 10 + 20)
    assert np.isclose(r.p_over_cost, r.p_max / r.cost)


def test_p_max_scales_linearly_with_test_load(monkeypatch):
    coords = np.array([[0, 0], [8, 0], [4, 3]], dtype=float)
    members = [(0, 2), (2, 1), (0, 1)]
    base = an.analyse(coords, members)
    monkeypatch.setattr(config, "TEST_LOAD", 7.0)
    scaled = an.analyse(coords, members)
    assert np.allclose(scaled.forces, 7 * base.forces)
    # P_max (absolute failure load) must not depend on the unit load chosen
    assert np.isclose(scaled.p_max, base.p_max)
    assert np.allclose(scaled.utilisation, base.utilisation)


def test_equilibrium_residual_is_zero():
    coords = g.params_to_coords([30.0, 14.0, 1.0, 8.0, 9.0, 22.0, 9.0])
    members = [(0, 3), (3, 2), (2, 4), (4, 1), (0, 2), (2, 1), (3, 4)]
    r = an.analyse(coords, members)
    assert r.valid
    A = an.equilibrium_matrix(coords, members)
    b = np.zeros(2 * len(coords))
    b[2 * config.LOAD_INDEX + 1] = config.TEST_LOAD
    x = np.concatenate([r.forces, r.reactions])
    assert np.allclose(A @ x, b, atol=1e-10)
    # vertical load only -> no horizontal reaction, vertical reactions sum to the load
    assert abs(r.reactions[0]) < 1e-10
    assert np.isclose(r.reactions[1] + r.reactions[2], config.TEST_LOAD)


def test_mirror_gives_same_p_max():
    coords = g.params_to_coords([30.0, 12.0, 1.0, 8.0, 9.0, 22.0, 9.0])
    members = [(0, 3), (3, 2), (2, 4), (4, 1), (0, 2), (2, 1), (3, 4)]
    a = an.analyse(coords, members)
    mc, mm = g.mirror_design(coords, members)
    b = an.analyse(mc, mm)
    assert np.isclose(a.p_max, b.p_max)
    assert np.isclose(a.cost, b.cost)


def test_collinear_mechanism_is_singular():
    coords = np.array([[0, 0], [30, 0], [15, 0]], dtype=float)
    r = an.analyse(coords, [(0, 2), (2, 1), (0, 1)])
    assert not r.valid and "singular" in r.reason


def test_non_determinate_is_rejected():
    coords = np.array([[0, 0], [8, 0], [4, 3]], dtype=float)
    r = an.analyse(coords, [(0, 2), (2, 1)])
    assert not r.valid and "determinate" in r.reason


def test_zero_length_member_is_rejected():
    coords = np.array([[0, 0], [8, 0], [4, 3], [4, 3]], dtype=float)
    r = an.analyse(coords, [(0, 2), (2, 1), (0, 1), (2, 3), (3, 1)])
    assert not r.valid


# --------------------------------------------------------------------------------------
# Buckling-fit uncertainty (+/- config.BUCKLING_UNCERTAINTY oz on each capacity)
# --------------------------------------------------------------------------------------
def test_uncertainty_triangle_hand_calculation():
    coords = np.array([[0, 0], [8, 0], [4, 3]], dtype=float)
    r = an.analyse(coords, [(0, 2), (2, 1), (0, 1)])
    cap = an.buckling_capacity(5.0)
    d = config.BUCKLING_UNCERTAINTY
    # both 5 in diagonals carry 5/6 oz of compression per oz of load
    assert np.isclose(r.p_max, cap / (5 / 6))
    assert np.isclose(r.p_max_uncertainty, d / (5 / 6))
    assert np.isclose(r.p_max_lower, (cap - d) / (5 / 6))
    assert np.isclose(r.p_max_lower, r.p_max - r.p_max_uncertainty)      # same member governs


def test_uncertainty_hanging_load_truss_hand_calculation():
    coords = np.array([[0, 0], [6, 0], [3, 0.0], [3, 4]], dtype=float)
    r = an.analyse(coords, [(0, 2), (2, 1), (0, 3), (3, 1), (2, 3)])
    cap, d = an.buckling_capacity(5.0), config.BUCKLING_UNCERTAINTY
    assert np.isclose(r.p_max_uncertainty, d / 0.625)
    assert np.isclose(r.p_max_lower, (cap - d) / 0.625)


def test_lower_bound_never_exceeds_nominal_minus_governing_uncertainty():
    coords = g.params_to_coords([30.0, 14.0, 1.0, 8.0, 9.0, 22.0, 9.0])
    members = [(0, 3), (3, 2), (2, 4), (4, 1), (0, 2), (2, 1), (3, 4)]
    r = an.analyse(coords, members)
    assert r.valid and 0 < r.p_max_lower <= r.p_max - r.p_max_uncertainty + 1e-9
    assert r.p_max_uncertainty > 0


def test_uncertainty_is_independent_of_the_unit_load(monkeypatch):
    coords = np.array([[0, 0], [8, 0], [4, 3]], dtype=float)
    members = [(0, 2), (2, 1), (0, 1)]
    base = an.analyse(coords, members)
    monkeypatch.setattr(config, "TEST_LOAD", 7.0)
    scaled = an.analyse(coords, members)
    assert np.isclose(scaled.p_max_lower, base.p_max_lower)
    assert np.isclose(scaled.p_max_uncertainty, base.p_max_uncertainty)


def test_invalid_designs_have_no_uncertainty_numbers():
    r = an.analyse(np.array([[0, 0], [30, 0], [15, 0]], dtype=float), [(0, 2), (2, 1), (0, 1)])
    assert not r.valid and r.p_max_lower == 0.0 and r.p_max_uncertainty == 0.0
