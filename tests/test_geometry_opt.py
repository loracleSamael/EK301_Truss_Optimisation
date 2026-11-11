# Stage 2: SLSQP geometry optimiser, rounding, grid polish, multi-start, driver.

import math
import random

import numpy as np
import pytest

import config
from src import analysis
from src import geometry as g
from src import geometry_opt as go
from src import optimiser as op
from src import scoring as sc
from src import topology as tp
from tests.test_optimiser import warren_candidate


@pytest.fixture(scope="module")
def warren():
    return warren_candidate()


@pytest.fixture(scope="module")
def warren_screen(warren):
    return op.screen_candidate(warren, 40)


def independent_statics(coords, members):
    """Joint-by-joint statics with a separately built system; returns (forces, P_max)."""
    n_j, n_m = len(coords), len(members)
    A = np.zeros((2 * n_j, n_m + 3))
    for k, (i, j) in enumerate(members):
        u = (coords[j] - coords[i]) / math.dist(coords[i], coords[j])
        A[2 * i:2 * i + 2, k] += u
        A[2 * j:2 * j + 2, k] -= u
    A[0, n_m], A[1, n_m + 1], A[3, n_m + 2] = 1, 1, 1
    b = np.zeros(2 * n_j)
    b[2 * config.LOAD_INDEX + 1] = 1.0
    x = np.linalg.solve(A, b)
    forces = x[:n_m]
    caps = [config.BUCKLING_A * math.dist(coords[i], coords[j]) ** config.BUCKLING_B
            for i, j in members]
    p_max = min(caps[k] / -forces[k] for k in range(n_m) if forces[k] < -1e-9)
    return forces, p_max


# --------------------------------------------------------------------------------------
# Rounding and grid polish
# --------------------------------------------------------------------------------------
def test_round_coords_snaps_to_grid_and_keeps_supports():
    raw = np.array([[0.004, 0.003], [30.004, 0.004], [8.6149, 1.4051], [3.1234, 9.8765]])
    r = go.round_coords(raw)
    assert np.allclose(r[0], (0, 0)) and r[1, 1] == 0.0
    steps = r / config.COORD_ROUNDING
    assert np.allclose(steps, np.round(steps), atol=1e-6)
    assert np.allclose(r[2], (8.61, 1.41)) and np.allclose(r[3], (3.12, 9.88))


def test_grid_polish_never_worsens_and_stays_legal(warren):
    start = go.round_coords(op.seed_geometries(warren)[0])
    s0, ev0 = go._score(start, warren.members, sc.OBJ_MAX_LOAD)
    polished, s1, ev1 = go.grid_polish(warren, start, sc.OBJ_MAX_LOAD)
    assert s1 <= s0 and ev1.valid
    assert np.isclose(sc.score(sc.evaluate(polished, warren.members), sc.OBJ_MAX_LOAD), s1)


# --------------------------------------------------------------------------------------
# Single SLSQP run
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("objective", sc.OBJECTIVES)
def test_slsqp_improves_on_the_seed_and_yields_a_legal_design(warren, objective):
    seed = op.seed_geometries(warren)[0]
    base = sc.score(sc.evaluate(seed, warren.members), objective)
    coords, res = go.optimise_geometry(warren, seed, objective)
    design = go.finalise(warren, coords, objective)
    assert design is not None and design.evaluation.valid
    assert design.score < base - 1e-3
    assert design.objective == objective


def test_final_design_obeys_every_rule_with_margins(warren, warren_screen):
    design = go.optimise_candidate(warren_screen, sc.OBJ_MAX_LOAD, n_starts=4)
    assert design is not None
    ev = design.evaluation
    assert ev.feasible and all(v == 0 for v in ev.violations.values())
    lengths = g.member_lengths(design.coords, design.candidate.members)
    assert lengths.min() >= config.MEMBER_LENGTH_MIN and lengths.max() <= config.MEMBER_LENGTH_MAX
    assert lengths.sum() <= config.MATERIAL_BUDGET
    assert g.count_crossings(design.coords, design.candidate.members) == 0
    assert g.joints_in_strip(design.coords) == []
    # exported coordinates are on the grid
    steps = design.coords / config.COORD_ROUNDING
    assert np.allclose(steps, np.round(steps), atol=1e-6)


def test_result_matches_independent_statics(warren, warren_screen):
    design = go.optimise_candidate(warren_screen, sc.OBJ_MAX_LOAD_PER_COST, n_starts=4)
    forces, p_max = independent_statics(design.coords, design.candidate.members)
    assert np.isclose(p_max, design.p_max)
    assert np.allclose(forces, design.evaluation.analysis.forces)
    assert np.isclose(design.p_over_cost, design.p_max / design.cost)


def test_strip_side_is_respected():
    # A joint that starts left of the load joint and below it must not jump to the other side.
    cand = tp.Candidate(7, tuple(sorted(tuple(sorted(m)) for m in
                                        [(0, 3), (3, 2), (2, 1), (4, 5), (5, 6), (0, 4), (4, 3),
                                         (3, 5), (5, 2), (2, 6), (6, 1)])), 0)
    start = g.params_to_coords([30, 20, 1, 10, 0, 5, 8, 15, 8, 25, 8])
    problem = go._Problem(cand, sc.OBJ_MAX_LOAD, start)
    # The pin (x=0) and joint 3 (x=10) are left of the load at x=20, the roller (x=30) is right.
    # Supports count as joints for the strip rule. Joints above the load joint are unconstrained.
    assert problem.strip == [(0, -1.0), (1, 1.0), (3, -1.0)]
    assert (problem.cons(go._start_vector(start, problem)) >= -1e-9).all()


def test_singular_start_does_not_crash(warren):
    flat = op.seed_geometries(warren)[0].copy()
    flat[:, 1] = 0.0
    flat[config.LOAD_INDEX, 1] = 1.0
    coords, _ = go.optimise_geometry(warren, flat, sc.OBJ_MAX_LOAD, max_iter=20)
    assert np.all(np.isfinite(coords))


# --------------------------------------------------------------------------------------
# Multi-start and driver
# --------------------------------------------------------------------------------------
def test_start_points_are_deterministic_and_counted(warren_screen):
    a = go.start_points(warren_screen, sc.OBJ_MAX_LOAD, 5, random.Random(1))
    b = go.start_points(warren_screen, sc.OBJ_MAX_LOAD, 5, random.Random(1))
    assert len(a) == 5 and all(np.allclose(x, y) for x, y in zip(a, b))
    for s in a:
        assert np.allclose(s[config.PIN_INDEX], 0.0)


def test_start_points_fall_back_when_screening_found_nothing():
    c = tp.enumerate_candidates(7, 7)[0]
    empty = op.Screen(c)
    pts = go.start_points(empty, sc.OBJ_MAX_LOAD, 3, random.Random(2))
    assert 0 < len(pts) <= 3


def test_optimise_candidate_is_deterministic(warren_screen):
    a = go.optimise_candidate(warren_screen, sc.OBJ_MAX_LOAD, n_starts=3)
    b = go.optimise_candidate(warren_screen, sc.OBJ_MAX_LOAD, n_starts=3)
    assert a.score == b.score and np.allclose(a.coords, b.coords)


def test_more_starts_never_hurt(warren_screen):
    few = go.optimise_candidate(warren_screen, sc.OBJ_MAX_LOAD, n_starts=2)
    many = go.optimise_candidate(warren_screen, sc.OBJ_MAX_LOAD, n_starts=6)
    assert many.score <= few.score + 1e-9


def test_polish_design_never_worsens(warren_screen):
    d = go.optimise_candidate(warren_screen, sc.OBJ_MAX_LOAD, n_starts=2)
    assert go.polish_design(d).score <= d.score + 1e-12


def test_optimise_all_sorts_and_matches_across_workers(warren, warren_screen):
    other = op.screen_candidate(tp.enumerate_candidates(7, 7)[5], 40)
    screens = [warren_screen, other]
    serial = go.optimise_all(screens, sc.OBJ_MAX_LOAD, n_starts=2, n_polish=1, workers=1)
    parallel = go.optimise_all(screens, sc.OBJ_MAX_LOAD, n_starts=2, n_polish=1, workers=2)
    assert [d.score for d in serial] == sorted(d.score for d in serial)
    assert [round(d.score, 9) for d in serial] == [round(d.score, 9) for d in parallel]
    assert all(d.evaluation.valid for d in serial)


# --------------------------------------------------------------------------------------
# Stock sticks (48 + 48 + 24 in)
# --------------------------------------------------------------------------------------
def test_stock_constraints_are_linear_per_stick(warren):
    start = op.seed_geometries(warren)[0]
    assignment = go.packing.min_overflow(g.member_lengths(start, warren.members)).assignment
    problem = go._Problem(warren, sc.OBJ_MAX_LOAD, start, assignment)
    assert problem.stick_matrix.shape == (3, problem.n_members)
    assert np.all(problem.stick_matrix.sum(axis=0) == 1)                          # one stick each
    lengths = g.member_lengths(start, warren.members)
    expected = problem.stick_room - problem.stick_matrix @ lengths
    assert np.all(expected >= 0)                                                  # seed is cuttable
    # overload one stick and the constraint turns negative
    heavy = assignment.copy()
    heavy[:] = 2                                                                  # everything on 24 in
    bad = go._Problem(warren, sc.OBJ_MAX_LOAD, start, heavy)
    assert (bad.stick_room - bad.stick_matrix @ lengths).min() < 0


def test_slsqp_result_respects_the_assignment_before_rounding(warren):
    start = op.seed_geometries(warren)[0]
    assignment = go.packing.min_overflow(g.member_lengths(start, warren.members)).assignment
    coords, _ = go.optimise_geometry(warren, start, sc.OBJ_MAX_LOAD, assignment=assignment)
    lengths = g.member_lengths(coords, warren.members)
    for s, cap in enumerate(config.STOCK_LENGTHS):
        assert lengths[assignment == s].sum() <= cap - config.STOCK_MARGIN + 1e-6


def test_final_designs_can_be_cut_from_the_sticks(warren_screen):
    for obj in sc.OBJECTIVES:
        d = go.optimise_candidate(warren_screen, obj, n_starts=3)
        lengths = g.member_lengths(d.coords, d.candidate.members)
        r = go.packing.min_overflow(lengths)
        assert r.packable and d.evaluation.violations["material"] == 0
        plan = go.packing.cutting_plan(lengths, r.assignment)
        assert all(p["left"] >= -1e-9 for p in plan)
        assert sorted(m for p in plan for m in p["members"]) == list(range(len(lengths)))


def test_optimiser_adapts_when_the_stock_is_tight(monkeypatch, warren_screen):
    """Shrink the sticks so the stock rule really binds: the result must still be cuttable."""
    normal = go.optimise_candidate(warren_screen, sc.OBJ_MAX_LOAD, n_starts=3)
    monkeypatch.setattr(config, "STOCK_LENGTHS", (44.0, 44.0, 22.0))
    monkeypatch.setattr(config, "MATERIAL_BUDGET", 110.0)
    tight_screen = op.screen_candidate(warren_screen.candidate, 60)
    tight = go.optimise_candidate(tight_screen, sc.OBJ_MAX_LOAD, n_starts=3)
    assert tight is not None
    lengths = g.member_lengths(tight.coords, tight.candidate.members)
    assert go.packing.min_overflow(lengths).packable
    assert lengths.sum() <= 110.0
    assert tight.p_max < normal.p_max - 0.5            # less material, so a clearly lower capacity


def test_equilibrium_matrix_tolerates_zero_length_member():
    coords = np.array([[0, 0], [8, 0], [4, 3], [4, 3]], dtype=float)
    A = analysis.equilibrium_matrix(coords, [(0, 2), (2, 1), (0, 1), (2, 3), (3, 1)])
    assert np.all(np.isfinite(A))
