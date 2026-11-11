# Screening stage: construction orders, random geometry, repair, seeds, ranking and determinism.

import random

import networkx as nx
import numpy as np
import pytest

import config
from src import geometry as g
from src import optimiser as op
from src import scoring as sc
from src import topology as tp
from tests.test_constraints import GOOD_MEMBERS, GOOD_PARAMS


@pytest.fixture(scope="module")
def cands6_7():
    return tp.enumerate_candidates(6, 7)


def warren_candidate():
    """The J = 7 candidate isomorphic (with roles) to the Warren truss used in the other tests."""
    warren = tp.candidate_graph(tp.Candidate(7, tuple(sorted(tuple(sorted(m))
                                                             for m in GOOD_MEMBERS)), 0))
    for c in tp.candidates_for_joints(7):
        if nx.is_isomorphic(tp.candidate_graph(c), warren,
                            node_match=lambda a, b: a["role"] == b["role"]):
            return c
    raise AssertionError("Warren candidate not found")


# --------------------------------------------------------------------------------------
# Construction orders
# --------------------------------------------------------------------------------------
def test_construction_orders_rebuild_the_graph(cands6_7):
    for c in cands6_7[::7]:
        rng = random.Random(1)
        orders = op.construction_orders(c, rng)
        assert orders
        for base, steps in orders:
            placed = set(base)
            edges = {tuple(sorted(base))}
            for v, u, w in steps:
                assert u in placed and w in placed and v not in placed
                placed.add(v)
                edges |= {tuple(sorted((v, u))), tuple(sorted((v, w)))}
            assert placed == set(range(c.n_joints))
            assert edges == set(c.members)


# --------------------------------------------------------------------------------------
# Random geometry
# --------------------------------------------------------------------------------------
def test_sample_geometry_is_fitted_to_supports(cands6_7):
    rng = random.Random(3)
    got = 0
    for c in cands6_7[:40]:
        orders = op.construction_orders(c, rng)
        for _ in range(20):
            co = op.sample_geometry(c, orders, rng)
            if co is None:
                continue
            got += 1
            assert co.shape == (c.n_joints, 2)
            assert np.allclose(co[config.PIN_INDEX], (0.0, 0.0))
            assert config.SPAN_MIN <= co[config.ROLLER_INDEX, 0] <= config.SPAN_MAX
            assert co[config.ROLLER_INDEX, 1] == 0.0
            assert co[:, 1].sum() >= 0.0                      # flipped to lie mostly above
    assert got > 50


def test_blocked_detects_crossing_and_foldback():
    pos = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (5.0, 8.0), 3: (5.0, -8.0)}
    edges = [(0, 1), (0, 2), (1, 2)]
    # a new joint at (5, -1) joined to 2 and 3 would cut through member 0-1
    assert op._blocked((5.0, -1.0), 2, 3, pos, edges)
    # a new joint above, joined to 0 and 1 is fine
    assert not op._blocked((5.0, 8.5), 0, 2, {**pos}, [(0, 1)])
    # folding back onto an existing member from the shared joint
    assert op._blocked((20.0, 0.0), 0, 1, pos, [(0, 1)])


# --------------------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------------------
def test_repair_fixes_smooth_rules_and_keeps_supports(cands6_7):
    c = warren_candidate()
    base = seed = op.seed_geometries(c)[0]
    rng = np.random.default_rng(0)
    batch = np.repeat(base[None], 16, axis=0)
    noise = rng.normal(0, 2.0, batch.shape)
    noise[:, [config.PIN_INDEX, config.ROLLER_INDEX]] = 0.0
    batch = batch + noise
    fixed = op.repair_batch(c, batch)
    assert np.allclose(fixed[:, config.PIN_INDEX], 0.0)
    assert np.allclose(fixed[:, config.ROLLER_INDEX], batch[:, config.ROLLER_INDEX])

    def smooth(co):
        v = sc.constraint_violations(co, c.members)
        return v["member_length"] + v["below_supports"] + v["load_height"] + v["material"]

    before = np.mean([smooth(b) for b in batch])
    after = np.mean([smooth(f) for f in fixed])
    assert after < 0.25 * before
    assert np.all(np.isfinite(fixed))


def test_repair_is_stable_for_wild_starts():
    c = warren_candidate()
    rng = np.random.default_rng(1)
    batch = rng.uniform(-200, 200, size=(32, c.n_joints, 2))
    batch[:, config.PIN_INDEX] = (0.0, 0.0)
    batch[:, config.ROLLER_INDEX] = (30.0, 0.0)
    assert np.all(np.isfinite(op.repair_batch(c, batch)))


# --------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------
def test_seed_layouts_are_legal():
    for seed in op.seed_layouts():
        ev = sc.evaluate(seed["coords"], seed["members"])
        assert ev.valid, (seed["name"], ev.violations)


def test_all_three_seeds_match_some_candidate(cands6_7):
    found = {tuple(map(tuple, np.round(co, 6))) for c in cands6_7 for co in op.seed_geometries(c)}
    assert len(found) >= 3


def test_seed_matches_when_pin_and_roller_labels_are_swapped():
    c = warren_candidate()
    swap = {0: 1, 1: 0}
    swapped = tp.Candidate(7, tuple(sorted(tuple(sorted((swap.get(a, a), swap.get(b, b))))
                                           for a, b in c.members)), 0)
    a = op.seed_geometries(c)[0]
    b = op.seed_geometries(swapped)[0]
    for co, cand in ((a, c), (b, swapped)):
        assert np.allclose(co[config.PIN_INDEX], (0, 0))
        assert sc.evaluate(co, cand.members).valid
    la = sorted(g.member_lengths(a, c.members))
    lb = sorted(g.member_lengths(b, swapped.members))
    assert np.allclose(la, lb)


# --------------------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------------------
def test_screen_candidate_is_deterministic():
    c = warren_candidate()
    r1, r2 = op.screen_candidate(c, 40), op.screen_candidate(c, 40)
    assert (r1.n_samples, r1.n_feasible) == (r2.n_samples, r2.n_feasible)
    for obj in sc.OBJECTIVES:
        assert r1.best[obj][0] == r2.best[obj][0]
        assert np.allclose(r1.best[obj][1], r2.best[obj][1])


def test_screen_results_are_legal_and_consistent():
    c = warren_candidate()
    res = op.screen_candidate(c, 40)
    assert res.n_feasible >= 1                       # at least the Warren seed
    for obj in sc.OBJECTIVES:
        score, co, ev = res.best[obj]
        again = sc.evaluate(co, c.members)
        assert again.valid
        assert np.isclose(sc.score(again, obj), score)
        assert score < 0


def test_screen_all_ranks_feasible_first_and_matches_across_workers(cands6_7):
    subset = cands6_7[:24]
    serial = op.screen_all(subset, n_samples=20, top_k=10, workers=1)
    for obj in sc.OBJECTIVES:
        keys = [s.rank_key(obj) for s in serial[obj]]
        assert keys == sorted(keys)
    assert len(serial["all"]) == 24
    parallel = op.screen_all(subset, n_samples=20, top_k=10, workers=2)
    for a, b in zip(serial["all"], parallel["all"]):
        assert (a.n_samples, a.n_feasible) == (b.n_samples, b.n_feasible)
        assert a.best_violation == b.best_violation


def test_rank_key_orders_feasible_before_infeasible():
    cand = tp.Candidate(3, ((0, 1),), 0)
    ok = op.Screen(cand, best={sc.OBJ_MAX_LOAD: (-5.0, None, None)})
    better = op.Screen(cand, best={sc.OBJ_MAX_LOAD: (-9.0, None, None)})
    near = op.Screen(cand, best_violation=0.1)
    far = op.Screen(cand, best_violation=3.0)
    ranked = sorted([far, ok, near, better], key=lambda s: s.rank_key(sc.OBJ_MAX_LOAD))
    assert ranked == [better, ok, near, far]


def test_seed_is_explicit_and_changes_the_samples():
    c = warren_candidate()
    a, b = op.screen_candidate(c, 40, seed=1), op.screen_candidate(c, 40, seed=1)
    other = op.screen_candidate(c, 40, seed=2)
    assert a.seed == 1 and other.seed == 2
    assert np.allclose(a.best[sc.OBJ_MAX_LOAD][1], b.best[sc.OBJ_MAX_LOAD][1])
    assert (a.n_samples, a.n_feasible) != (other.n_samples, other.n_feasible) or \
        not np.allclose(a.best[sc.OBJ_MAX_LOAD][1], other.best[sc.OBJ_MAX_LOAD][1])
    # default seed comes from config
    assert op.screen_candidate(c, 10).seed == config.RANDOM_SEED


def test_seed_reaches_worker_processes():
    subset = tp.enumerate_candidates(6, 7)[:12]
    serial = op.screen_all(subset, n_samples=20, top_k=5, workers=1, seed=7)
    parallel = op.screen_all(subset, n_samples=20, top_k=5, workers=2, seed=7)
    assert all(s.seed == 7 for s in parallel["all"])
    for a, b in zip(serial["all"], parallel["all"]):
        assert (a.n_samples, a.n_feasible, a.best_violation) == (b.n_samples, b.n_feasible,
                                                                  b.best_violation)
    default = op.screen_all(subset, n_samples=20, top_k=5, workers=1)
    assert any(a.best_violation != b.best_violation
               for a, b in zip(serial["all"], default["all"]))
