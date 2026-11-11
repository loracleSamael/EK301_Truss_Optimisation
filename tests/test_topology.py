# Topology enumeration: every graph is determinate, connected, planar and rigid, duplicates are
# removed, and known layouts (Warren) are present.

import itertools

import networkx as nx
import numpy as np
import pytest

import config
from src import analysis, geometry
from src import topology as tp
from tests.test_constraints import GOOD_MEMBERS


def test_small_graph_counts():
    # Hand enumeration: J=3 triangle; J=4 K4 minus an edge; J=5 gives exactly three graphs.
    assert [len(tp.enumerate_graphs(j)) for j in (2, 3, 4, 5)] == [1, 1, 1, 3]


@pytest.mark.parametrize("j", range(3, config.J_MAX + 1))
def test_graph_properties(j):
    for g in tp.enumerate_graphs(j):
        assert g.number_of_nodes() == j
        assert g.number_of_edges() == config.members_for_joints(j)
        assert nx.is_connected(g)
        assert min(d for _, d in g.degree) >= 2
        assert tp.is_planar(g)


@pytest.mark.parametrize("j", range(3, config.J_MAX))
def test_graphs_have_no_isomorphic_duplicates(j):
    gs = tp.enumerate_graphs(j)
    for a, b in itertools.combinations(gs, 2):
        assert not nx.is_isomorphic(a, b)


def test_graph_counts_are_stable():
    # Regression guard on the enumeration (computed from this implementation).
    assert [len(tp.enumerate_graphs(j)) for j in range(2, 9)] == [1, 1, 1, 3, 11, 59, 442]


def test_candidate_counts_are_stable():
    counts = tp.candidate_counts()
    assert counts[4] == (1, 0) and counts[5] == (3, 0)   # diameter 2: pin/roller < 3 hops apart
    assert counts[6] == (11, 4)
    assert counts[7] == (59, 143)
    # 13 members cannot be cut from the 48 + 48 + 24 in sticks, so J = 8 is outside the default range
    assert 8 not in counts and config.J_MAX == 7
    assert tp.candidate_counts(8, 8)[8] == (442, 3708)           # still enumerable on request
    assert not any(tp.fits_material_budget(c) for c in tp.candidates_for_joints(8)[:50])


def test_candidates_respect_joint_convention_and_hops():
    for c in tp.enumerate_candidates(6, 7):
        g = tp.candidate_graph(c)
        assert c.n_members == config.members_for_joints(c.n_joints)
        assert all(i < j for i, j in c.members)
        assert list(c.members) == sorted(c.members)
        hops = nx.shortest_path_length(g, config.PIN_INDEX, config.ROLLER_INDEX)
        assert hops >= config.MIN_HOPS_PIN_TO_ROLLER
        assert config.LOAD_INDEX != config.PIN_INDEX and g.has_node(config.LOAD_INDEX)
        assert g.nodes[config.LOAD_INDEX]["role"] == "load"


def test_no_duplicate_candidates():
    for j in (6, 7):
        cands = tp.candidates_for_joints(j)
        for a, b in itertools.combinations(cands, 2):
            if a.graph_id != b.graph_id:
                continue
            ga, gb = tp.candidate_graph(a), tp.candidate_graph(b)
            same = nx.is_isomorphic(ga, gb, node_match=lambda x, y: x["role"] == y["role"])
            assert not same, (a, b)


def test_candidates_are_rigid_for_generic_geometry():
    rng = np.random.default_rng(0)
    for c in tp.enumerate_candidates(6, 7):
        coords = rng.uniform(0, 30, size=(c.n_joints, 2))
        coords[config.PIN_INDEX] = (0.0, 0.0)
        coords[config.ROLLER_INDEX] = (30.0, 0.0)
        res = analysis.analyse(coords, c.members)
        assert res.valid, (c, res.reason)


def test_warren_truss_is_in_the_candidate_set():
    target = nx.Graph()
    target.add_nodes_from(range(7))
    target.add_edges_from(GOOD_MEMBERS)
    roles = {0: "end", 1: "end", 2: "load"}
    for n in target.nodes:
        target.nodes[n]["role"] = roles.get(n, "free")
    found = any(
        nx.is_isomorphic(target, tp.candidate_graph(c), node_match=lambda a, b: a["role"] == b["role"])
        for c in tp.candidates_for_joints(7)
    )
    assert found


def test_material_prefilter():
    assert all(tp.fits_material_budget(c) for c in tp.enumerate_candidates())
    too_many = tp.Candidate(9, tuple((i, i + 1) for i in range(15)), 0)   # 15 members * 8.5 > 120
    assert not tp.fits_material_budget(too_many)


def test_enumeration_rejects_bad_input():
    with pytest.raises(ValueError):
        tp.enumerate_graphs(1)
