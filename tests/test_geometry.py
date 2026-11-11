import numpy as np
import pytest

import config
from src import geometry as g


@pytest.fixture
def warren():
    # pin, roller, load joint, one free joint on top
    params = [30.0, 15.0, 1.0, 7.0, 6.0]
    coords = g.params_to_coords(params)
    members = [(0, 3), (3, 2), (2, 1), (0, 2), (3, 1)]
    return params, coords, members


def test_params_round_trip(warren):
    params, coords, _ = warren
    assert coords.shape == (4, 2)
    assert np.allclose(coords[0], (0, 0)) and np.allclose(coords[1], (30, 0))
    assert np.allclose(coords[2], (15, 1)) and np.allclose(coords[3], (7, 6))
    assert np.allclose(g.coords_to_params(coords), params)
    assert g.span(coords) == 30.0


def test_params_bad_length():
    with pytest.raises(ValueError):
        g.params_to_coords([30.0, 15.0, 1.0, 7.0])


def test_lengths_and_angles():
    coords = np.array([[0, 0], [30, 0], [3, 4]], dtype=float)
    members = [(0, 2), (0, 1), (1, 2)]
    assert np.allclose(g.member_lengths(coords, members), [5.0, 30.0, np.hypot(27, 4)])
    ang = g.member_angles(coords, members)
    assert np.isclose(ang[1], 0.0)
    assert np.isclose(ang[0], np.degrees(np.arctan2(4, 3)))
    # direction of the member does not matter
    assert np.allclose(g.member_angles(coords, [(2, 0)]), ang[0])


def test_segments_cross():
    assert g.segments_cross((0, 0), (2, 2), (0, 2), (2, 0))          # X
    assert not g.segments_cross((0, 0), (1, 0), (0, 1), (1, 1))      # parallel
    assert not g.segments_cross((0, 0), (1, 1), (2, 2), (3, 3))      # collinear, disjoint
    assert g.segments_cross((0, 0), (2, 2), (1, 1), (3, 3))          # collinear overlap
    assert g.segments_cross((0, 0), (2, 0), (1, 0), (1, 1))          # T-touch


def test_shared_endpoint_is_not_a_crossing():
    coords = np.array([[0, 0], [10, 0], [5, 5]], dtype=float)
    assert g.count_crossings(coords, [(0, 1), (1, 2), (2, 0)]) == 0


def test_folded_back_members_are_flagged():
    coords = np.array([[0, 0], [5, 0], [10, 0]], dtype=float)
    assert g.count_crossings(coords, [(0, 1), (0, 2)]) == 1


def test_crossing_count_planar_and_not(warren):
    _, coords, members = warren
    assert g.count_crossings(coords, members) == 0
    # (0,2) and (3,1) cross in this layout
    crossed = np.array([[0, 0], [30, 0], [20, 5], [10, 5]], dtype=float)
    assert g.count_crossings(crossed, [(0, 2), (3, 1)]) == 1


def test_duplicate_member_counts_as_crossing():
    coords = np.array([[0, 0], [10, 0], [5, 5]], dtype=float)
    assert g.count_crossings(coords, [(0, 1), (1, 0)]) == 1


def test_strip_joint_below_load():
    coords = np.array([[0, 0], [30, 0], [15, 1.0], [15.3, 0.2]], dtype=float)
    assert g.joints_in_strip(coords) == [3]
    assert g.joints_in_strip(coords, tol=0.1) == []


def test_strip_outside_or_above_is_fine():
    coords = np.array([[0, 0], [30, 0], [15, 1.0], [16.0, 0.2], [15.2, 5.0]], dtype=float)
    assert g.joints_in_strip(coords) == []   # one outside strip, one above the load joint


def test_member_under_load_joint_is_allowed():
    # chord from pin to roller passes directly under the load joint; no joint there
    coords = np.array([[0, 0], [30, 0], [15, 1.0]], dtype=float)
    assert g.joints_in_strip(coords) == []


def test_mirror_is_involution(warren):
    _, coords, members = warren
    mc, mm = g.mirror_design(coords, members)
    assert np.allclose(mc[0], (0, 0)) and np.allclose(mc[1], (30, 0))
    mmc, mmm = g.mirror_design(mc, mm)
    assert np.allclose(mmc, coords)
    assert sorted(tuple(sorted(m)) for m in mmm) == sorted(tuple(sorted(m)) for m in members)


def test_mirror_preserves_lengths(warren):
    _, coords, members = warren
    mc, mm = g.mirror_design(coords, members)
    assert np.allclose(sorted(g.member_lengths(coords, members)),
                       sorted(g.member_lengths(mc, mm)))


def test_canonical_key_matches_mirror(warren):
    _, coords, members = warren
    mc, mm = g.mirror_design(coords, members)
    assert g.canonical_key(coords, members) == g.canonical_key(mc, mm)
    other = coords.copy()
    other[3, 1] += 1.0
    assert g.canonical_key(other, members) != g.canonical_key(coords, members)
