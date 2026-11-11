import math

import config


def test_joint_and_member_limits():
    # Sticks 48 + 48 + 24: floor(48/8.5) + floor(48/8.5) + floor(24/8.5) = 5 + 5 + 2 = 12 members
    # at most. M = 2J - 3 is odd, so M <= 11 and J <= 7 (13-member trusses cannot be cut).
    assert config.STOCK_MEMBER_LIMIT == 12
    assert config.J_MAX == 7
    assert config.M_MAX == 11
    # ceil(30 / 14) = 3 hops between pin and roller -> at least 4 joints
    assert config.J_MIN == 4


def test_members_for_joints():
    assert config.members_for_joints(config.J_MIN) == 5
    assert config.members_for_joints(config.J_MAX) == config.M_MAX


def test_stock_sticks_and_budget():
    assert config.STOCK_LENGTHS == (48.0, 48.0, 24.0)
    assert config.MATERIAL_BUDGET == sum(config.STOCK_LENGTHS) == 120.0


def test_max_design_fits_the_sticks_but_one_more_member_does_not():
    from src import packing
    assert packing.is_packable([config.MEMBER_LENGTH_MIN] * config.M_MAX)
    assert packing.is_packable([config.MEMBER_LENGTH_MIN] * config.STOCK_MEMBER_LIMIT)
    assert not packing.is_packable([config.MEMBER_LENGTH_MIN] * (config.STOCK_MEMBER_LIMIT + 1))
    # the plain length total would have allowed 14 members of 8.5 in
    assert math.floor(config.MATERIAL_BUDGET / config.MEMBER_LENGTH_MIN) == 14


def test_joint_indices_distinct():
    idx = {config.PIN_INDEX, config.ROLLER_INDEX, config.LOAD_INDEX}
    assert len(idx) == 3 and config.FIRST_FREE_INDEX == 3
