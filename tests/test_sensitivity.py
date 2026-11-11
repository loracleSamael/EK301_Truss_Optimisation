# Sensitivity of the finalists to the +/-1.52 oz buckling-fit uncertainty.

import numpy as np
import pytest

import config
from src import geometry_opt as go
from src import optimiser as op
from src import scoring as sc
from src import sensitivity as sens
from tests.test_optimiser import warren_candidate


@pytest.fixture(scope="module")
def screen():
    return op.screen_candidate(warren_candidate(), 60)


@pytest.fixture(scope="module")
def designs(screen):
    """A few different legal designs for the same candidate, best first (nominal order)."""
    out = []
    for n_starts in (1, 2, 4):
        d = go.optimise_candidate(screen, sc.OBJ_MAX_LOAD, n_starts=n_starts)
        if d is not None and not any(np.allclose(d.coords, o.coords) for o in out):
            out.append(d)
    other = go.optimise_candidate(screen, sc.OBJ_MAX_LOAD_PER_COST, n_starts=2)
    out.append(other)
    out.sort(key=lambda d: sc.score(d.evaluation, sc.OBJ_MAX_LOAD))
    return out


def test_lower_bound_value_matches_definition(designs):
    d = designs[0]
    a = d.evaluation.analysis
    assert sens.lower_bound_value(d, sc.OBJ_MAX_LOAD) == a.p_max_lower
    assert np.isclose(sens.lower_bound_value(d, sc.OBJ_MAX_LOAD_PER_COST), a.p_max_lower / d.cost)
    with pytest.raises(ValueError):
        sens.lower_bound_value(d, "nope")


def test_report_rows_and_ranks(designs):
    rep = sens.sensitivity_check(designs, sc.OBJ_MAX_LOAD, n=len(designs), with_regret=False)
    assert [r["nominal_rank"] for r in rep.rows] == list(range(1, len(designs) + 1))
    assert sorted(r["lower_rank"] for r in rep.rows) == list(range(1, len(designs) + 1))
    for r in rep.rows:
        assert 0 < r["drop"] < 0.2 and r["p_max_lower"] < r["p_max"]
    assert rep.winner_changed == (rep.rows[0]["lower_rank"] != 1)
    assert rep.regret is None


def test_drop_is_a_few_percent_for_realistic_designs(designs):
    rep = sens.sensitivity_check(designs[:1], sc.OBJ_MAX_LOAD, n=1, with_regret=False)
    # the lower bound removes 1.52 oz from capacities of roughly 18-46 oz: about 3-8 %
    assert 0.02 < rep.rows[0]["drop"] < 0.10


def test_regret_is_small_and_non_negative(designs):
    rep = sens.sensitivity_check(designs, sc.OBJ_MAX_LOAD, n=2, with_regret=True)
    assert rep.regret is not None and 0.0 <= rep.regret < 0.05


def test_offset_run_gives_a_legal_design_with_lower_bound_at_least_as_good(designs):
    d = designs[0]
    better, value = sens.reoptimise_against_lower_bound(d)
    assert better is not None and better.evaluation.valid
    # re-optimising against the lower bound cannot do much worse than the nominal design in
    # lower-bound terms (it starts there)
    assert value >= sens.lower_bound_value(d, d.objective) * 0.99


def test_summary_lines_mention_winner_status(designs):
    rep = sens.sensitivity_check(designs[:2], sc.OBJ_MAX_LOAD, n=2, with_regret=False)
    text = "\n".join(rep.summary_lines())
    assert "nominal #1" in text and ("Winner" in text or "WARNING" in text)


def test_empty_input():
    rep = sens.sensitivity_check([], sc.OBJ_MAX_LOAD)
    assert rep.rows == [] and not rep.winner_changed


def test_optimiser_offset_changes_the_problem(screen):
    """With the capacity offset the same start gives a lower failure load (sanity check)."""
    start = screen.best_coords(sc.OBJ_MAX_LOAD)
    plain, _ = go.optimise_geometry(screen.candidate, start, sc.OBJ_MAX_LOAD)
    offset, _ = go.optimise_geometry(screen.candidate, start, sc.OBJ_MAX_LOAD,
                                     offset=config.BUCKLING_UNCERTAINTY)
    from src import analysis
    assert (analysis.analyse(offset, screen.candidate.members).p_max
            <= analysis.analyse(plain, screen.candidate.members).p_max + 1e-6)
