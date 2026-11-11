# Random legal designs and the Python-vs-MATLAB validation (live MATLAB tests are skipped when it is
# not installed; the rest uses a Python re-implementation of truss_analysis.m).

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import config
from scripts import validate_against_matlab as cli
from src import matlab_io as mio
from src import packing
from src import scoring as sc
from src import topology as tp
from src import validation as val
from tests.test_matlab_io import simulate_truss_analysis

requires_matlab = pytest.mark.skipif(not mio.matlab_available(), reason="MATLAB is not installed")


@pytest.fixture(scope="module")
def designs():
    return val.random_valid_designs(12, seed=3)


def fake_run(perturb=None):
    from scipy.io import loadmat

    def run(inputs, workdir=None, **_):
        out = {}
        for path in inputs:
            data = loadmat(str(path))
            result = simulate_truss_analysis({k: data[k] for k in ("C", "X", "Y", "Sx", "Sy", "L")})
            if perturb:
                perturb(Path(path).name, result)
            out[Path(path).resolve()] = result
        return out
    return run


def test_random_designs_are_legal_varied_and_on_the_grid(designs):
    assert len(designs) == 12
    for d in designs:
        assert d.evaluation.valid and d.evaluation.violation == 0
        assert packing.is_packable(np.hypot(*np.diff(d.coords[np.array(d.candidate.members)],
                                                     axis=1)[:, 0].T))
        steps = d.coords / config.COORD_ROUNDING
        assert np.allclose(steps, np.round(steps), atol=1e-6)
    assert len({d.candidate for d in designs}) >= 4                      # several topologies
    assert len({tuple(np.round(d.coords.ravel(), 2)) for d in designs}) == 12   # all different


def test_random_designs_respect_the_per_candidate_limit(designs):
    assert max(Counter(d.candidate for d in designs).values()) <= 3
    few = val.random_valid_designs(6, seed=3, per_candidate=1)
    assert max(Counter(d.candidate for d in few).values()) == 1


def test_random_designs_are_deterministic_and_seed_dependent(designs):
    again = val.random_valid_designs(12, seed=3)
    assert all(np.array_equal(a.coords, b.coords) for a, b in zip(designs, again))
    other = val.random_valid_designs(12, seed=4)
    assert any(not np.array_equal(a.coords, b.coords) for a, b in zip(designs, other))


def test_random_designs_give_up_gracefully():
    cand = tp.candidates_for_joints(7)[0]
    got = val.random_valid_designs(5, candidates=[cand], per_candidate=1)
    assert len(got) <= 1                                     # cannot invent designs it cannot find


def test_validate_passes_against_the_simulated_script(designs, monkeypatch, tmp_path):
    monkeypatch.setattr(val.matlab_io, "run_matlab", fake_run())
    report = val.validate(designs, tmp_path)
    assert report.passed and not report.failures and len(report.names) == 12
    assert all(c.ok for c in report.comparisons.values())
    worst = report.worst_case()
    assert worst["member forces, max |diff| (oz/oz)"] < 1e-9
    text = "\n".join(report.lines())
    assert "RESULT: PASS" in text and "12 designs" in text and "different topologies" in text
    for name in report.names:
        assert (tmp_path / f"{name}_input.mat").exists()


def test_validate_flags_a_disagreement_with_hints(designs, monkeypatch, tmp_path):
    def break_one(filename, out):
        if filename.startswith("val_002"):
            out.member_forces = -out.member_forces            # a sign-convention problem
        if filename.startswith("val_005"):
            out.max_load *= 1.1
    monkeypatch.setattr(val.matlab_io, "run_matlab", fake_run(break_one))
    report = val.validate(designs, tmp_path)
    assert not report.passed and report.failures == ["val_002", "val_005"]
    assert any(h.startswith("sign convention") for h in report.diagnostics["val_002"])
    text = "\n".join(report.lines())
    assert "RESULT: FAIL" in text and "val_002" in text and "val_005" in text and "hint:" in text
    assert report.worst_case()["max load, lower bound (oz)"] > 1.0


def test_an_empty_validation_does_not_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(val.matlab_io, "run_matlab", lambda *a, **k: {})
    assert not val.validate([], tmp_path).passed


@requires_matlab
def test_live_validation_with_real_matlab(designs, tmp_path):
    report = val.validate(designs[:6], tmp_path)
    assert report.passed, "\n".join(report.lines())
    assert report.worst_case()["max load, lower bound (oz)"] < 0.01
    assert (tmp_path / "matlab_run.log").exists()


# --------------------------------------------------------------------------------------
# Command line and exit statuses
# --------------------------------------------------------------------------------------
def test_cli_defaults():
    a = cli.build_parser().parse_args([])
    assert a.designs == 50 and a.seed == config.RANDOM_SEED
    assert a.out_dir == config.MATLAB_IO_DIR / "validation"


def test_cli_exit_2_when_matlab_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "MATLAB_EXECUTABLE", str(tmp_path / "no_such_matlab"))
    assert cli.main(["--out-dir", str(tmp_path)]) == 2
    assert "MATLAB was not found" in capsys.readouterr().err


def test_cli_exit_2_when_matlab_fails_to_run(monkeypatch, tmp_path, capsys, designs):
    monkeypatch.setattr(cli.matlab_io, "matlab_available", lambda: True)
    monkeypatch.setattr(cli.validation, "random_valid_designs", lambda n, seed: designs[:2])

    def boom(*a, **k):
        raise mio.MatlabError("licence checkout failed")
    monkeypatch.setattr(cli.validation, "validate", boom)
    assert cli.main(["--out-dir", str(tmp_path)]) == 2
    assert "licence checkout failed" in capsys.readouterr().err


def test_cli_exit_1_when_no_designs_can_be_generated(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.matlab_io, "matlab_available", lambda: True)
    monkeypatch.setattr(cli.validation, "random_valid_designs", lambda n, seed: [])
    assert cli.main(["--out-dir", str(tmp_path)]) == 1


def test_cli_exit_status_follows_the_report(monkeypatch, tmp_path, capsys, designs):
    monkeypatch.setattr(cli.matlab_io, "matlab_available", lambda: True)
    monkeypatch.setattr(cli.validation, "random_valid_designs", lambda n, seed: designs[:3])
    monkeypatch.setattr(val.matlab_io, "run_matlab", fake_run())
    assert cli.main(["-n", "3", "--out-dir", str(tmp_path)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    monkeypatch.setattr(val.matlab_io, "run_matlab",
                        fake_run(lambda f, o: setattr(o, "cost", o.cost + 5)))
    assert cli.main(["-n", "3", "--out-dir", str(tmp_path)]) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out


@requires_matlab
def test_live_cli_end_to_end(tmp_path, capsys):
    assert cli.main(["-n", "5", "--out-dir", str(tmp_path)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
