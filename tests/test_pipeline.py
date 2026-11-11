# The pipeline end to end (on a small candidate subset to keep the tests quick) and its CLI.

import json
import subprocess
import sys

import pytest

import config
from scripts import run_pipeline as cli
from src import geometry, packing
from src import pipeline as pl
from src import scoring as sc
from src import topology as tp
from tests.test_optimiser import warren_candidate


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """A real run on 41 candidates (including the Warren layout) with small settings."""
    out = tmp_path_factory.mktemp("pipeline")
    subset = tp.enumerate_candidates(6, 7)[:40] + [warren_candidate()]
    original = pl.topology.enumerate_candidates
    pl.topology.enumerate_candidates = lambda *a, **k: subset
    try:
        opts = pl.Options(workers=2, seed=5, n_samples=40, top_k=6, n_starts=2, out_dir=out,
                          quiet=True)
        result = pl.run(opts)
    finally:
        pl.topology.enumerate_candidates = original
    return result, out


def test_quick_option_applies_small_settings():
    o = pl.Options(quick=True).resolved()
    assert (o.n_samples, o.top_k, o.n_starts) == (40, 6, 2)
    plain = pl.Options().resolved()
    assert plain.n_samples == config.SCREEN_SAMPLES_PER_CANDIDATE and plain.top_k == config.TOP_K


def test_pipeline_selects_one_legal_design_per_objective(small_run):
    result, _ = small_run
    assert set(result.designs) == set(sc.OBJECTIVES)
    for objective, design in result.designs.items():
        assert design.objective == objective and design.evaluation.valid
        assert design.evaluation.violations["material"] == 0
        assert design.candidate.n_joints <= config.J_MAX
        assert packing.is_packable(geometry.member_lengths(design.coords, design.candidate.members))
        assert result.ranked[objective][0] is design
        scores = [d.score for d in result.ranked[objective]]
        assert scores == sorted(scores)
    assert result.designs[sc.OBJ_MAX_LOAD].p_max >= result.designs[sc.OBJ_MAX_LOAD_PER_COST].p_max - 1e-6
    assert (result.designs[sc.OBJ_MAX_LOAD_PER_COST].p_over_cost
            >= result.designs[sc.OBJ_MAX_LOAD].p_over_cost - 1e-6)


def test_pipeline_writes_every_output(small_run):
    result, out = small_run
    assert result.names == {sc.OBJ_MAX_LOAD: "truss1_max_load",
                            sc.OBJ_MAX_LOAD_PER_COST: "truss2_max_load_per_cost"}
    for objective, paths in result.paths.items():
        assert set(paths) == {"figure", "report", "json", "matlab_mat", "matlab_text"}
        for p in paths.values():
            assert p.exists() and p.stat().st_size > 0 and out in p.parents
        data = json.loads(paths["json"].read_text())
        assert data["objective"] == objective
        assert data["metrics"]["p_max_lower"] < data["metrics"]["p_max"]
        assert "Cutting plan" in paths["report"].read_text()
    assert result.summary_path.exists()
    text = result.summary_path.read_text()
    assert "truss1_max_load" in text and "truss2_max_load_per_cost" in text and "seed 5" in text


def test_pipeline_runs_the_sensitivity_check(small_run):
    result, _ = small_run
    for objective in sc.OBJECTIVES:
        rep = result.sensitivity[objective]
        assert rep.objective == objective and rep.rows
        assert rep.regret is not None and 0 <= rep.regret < 0.1
        assert any("Regret" in line for line in rep.summary_lines())


def run_small(tmp_path, **kw):
    """A tiny real pipeline run (three candidates, one of them the Warren layout)."""
    subset = tp.enumerate_candidates(6, 7)[:2] + [warren_candidate()]
    original = pl.topology.enumerate_candidates
    pl.topology.enumerate_candidates = lambda *a, **k: subset
    try:
        opts = dict(workers=1, n_samples=30, top_k=3, n_starts=2, out_dir=tmp_path, regret=False,
                    quiet=True)
        opts.update(kw)
        return pl.run(pl.Options(**opts))
    finally:
        pl.topology.enumerate_candidates = original


def fake_matlab_run(perturb=None):
    """A stand-in for matlab_io.run_matlab that applies the Python re-implementation of the script."""
    from scipy.io import loadmat
    from tests.test_matlab_io import simulate_truss_analysis

    def run(inputs, workdir=None, **_):
        out = {}
        for path in inputs:
            data = loadmat(str(path))
            result = simulate_truss_analysis({k: data[k] for k in ("C", "X", "Y", "Sx", "Sy", "L")})
            if perturb:
                perturb(result)
            out[Path(path).resolve()] = result
        return out
    return run


from pathlib import Path  # noqa: E402

requires_matlab = pytest.mark.skipif(not pl.matlab_io.matlab_available(),
                                     reason="MATLAB is not installed")


def test_matlab_not_requested_by_default(tmp_path):
    result = run_small(tmp_path)
    assert result.matlab_status == "not requested" and not result.matlab_failed
    assert result.verification == {}
    assert "MATLAB verification (matlab" not in result.summary_path.read_text()


@requires_matlab
def test_live_matlab_verification_in_the_pipeline(tmp_path):
    result = run_small(tmp_path, matlab=True)
    assert result.matlab_status == "passed" and not result.matlab_failed
    for objective in sc.OBJECTIVES:
        cmp = result.verification[objective]
        assert cmp.ok, "\n".join(cmp.lines())
        paths = result.paths[objective]
        report = paths["report"].read_text()
        assert "MATLAB verification (truss_analysis.m vs Python solver" in report and "PASSED" in report
        assert "Forces shown are the MATLAB" in report
        data = json.loads(paths["json"].read_text())
        assert data["matlab_verification"]["ok"] is True
        assert paths["figure"].stat().st_size > 10_000
        assert (tmp_path / "matlab_io" / f"{result.names[objective]}_output.txt").exists()
        expected = min(len(result.ranked[objective]), pl.config.N_FINALISTS) - 1
        assert len(result.finalist_checks[objective]) == expected      # every runner-up checked
        assert all(c.ok for _, c in result.finalist_checks[objective])
    text = result.summary_path.read_text()
    assert "MATLAB verification (matlab/truss_analysis.m): passed" in text
    assert (tmp_path / "matlab_io" / "matlab_run.log").exists()


def test_matlab_requested_but_missing_is_skipped_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(pl.config, "MATLAB_EXECUTABLE", str(tmp_path / "no_such_matlab"))
    result = run_small(tmp_path, matlab=True)
    assert result.matlab_status.startswith("skipped") and result.matlab_failed
    assert any("MATLAB was not found" in w for w in result.warnings)
    for paths in result.paths.values():                       # the Python results are still written
        assert paths["report"].exists() and paths["figure"].exists()
    assert "MATLAB verification (matlab/truss_analysis.m): skipped" in result.summary_path.read_text()


def test_matlab_that_fails_to_run_is_reported_not_raised(monkeypatch, tmp_path):
    fake = tmp_path / "fake_matlab"
    fake.write_text("#!/bin/sh\necho 'licence checkout failed'; exit 3\n")
    fake.chmod(0o755)
    monkeypatch.setattr(pl.config, "MATLAB_EXECUTABLE", str(fake))
    result = run_small(tmp_path, matlab=True)
    assert result.matlab_status.startswith("FAILED to run") and result.matlab_failed
    assert any("licence checkout failed" in w for w in result.warnings)
    assert result.designs                                       # nothing was lost


def test_matlab_agreement_updates_the_winners_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(pl.matlab_io, "matlab_available", lambda: True)
    monkeypatch.setattr(pl.matlab_io, "run_matlab", fake_matlab_run())
    result = run_small(tmp_path, matlab=True)
    assert result.matlab_status == "passed"
    for objective in sc.OBJECTIVES:
        report = result.paths[objective]["report"].read_text()
        assert "PASSED" in report and "Forces shown are the MATLAB" in report


def test_matlab_mismatch_is_flagged_and_python_forces_are_kept(monkeypatch, tmp_path):
    def break_max_load(out):
        out.max_load *= 1.05
    monkeypatch.setattr(pl.matlab_io, "matlab_available", lambda: True)
    monkeypatch.setattr(pl.matlab_io, "run_matlab", fake_matlab_run(break_max_load))
    result = run_small(tmp_path, matlab=True)
    assert result.matlab_status == "FAILED: mismatch" and result.matlab_failed
    assert any("MATLAB verification FAILED for truss1_max_load" in w for w in result.warnings)
    for objective in sc.OBJECTIVES:
        report = result.paths[objective]["report"].read_text()
        assert "FAILED" in report and "MISMATCH" in report
        assert "the MATLAB check FAILED" in report and "Forces shown are the MATLAB" not in report
        assert json.loads(result.paths[objective]["json"].read_text())["matlab_verification"]["ok"] is False
    assert "WARNINGS" in result.summary_path.read_text()


def test_quiet_run_prints_nothing(capsys, tmp_path):
    subset = [warren_candidate()]
    original = pl.topology.enumerate_candidates
    pl.topology.enumerate_candidates = lambda *a, **k: subset
    try:
        pl.run(pl.Options(workers=1, n_samples=30, top_k=2, n_starts=1, out_dir=tmp_path,
                          regret=False, quiet=True))
    finally:
        pl.topology.enumerate_candidates = original
    assert capsys.readouterr().out == ""


def test_pipeline_error_when_nothing_is_legal(tmp_path):
    # a candidate that matches no seed layout, with no random samples: nothing can be legal
    cand = tp.candidates_for_joints(7)[0]
    original = pl.topology.enumerate_candidates
    pl.topology.enumerate_candidates = lambda *a, **k: [cand]
    try:
        with pytest.raises(pl.PipelineError, match="no legal design"):
            pl.run(pl.Options(workers=1, n_samples=0, top_k=1, n_starts=1, out_dir=tmp_path,
                              quiet=True))
    finally:
        pl.topology.enumerate_candidates = original


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def test_cli_parser_defaults_and_flags(tmp_path):
    p = cli.build_parser()
    d = p.parse_args([])
    assert d.seed == config.RANDOM_SEED and d.samples == config.SCREEN_SAMPLES_PER_CANDIDATE
    assert d.top_k == config.TOP_K and d.starts == config.SLSQP_STARTS
    assert not d.quick and not d.matlab and not d.no_regret and d.out_dir is None
    a = p.parse_args(["--quick", "--workers", "3", "--seed", "9", "--matlab", "--no-regret",
                      "--out-dir", str(tmp_path), "--samples", "11", "--top-k", "4", "--starts", "5"])
    assert (a.quick, a.workers, a.seed, a.matlab, a.no_regret) == (True, 3, 9, True, True)
    assert (a.samples, a.top_k, a.starts, a.out_dir) == (11, 4, 5, tmp_path)


def test_cli_help_runs_as_a_script():
    out = subprocess.run([sys.executable, str(config.ROOT / "scripts" / "run_pipeline.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--quick" in out.stdout and "--matlab" in out.stdout


def test_cli_exit_status_3_when_matlab_verification_fails(monkeypatch, tmp_path, capsys):
    class Failed:
        matlab_failed = True
        matlab_status = "FAILED: mismatch"
    monkeypatch.setattr(cli.pipeline, "run", lambda _options: Failed())
    assert cli.main(["--matlab", "--out-dir", str(tmp_path)]) == 3
    assert "FAILED: mismatch" in capsys.readouterr().err

    class Fine:
        matlab_failed = False
        matlab_status = "passed"
    monkeypatch.setattr(cli.pipeline, "run", lambda _options: Fine())
    assert cli.main(["--matlab", "--out-dir", str(tmp_path)]) == 0


def test_cli_returns_error_code_when_nothing_is_legal(monkeypatch, tmp_path, capsys):
    def boom(_options):
        raise pl.PipelineError("no legal design found")
    monkeypatch.setattr(cli.pipeline, "run", boom)
    assert cli.main(["--out-dir", str(tmp_path)]) == 1
    assert "no legal design found" in capsys.readouterr().err
