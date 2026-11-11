# test_matlab_io.py
# Matrix shapes and connection matrix correctness, coordinate rounding, file export, a check that the
# exported matrices reproduce the Python forces, the Python-vs-MATLAB comparison (against a Python
# re-implementation of truss_analysis.m), the failure paths of run_matlab (fake launchers), and live
# tests that run the real MATLAB (skipped when MATLAB is not installed).

import re
import stat

import numpy as np
import pytest
from scipy.io import loadmat

import config
from src import geometry_opt as go
from src import matlab_io as mio
from src import optimiser as op
from src import scoring as sc
from tests.test_optimiser import warren_candidate


@pytest.fixture(scope="module")
def design():
    cand = warren_candidate()
    return go.optimise_candidate(op.screen_candidate(cand, 40), sc.OBJ_MAX_LOAD, n_starts=2)


def test_matrix_shapes_and_contents(design):
    n_j, n_m = design.candidate.n_joints, design.candidate.n_members
    m = mio.build_matrices(design.coords, design.candidate.members)
    assert m["C"].shape == (n_j, n_m)
    assert m["X"].shape == (1, n_j) and m["Y"].shape == (1, n_j)
    assert m["Sx"].shape == (n_j, 3) and m["Sy"].shape == (n_j, 3)
    assert m["L"].shape == (2 * n_j, 1)
    assert np.all(m["C"].sum(axis=0) == 2)                       # every member joins two joints
    assert m["Sx"].sum() == 1 and m["Sx"][config.PIN_INDEX, 0] == 1
    assert m["Sy"].sum() == 2
    assert m["Sy"][config.PIN_INDEX, 1] == 1 and m["Sy"][config.ROLLER_INDEX, 2] == 1
    assert np.count_nonzero(m["L"]) == 1
    assert m["L"][n_j + config.LOAD_INDEX, 0] == mio.LOAD_SIGN * config.TEST_LOAD
    assert np.allclose(m["X"].ravel(), design.coords[:, 0])
    assert np.allclose(m["Y"].ravel(), design.coords[:, 1])


def test_connection_matrix_matches_members(design):
    m = mio.build_matrices(design.coords, design.candidate.members)
    for k, (i, j) in enumerate(design.candidate.members):
        assert set(np.flatnonzero(m["C"][:, k])) == {i, j}


def test_load_value_is_used(design):
    m = mio.build_matrices(design.coords, design.candidate.members, load=25.0)
    assert m["L"].sum() == mio.LOAD_SIGN * 25.0


def test_round_for_export():
    r = mio.round_for_export([[0.004, 0.0], [8.6149, 1.4051]])
    assert np.allclose(r, [[0.0, 0.0], [8.61, 1.41]])


def test_exported_matrices_reproduce_python_forces(design):
    m = mio.build_matrices(design.coords, design.candidate.members)
    f_matlab = mio.solve_exported(m)
    res = design.evaluation.analysis
    python = np.concatenate([res.forces, res.reactions])
    assert np.allclose(f_matlab, mio.MATLAB_FORCE_SIGN * python, atol=1e-9)


def test_buckling_ratio_from_exported_matrices_gives_p_max(design):
    # Re-do the script's max-load step on the exported data: scale the load by the smallest
    # capacity / compressive force ratio. It must agree with the Python P_max.
    m = mio.build_matrices(design.coords, design.candidate.members)
    f = mio.solve_exported(m)[:design.candidate.n_members] * mio.MATLAB_FORCE_SIGN
    lengths = np.hypot(*np.diff(design.coords[np.array(design.candidate.members)], axis=1)[:, 0].T)
    caps = config.BUCKLING_A * lengths ** config.BUCKLING_B
    p_max = min(caps[k] / -f[k] for k in range(len(f)) if f[k] < -1e-9)
    assert np.isclose(p_max, design.p_max)


def test_export_writes_mat_and_text(design, tmp_path):
    paths = mio.export_inputs(design.coords, design.candidate.members, "t1", tmp_path)
    assert paths["mat"].exists() and paths["text"].exists()
    expected = mio.build_matrices(mio.round_for_export(design.coords), design.candidate.members)
    loaded = loadmat(str(paths["mat"]))
    for name in mio.MAT_VARIABLES:
        assert np.allclose(loaded[name], expected[name]), name
    text = paths["text"].read_text()
    assert "PLACEHOLDER" in text and "negative" in text
    for name in mio.MAT_VARIABLES:
        block = re.search(rf"^{name} = \[\n(.*?)\n\];", text, re.S | re.M)
        assert block, name
        rows = [[float(v) for v in row.split(",")] for row in block.group(1).split(";")]
        assert np.allclose(np.array(rows), expected[name], atol=1e-5), name


# --------------------------------------------------------------------------------------
# A Python re-implementation of matlab/truss_analysis.m (same constants and compression-only loop)
# --------------------------------------------------------------------------------------
def simulate_truss_analysis(matrices) -> mio.MatlabOutput:
    f = mio.solve_exported(matrices)                      # e \\ L, MATLAB sign convention
    C, X, Y = matrices["C"], matrices["X"].ravel(), matrices["Y"].ravel()
    n_m = C.shape[1]
    lengths = np.array([np.hypot(*(np.array([X[b], Y[b]]) - np.array([X[a], Y[a]])))
                        for a, b in (np.flatnonzero(C[:, m]) for m in range(n_m))])
    buckling = 3391.3 * lengths ** -2.012 - 1.52
    ratio, gov = 1e37, 0
    for i in range(n_m):
        if f[i] > 0 and abs(buckling[i] / f[i]) < ratio:        # only compression can buckle
            ratio, gov = buckling[i] / f[i], i + 1
    load = -matrices["L"].sum()
    n_j = C.shape[0]
    return mio.MatlabOutput(member_forces=f[:n_m], reactions=f[n_m:], member_lengths=lengths,
                            buckling_forces=buckling, governing=gov, ratio=ratio, applied_load=load,
                            max_load=load * ratio, cost=10 * n_j + lengths.sum())


def test_comparison_passes_against_the_simulated_script(design):
    out = simulate_truss_analysis(mio.build_matrices(design.coords, design.candidate.members))
    cmp = mio.compare_results(design, out)
    assert cmp.ok, "\n".join(cmp.lines())
    assert len(cmp.rows) == 6 and "MISMATCH" not in "\n".join(cmp.lines())
    # the simulated script's limit is the lower-bound one, not the nominal one
    assert np.isclose(out.max_load, design.evaluation.analysis.p_max_lower, rtol=1e-3)
    assert out.max_load < design.p_max


def test_comparison_detects_each_kind_of_mismatch(design):
    import copy
    base = simulate_truss_analysis(mio.build_matrices(design.coords, design.candidate.members))

    def broken(**changes):
        out = copy.deepcopy(base)
        for k, v in changes.items():
            setattr(out, k, v)
        return mio.compare_results(design, out)

    assert not broken(member_forces=base.member_forces * -1).ok          # wrong sign convention
    assert not broken(reactions=base.reactions * 1.1).ok
    assert not broken(member_lengths=base.member_lengths + 0.01).ok
    assert not broken(cost=base.cost + 5).ok
    assert not broken(max_load=base.max_load * 1.02).ok
    bad = broken(max_load=base.max_load * 1.02)
    assert any(r["name"].startswith("max load") and not r["ok"] for r in bad.rows)
    assert "MISMATCH" in "\n".join(bad.lines())


def test_comparison_uses_the_lower_bound_not_the_nominal_p_max(design):
    out = simulate_truss_analysis(mio.build_matrices(design.coords, design.candidate.members))
    import copy
    wrong = copy.deepcopy(out)
    wrong.max_load = design.p_max                       # comparing against the nominal value
    assert not mio.compare_results(design, wrong).ok


def test_comparison_handles_a_different_applied_load(design):
    out = simulate_truss_analysis(mio.build_matrices(design.coords, design.candidate.members,
                                                     load=25.0))
    assert mio.compare_results(design, out, load=25.0).ok
    forces = mio.matlab_forces_as_python(out, load=25.0)
    assert np.allclose(forces, design.evaluation.analysis.forces, atol=1e-9)


def test_diagnose_reports_sign_convention_and_shape_problems(design):
    import copy
    base = simulate_truss_analysis(mio.build_matrices(design.coords, design.candidate.members))
    assert mio.diagnose(design, base) == []
    flipped = copy.deepcopy(base)
    flipped.member_forces = -base.member_forces                  # as if MATLAB used the other sign
    msgs = mio.diagnose(design, flipped)
    assert len(msgs) == 1 and msgs[0].startswith("sign convention")
    scrambled = copy.deepcopy(base)
    scrambled.member_forces = np.roll(base.member_forces, 1)
    assert "differ" in mio.diagnose(design, scrambled)[0]
    short = copy.deepcopy(base)
    short.member_forces = base.member_forces[:-1]
    assert "member_forces has shape" in mio.diagnose(design, short)[0]
    cmp = mio.compare_results(design, short)                      # must not crash on a bad shape
    assert not cmp.ok and cmp.rows[0]["name"] == "output shapes"


# --------------------------------------------------------------------------------------
# Finding MATLAB, reading outputs and the failure paths of run_matlab (no real MATLAB needed)
# --------------------------------------------------------------------------------------
def make_fake_launcher(tmp_path, body, name="fake_matlab"):
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_find_matlab_honours_the_config_override(monkeypatch, tmp_path):
    fake = make_fake_launcher(tmp_path, "exit 0")
    monkeypatch.setattr(config, "MATLAB_EXECUTABLE", str(fake))
    assert mio.find_matlab() == fake and mio.matlab_available()
    monkeypatch.setattr(config, "MATLAB_EXECUTABLE", str(tmp_path / "missing"))
    assert mio.find_matlab() is None and not mio.matlab_available()


def test_find_matlab_uses_path_then_applications(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MATLAB_EXECUTABLE", None)
    fake = make_fake_launcher(tmp_path, "exit 0", "matlab")
    monkeypatch.setattr(mio.shutil, "which", lambda name: str(fake))
    assert mio.find_matlab() == fake
    monkeypatch.setattr(mio.shutil, "which", lambda name: None)
    monkeypatch.setattr(mio.glob, "glob", lambda pattern: ["/Applications/MATLAB_R2024b.app/bin/matlab",
                                                           "/Applications/MATLAB_R2025a.app/bin/matlab"])
    assert str(mio.find_matlab()).endswith("MATLAB_R2025a.app/bin/matlab")      # newest wins
    monkeypatch.setattr(mio.glob, "glob", lambda pattern: [])
    assert mio.find_matlab() is None


def test_output_paths():
    mat, txt = mio.output_paths("/x/truss1_input.mat")
    assert str(mat) == "/x/truss1_output.mat" and str(txt) == "/x/truss1_output.txt"
    with pytest.raises(ValueError):
        mio.output_paths("/x/other.mat")


def test_read_output_errors(tmp_path):
    from scipy.io import savemat
    with pytest.raises(mio.MatlabError, match="no output"):
        mio.read_output(tmp_path / "a_input.mat")
    savemat(str(tmp_path / "b_output.mat"), {"error_message": "Undefined variable", "member_forces": [],
                                              "reactions": [], "member_lengths": [],
                                              "buckling_forces": [], "governing": 0, "ratio": 0,
                                              "applied_load": 0, "max_load": 0, "cost": 0})
    with pytest.raises(mio.MatlabError, match="Undefined variable"):
        mio.read_output(tmp_path / "b_input.mat")


def test_run_matlab_without_matlab(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MATLAB_EXECUTABLE", str(tmp_path / "missing"))
    with pytest.raises(mio.MatlabNotAvailable, match="not found"):
        mio.run_matlab([tmp_path / "x_input.mat"])


def test_run_matlab_reports_a_missing_input(tmp_path):
    fake = make_fake_launcher(tmp_path, "exit 0")
    with pytest.raises(mio.MatlabError, match="input file not found"):
        mio.run_matlab([tmp_path / "nope_input.mat"], executable=fake)
    assert mio.run_matlab([], executable=fake) == {}


def test_run_matlab_reports_a_failing_launcher(tmp_path, design):
    inp = mio.export_inputs(design.coords, design.candidate.members, "t", tmp_path)["mat"]
    fake = make_fake_launcher(tmp_path, "echo 'licence checkout failed'; exit 3")
    with pytest.raises(mio.MatlabError, match="(?s)status 3.*licence checkout failed"):
        mio.run_matlab([inp], executable=fake)
    assert "licence checkout failed" in (tmp_path / "matlab_run.log").read_text()


def test_run_matlab_reports_missing_results(tmp_path, design):
    inp = mio.export_inputs(design.coords, design.candidate.members, "t", tmp_path)["mat"]
    fake = make_fake_launcher(tmp_path, "exit 0")              # "succeeds" but writes nothing
    with pytest.raises(mio.MatlabError, match="no output"):
        mio.run_matlab([inp], executable=fake)


def test_run_matlab_times_out(tmp_path, design):
    inp = mio.export_inputs(design.coords, design.candidate.members, "t", tmp_path)["mat"]
    fake = make_fake_launcher(tmp_path, "sleep 5")
    with pytest.raises(mio.MatlabError, match="did not finish"):
        mio.run_matlab([inp], executable=fake, timeout=0.5)


def test_run_matlab_removes_stale_results_and_passes_a_batch_command(tmp_path, design):
    inp = mio.export_inputs(design.coords, design.candidate.members, "t", tmp_path)["mat"]
    stale_mat, stale_txt = mio.output_paths(inp)
    stale_mat.write_text("old"); stale_txt.write_text("old")
    fake = make_fake_launcher(tmp_path, 'echo "$@" > "%s/args.txt"; exit 0' % tmp_path)
    with pytest.raises(mio.MatlabError, match="no output"):    # the stale files are not reused
        mio.run_matlab([inp], executable=fake)
    args = (tmp_path / "args.txt").read_text()
    assert "-batch" in args and "run_verification" in args and "addpath" in args
    assert (tmp_path / "_matlab_batch_list.txt").read_text().strip() == str(inp.resolve())


# --------------------------------------------------------------------------------------
# Live tests: the real MATLAB (skipped when it is not installed)
# --------------------------------------------------------------------------------------
requires_matlab = pytest.mark.skipif(not mio.matlab_available(), reason="MATLAB is not installed")


@pytest.fixture(scope="module")
def live_results(design, tmp_path_factory):
    folder = tmp_path_factory.mktemp("live")
    seed = op.seed_geometries(warren_candidate())[0]
    other = go.finalise(warren_candidate(), seed, sc.OBJ_MAX_LOAD, polish=False)
    designs = {"design_a": design, "design_b": other}
    inputs = {n: mio.export_inputs(d.coords, d.candidate.members, n, folder)["mat"]
              for n, d in designs.items()}
    outputs = mio.run_matlab(list(inputs.values()), folder)
    return folder, designs, inputs, outputs


@requires_matlab
def test_live_matlab_agrees_with_python(live_results):
    _, designs, inputs, outputs = live_results
    for name, d in designs.items():
        cmp = mio.compare_results(d, outputs[inputs[name].resolve()])
        assert cmp.ok, name + "\n" + "\n".join(cmp.lines())


@requires_matlab
def test_live_matlab_writes_logs_and_outputs(live_results):
    folder, designs, inputs, outputs = live_results
    assert (folder / "matlab_run.log").exists()
    for name in designs:
        text = (folder / f"{name}_output.txt").read_text()
        assert "Member forces in oz" in text and "Theoretical max load" in text
        assert (folder / f"{name}_output.mat").exists()


@requires_matlab
def test_live_matlab_only_lets_compression_members_limit_the_load(live_results):
    _, designs, inputs, outputs = live_results
    d = designs["design_a"]
    out = outputs[inputs["design_a"].resolve()]
    forces = d.evaluation.analysis.forces
    assert forces[out.governing - 1] < 0                      # the limiting member is in compression
    # a tension member whose force exceeds its own buckling capacity must not set the limit
    caps = d.evaluation.analysis.capacities
    assert np.any((forces > 0) & (forces * d.p_max > caps))
    assert out.max_load > 0.5 * d.evaluation.analysis.p_max_lower


@requires_matlab
def test_live_matlab_reports_a_bad_input_without_stopping_the_batch(tmp_path, design):
    from scipy.io import savemat
    good = mio.export_inputs(design.coords, design.candidate.members, "good", tmp_path)["mat"]
    bad = tmp_path / "bad_input.mat"
    savemat(str(bad), {"C": np.eye(2)})                       # the other five variables are missing
    list_outputs = {}
    with pytest.raises(mio.MatlabError, match="bad_input.mat"):
        list_outputs = mio.run_matlab([good, bad], tmp_path)
    assert list_outputs == {}
    assert mio.read_output(good).max_load > 0                 # the good one was still processed
