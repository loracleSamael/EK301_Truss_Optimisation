# Report, drawing, JSON and combined output writer.

import json

import numpy as np
import pytest

import config
from src import geometry_opt as go
from src import optimiser as op
from src import scoring as sc
from src import visualise as vz
from tests.test_optimiser import warren_candidate


@pytest.fixture(scope="module")
def design():
    cand = warren_candidate()
    return go.optimise_candidate(op.screen_candidate(cand, 40), sc.OBJ_MAX_LOAD_PER_COST,
                                 n_starts=2)


def test_summary_matches_evaluation(design):
    s = vz.summary(design)
    assert s["p_max"] == design.p_max and s["cost"] == design.cost
    assert np.isclose(s["p_over_cost"], s["p_max"] / s["cost"])
    assert np.isclose(s["material_left"], config.MATERIAL_BUDGET - s["total_length"])
    assert s["n_joints"] == 7 and s["n_members"] == 11


def test_member_rows_forces_are_scaled_to_failure_load(design):
    rows = vz.member_rows(design)
    unit = design.evaluation.analysis.forces
    scale = design.p_max / config.TEST_LOAD
    assert np.allclose([r["force"] for r in rows], unit * scale)
    assert [r["id"] for r in rows] == list(range(1, 12))
    assert all(r["kind"] in ("T", "C", "zero") for r in rows)
    # the governing compression member is exactly at its buckling capacity
    gov = rows[design.evaluation.analysis.governing]
    assert gov["kind"] == "C" and np.isclose(gov["utilisation"], 1.0)
    assert max(r["utilisation"] for r in rows) <= 1.0 + 1e-9
    for r in rows:
        assert (r["kind"] == "T") == (r["force"] > 0)
        if r["kind"] != "C":
            assert r["utilisation"] == 0.0


def test_member_rows_are_one_based(design):
    rows = vz.member_rows(design)
    first = design.candidate.members[0]
    assert rows[0]["joints"] == (first[0] + 1, first[1] + 1)


def test_forces_override_replaces_python_forces(design):
    unit = design.evaluation.analysis.forces
    rows = vz.member_rows(design, forces=unit * 0.5)
    assert np.allclose([r["force"] for r in rows], unit * 0.5 * design.p_max)


def test_report_contains_the_key_facts(design):
    text = vz.report_text(design, "My $title")
    assert "My $title" in text
    assert f"{design.p_max:.3f} oz" in text
    assert f"${design.cost:.2f}" in text
    assert "m1 " in text and "m11" in text and "m12" not in text
    assert "pin support" in text and "roller support" in text and "load joint" in text
    assert "-0.000" not in text
    assert "not verified by MATLAB" in text                      # MATLAB status is stated
    # one joint line per joint, one member line per member
    assert sum(1 for line in text.splitlines() if line.strip().startswith("m")
               and "-" in line.split()[1]) == design.candidate.n_members


def test_reactions_sum_to_the_failure_load(design):
    r = vz.reaction_values(design)
    assert np.isclose(r["pin_y"] + r["roller_y"], design.p_max)
    assert r["pin_x"] == 0.0


def test_draw_truss_saves_a_png(design, tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = vz.draw_truss(design, "A $5 title", path=out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_draw_truss_is_to_scale(design):
    import matplotlib.pyplot as plt
    fig = vz.draw_truss(design)
    ax = fig.axes[0]
    assert ax.get_aspect() == 1.0 or ax.get_aspect() == "equal"
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for x, y in design.coords:
        assert xlim[0] < x < xlim[1] and ylim[0] < y < ylim[1]
    plt.close(fig)


def test_cutting_plan_covers_every_member_and_fits_the_sticks(design):
    plan = vz.cutting_plan(design)
    lengths = np.hypot(*np.diff(design.coords[np.array(design.candidate.members)], axis=1)[:, 0].T)
    assert [p["capacity"] for p in plan] == list(config.STOCK_LENGTHS)
    assert sorted(m for p in plan for m in p["members"]) == list(range(len(lengths)))
    for p in plan:
        assert np.isclose(p["used"], lengths[p["members"]].sum() if p["members"] else 0.0)
        assert p["left"] >= -1e-9


def test_report_has_the_cutting_plan(design):
    text = vz.report_text(design)
    assert "Cutting plan" in text
    for letter in "ABC":
        assert f"Stick {letter} (" in text
    assert text.count("left ") >= 3
    assert "sticks 48 + 48 + 24 in" in text
    # every member id (1-based) appears in a cutting-plan line
    plan_lines = [ln for ln in text.splitlines() if ln.strip().startswith("Stick ")]
    for k in range(1, design.candidate.n_members + 1):
        assert any(f"m{k} (" in ln for ln in plan_lines), k


def test_figure_has_drawing_member_table_and_cutting_table(design):
    import matplotlib.pyplot as plt
    fig = vz.draw_truss(design)
    assert len(fig.axes) == 3
    cut_texts = [t.get_text() for t in fig.axes[2].texts]
    assert any("single stick" in t for t in cut_texts)
    assert len(fig.axes[2].tables[0].get_celld()) == 4 * (len(config.STOCK_LENGTHS) + 1)
    plt.close(fig)


def test_json_round_trip(design):
    d = vz.design_to_dict(design, "x")
    d = json.loads(json.dumps(d))
    assert d["index_base"] == 0
    assert np.allclose(d["joints"], design.coords)
    assert [tuple(m) for m in d["members"]] == list(design.candidate.members)
    assert np.isclose(d["metrics"]["p_max"], design.p_max)
    assert d["roles"] == {"pin": 0, "roller": 1, "load_joint": 2}
    assert d["stock_lengths"] == [48.0, 48.0, 24.0]
    assert [p["capacity"] for p in d["cutting_plan"]] == [48.0, 48.0, 24.0]
    assert sorted(m for p in d["cutting_plan"] for m in p["members"]) == \
        list(range(design.candidate.n_members))                       # 0-based like the rest


def test_write_outputs_creates_every_file(design, tmp_path):
    paths = vz.write_outputs(design, "truss_test", "Title", figures_dir=tmp_path / "f",
                             designs_dir=tmp_path / "d", matlab_dir=tmp_path / "m")
    assert set(paths) == {"figure", "report", "json", "matlab_mat", "matlab_text"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    assert json.loads(paths["json"].read_text())["name"] == "truss_test"
    assert paths["report"].read_text().startswith("=")


def test_report_and_json_show_the_buckling_uncertainty(design):
    a = design.evaluation.analysis
    text = vz.report_text(design)
    assert f"{a.p_max_uncertainty:.2f} oz" in text and f"{a.p_max_lower:.3f} oz" in text
    assert "Lower-bound P_max" in text and "buckling-fit uncertainty" in text
    s = vz.summary(design)
    assert s["p_max_lower"] == a.p_max_lower and s["p_max_uncertainty"] == a.p_max_uncertainty
    d = vz.design_to_dict(design)
    assert d["metrics"]["p_max_lower"] == a.p_max_lower
    assert d["metrics"]["p_max_uncertainty"] == a.p_max_uncertainty
    assert 0 < a.p_max_lower < a.p_max


def _comparison(design, ok=True):
    from src import matlab_io as mio
    rows = [{"name": "member forces, max |diff| (oz/oz)", "python": "0.7532", "matlab": "0.7532",
             "diff": "2.22e-16", "ok": True},
            {"name": "max load, lower bound (oz)", "python": "57.151", "matlab": "57.150" if ok else "60.000",
             "diff": "9.0e-04" if ok else "2.85e+00", "ok": ok}]
    return mio.Comparison(rows=rows, rel_tol=1e-3)


def test_report_shows_the_matlab_verification(design):
    unit = design.evaluation.analysis.forces
    text = vz.report_text(design, forces=unit, verification=_comparison(design))
    assert "MATLAB verification (truss_analysis.m vs Python solver" in text and "PASSED" in text
    assert "checked against the Python solver" in text and "not verified by MATLAB" not in text
    # a failed check keeps the Python forces and says so
    failed = vz.report_text(design, verification=_comparison(design, ok=False))
    assert "FAILED" in failed and "MISMATCH" in failed
    assert "the MATLAB check FAILED" in failed and "Forces shown are the MATLAB" not in failed


def test_json_carries_the_verification(design):
    d = vz.design_to_dict(design, "x", _comparison(design, ok=False))
    assert d["matlab_verification"]["ok"] is False and len(d["matlab_verification"]["checks"]) == 2
    assert "matlab_verification" not in vz.design_to_dict(design, "x")


def test_verified_forces_replace_the_python_ones_in_the_report(design):
    unit = design.evaluation.analysis.forces
    text = vz.report_text(design, forces=unit * 0.5, verification=_comparison(design))
    first = vz.member_rows(design, forces=unit * 0.5)[0]["force"]
    assert f"{first:.3f}" in text


def test_figure_footnote_says_verified(design):
    import matplotlib.pyplot as plt
    fig = vz.draw_truss(design, verified=True)
    assert any("verified by MATLAB" in t.get_text() for t in fig.axes[1].texts)
    plt.close(fig)
    fig = vz.draw_truss(design)
    assert not any("verified by MATLAB" in t.get_text() for t in fig.axes[1].texts)
    plt.close(fig)
