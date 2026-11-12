# visualise.py
# Everything a person reads about a finished design: the to-scale drawing, the text report with the
# same information, a machine-readable JSON copy, and the MATLAB input files (via matlab_io).
#
# Numbering in all human-facing output is 1-based so it matches the MATLAB files:
#   joint 1 = pin, joint 2 = roller, joint 3 = load joint, joints 4+ = free joints; members m1, m2...
# (The JSON file keeps the project's 0-based indices and says so.)
#
# Forces are shown at the FAILURE load P_max, in oz: negative = compression, positive = tension.
# A `forces` argument (member forces at config.TEST_LOAD, analysis.py sign convention) can replace
# the Python solver's values, e.g. with MATLAB-verified ones, in the drawing and the report.

import json
import math

import numpy as np

import config
from src import analysis, geometry, matlab_io, packing

COMPRESSION_COLOUR = "#c62828"
TENSION_COLOUR = "#1565c0"
ZERO_COLOUR = "#757575"
ZERO_FORCE_TOL = 1e-6                # oz at P_max below which a member is labelled zero-force
LOAD_ARROW_LENGTH = 4.0              # in, drawn length of the load arrow


def _mt(text: str) -> str:
    """Escape '$' so matplotlib does not treat it as the start of math text."""
    return text.replace("$", r"\$")


# --------------------------------------------------------------------------------------
# Numbers shared by the drawing and the report
# --------------------------------------------------------------------------------------
def _role(j: int) -> str:
    return {config.PIN_INDEX: "pin support", config.ROLLER_INDEX: "roller support",
            config.LOAD_INDEX: "load joint"}.get(j, "free joint")


def _kind(force: float) -> str:
    if abs(force) <= ZERO_FORCE_TOL:
        return "zero"
    return "T" if force > 0 else "C"


def summary(design) -> dict:
    """Headline numbers for a design."""
    ev = design.evaluation
    lengths = geometry.member_lengths(design.coords, design.candidate.members)
    return {
        "p_max": ev.p_max,
        "p_max_lower": ev.analysis.p_max_lower,
        "p_max_uncertainty": ev.analysis.p_max_uncertainty,
        "cost": ev.cost,
        "p_over_cost": ev.p_over_cost,
        "n_joints": design.candidate.n_joints,
        "n_members": design.candidate.n_members,
        "total_length": float(lengths.sum()),
        "material_budget": config.MATERIAL_BUDGET,
        "material_left": config.MATERIAL_BUDGET - float(lengths.sum()),
        "span": geometry.span(design.coords),
        "load_x": float(design.coords[config.LOAD_INDEX, 0]),
        "load_height": float(design.coords[config.LOAD_INDEX, 1]),
        "objective": design.objective,
    }


def member_rows(design, forces=None) -> list:
    """One dict per member with length, angle, force at P_max, T/C, capacity and utilisation."""
    coords, members = design.coords, design.candidate.members
    res = design.evaluation.analysis
    unit = np.asarray(res.forces if forces is None else forces, dtype=float)
    scale = design.evaluation.p_max / config.TEST_LOAD
    lengths = geometry.member_lengths(coords, members)
    angles = geometry.member_angles(coords, members)
    caps = analysis.buckling_capacity(lengths)
    rows = []
    for k, (i, j) in enumerate(members):
        f = float(unit[k] * scale)
        rows.append({
            "id": k + 1, "joints": (i + 1, j + 1), "length": float(lengths[k]),
            "angle": float(angles[k]), "force": f, "kind": _kind(f),
            "capacity": float(caps[k]),
            "utilisation": abs(f) / caps[k] if f < -ZERO_FORCE_TOL else 0.0,
        })
    return rows


def reaction_values(design) -> dict:
    """Support reactions at P_max (oz): pin x, pin y, roller y."""
    r = np.asarray(design.evaluation.analysis.reactions) * design.evaluation.p_max / config.TEST_LOAD
    r = np.where(np.abs(r) < ZERO_FORCE_TOL, 0.0, r) + 0.0          # avoid printing "-0.000"
    return {"pin_x": float(r[0]), "pin_y": float(r[1]), "roller_y": float(r[2])}


def cutting_plan(design) -> list:
    """Which stick each member is cut from: per stick its name, capacity, members (0-based), used, left."""
    lengths = geometry.member_lengths(design.coords, design.candidate.members)
    return packing.cutting_plan(lengths, packing.min_overflow(lengths).assignment)


def _plan_line(entry: dict, lengths) -> str:
    cuts = ", ".join(f"m{m + 1} ({lengths[m]:.2f})" for m in entry["members"]) or "(unused)"
    return f"Stick {entry['stick']}: {cuts}"


# --------------------------------------------------------------------------------------
# Text report
# --------------------------------------------------------------------------------------
def report_text(design, title: str = "Truss design", forces=None, verification=None) -> str:
    s = summary(design)
    rows = member_rows(design, forces)
    rx = reaction_values(design)
    line = "=" * 78
    out = [line, title, line, ""]
    obj = {"max_load": "maximum failure load (cost ignored)",
           "max_load_per_cost": "maximum failure load / cost"}.get(s["objective"], s["objective"])
    out += [
        f"Objective              : {obj}",
        f"Failure load P_max     : {s['p_max']:.3f} oz  (+/- {s['p_max_uncertainty']:.2f} oz from the "
        f"+/-{config.BUCKLING_UNCERTAINTY:g} oz buckling-fit uncertainty)",
        f"Lower-bound P_max      : {s['p_max_lower']:.3f} oz  (every capacity reduced by "
        f"{config.BUCKLING_UNCERTAINTY:g} oz)",
        f"Cost                   : ${s['cost']:.2f}  "
        f"({s['n_joints']} joints x {config.COST_PER_JOINT:g} + {s['total_length']:.2f} in "
        f"x {config.COST_PER_LENGTH:g})",
        f"P_max / cost           : {s['p_over_cost']:.4f} oz/$",
        f"Joints / members       : {s['n_joints']} / {s['n_members']}",
        f"Material used          : {s['total_length']:.2f} in of {s['material_budget']:g} in "
        f"({s['material_left']:.2f} in left); sticks "
        f"{' + '.join(f'{x:g}' for x in config.STOCK_LENGTHS)} in",
        f"Span                   : {s['span']:.2f} in",
        f"Load joint             : x = {s['load_x']:.2f} in, height = {s['load_height']:.2f} in",
        "",
        "Joints (units: in)",
        f"{'#':>3}  {'role':<15}{'x':>9}{'y':>9}",
    ]
    for j, (x, y) in enumerate(design.coords):
        out.append(f"{j + 1:>3}  {_role(j):<15}{x:>9.2f}{y:>9.2f}")
    out += ["", f"Members (forces at P_max = {s['p_max']:.3f} oz; negative = compression)",
            f"{'ID':>4} {'joints':>8} {'length':>8} {'angle':>7} {'force':>9} {'T/C':>4} "
            f"{'capacity':>9} {'util':>6}",
            f"{'':>4} {'':>8} {'(in)':>8} {'(deg)':>7} {'(oz)':>9} {'':>4} {'(oz)':>9} {'':>6}"]
    for r in rows:
        a, b = r["joints"]
        out.append(f"{'m' + str(r['id']):>4} {f'{a}-{b}':>8} {r['length']:>8.3f} {r['angle']:>7.1f} "
                   f"{r['force']:>9.3f} {r['kind']:>4} {r['capacity']:>9.3f} "
                   f"{r['utilisation']:>6.3f}")
    lengths = geometry.member_lengths(design.coords, design.candidate.members)
    out += ["", "Cutting plan (each member from ONE stick; lengths in in)"]
    for entry in cutting_plan(design):
        out.append(f"  {_plan_line(entry, lengths)}")
        out.append(f"      used {entry['used']:.2f} of {entry['capacity']:g} in, "
                   f"left {entry['left']:.2f} in")
    gov = design.evaluation.analysis.governing
    out += ["",
            f"Reactions at P_max (oz): pin x = {rx['pin_x']:.3f}, pin y = {rx['pin_y']:.3f}, "
            f"roller y = {rx['roller_y']:.3f}",
            f"First member to buckle : m{gov + 1}  (utilisation 1.000)" if gov >= 0 else
            "First member to buckle : none",
            "",
            "Notes",
            "  Joint and member numbers are 1-based (joint 1 = pin, 2 = roller, 3 = load joint).",
            f"  Capacity per member = {config.BUCKLING_A:g} * L^({config.BUCKLING_B:g}) oz; "
            "utilisation = |compressive force| / capacity at P_max; tension members show 0.",
            ("  Forces are from the Python solver (not verified by MATLAB in this run)."
             if verification is None else
             "  Forces shown are the MATLAB (truss_analysis.m) results, checked against the Python "
             "solver." if forces is not None else
             "  Forces are from the Python solver; the MATLAB check FAILED (see below)."),
            ]
    if verification is not None:
        out += ["", "MATLAB verification (truss_analysis.m vs Python solver, tolerance "
                    f"{verification.rel_tol:g}): {'PASSED' if verification.ok else 'FAILED'}"]
        out += [f"  {line}" for line in verification.lines()]
        out.append("  MATLAB's max load uses the lower-bound capacities (3391.3 L^-2.012 - 1.52 oz), "
                   "so it is compared with the Python lower-bound P_max.")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------
def draw_truss(design, title: str = "Truss design", forces=None, path=None, verified: bool = False):
    """To-scale drawing with member labels, supports, load arrow, span dimension and a table.

    Returns the matplotlib Figure; saves it to `path` if given.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    coords, members = design.coords, design.candidate.members
    s = summary(design)
    rows = member_rows(design, forces)
    n_rows = len(rows)

    # Axis limits are set explicitly: annotations do not autoscale, and the load arrow, supports and
    # span dimension line all sit outside the joint bounding box.
    x_lo, x_hi = min(coords[:, 0].min(), 0.0) - 3.0, max(coords[:, 0].max(), s["span"]) + 3.0
    y_dim = min(coords[:, 1].min(), 0.0) - LOAD_ARROW_LENGTH - 2.8
    y_lo, y_hi = y_dim - 1.6, coords[:, 1].max() + 3.5

    fig_w, left, right = 13.0, 0.06, 0.98
    axes_w = fig_w * (right - left)
    draw_h = axes_w * (y_hi - y_lo) / (x_hi - x_lo) + 0.9          # + tick labels and x label
    plan = cutting_plan(design)
    table_h = 0.24 * (n_rows + 1) + 0.7
    cut_h = 0.24 * (len(plan) + 1) + 0.9
    fig_h = draw_h + table_h + cut_h + 1.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(3, 1, height_ratios=[draw_h, table_h, cut_h], hspace=0.12, left=left,
                            right=right, top=1 - 1.1 / fig_h, bottom=0.02)
    ax = fig.add_subplot(grid[0])
    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")
    ax_cut = fig.add_subplot(grid[2])
    ax_cut.axis("off")

    fig.suptitle(_mt(title), fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 1 - 0.7 / fig_h,
             _mt(f"P_max = {s['p_max']:.2f} \u00b1 {s['p_max_uncertainty']:.2f} oz "
                 f"(lower bound {s['p_max_lower']:.2f})    cost = ${s['cost']:.2f}    "
                 f"P_max/cost = {s['p_over_cost']:.4f} oz/$    {s['n_joints']} joints, "
                 f"{s['n_members']} members    material {s['total_length']:.2f}/"
                 f"{s['material_budget']:g} in"),
             ha="center", va="center", fontsize=11)

    centre = coords.mean(axis=0)
    # members
    for r, (i, j) in zip(rows, members):
        colour = {"C": COMPRESSION_COLOUR, "T": TENSION_COLOUR}.get(r["kind"], ZERO_COLOUR)
        p, q = coords[i], coords[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=colour, lw=2.6, solid_capstyle="round", zorder=2)
        mid = (p + q) / 2
        d = q - p
        normal = np.array([-d[1], d[0]]) / max(math.hypot(*d), 1e-9)
        if np.dot(normal, mid - centre) < 0:
            normal = -normal                              # push labels outwards, away from the middle
        pos = mid + normal * 0.9
        label = (f"m{r['id']}  {r['length']:.2f} in\n"
                 f"{r['force']:+.2f} oz ({r['kind']})" if r["kind"] != "zero"
                 else f"m{r['id']}  {r['length']:.2f} in\n0 oz")
        ax.text(pos[0], pos[1], label, ha="center", va="center", fontsize=6.5, color=colour,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=4)

    # joints and labels (supports: below the symbol; load joint: beside the load arrow)
    for j, (x, y) in enumerate(coords):
        ax.plot(x, y, "o", color="black", ms=6, zorder=5)
        away = np.array([x, y]) - centre
        away = away / max(math.hypot(*away), 1e-9)
        if j in (config.PIN_INDEX, config.ROLLER_INDEX):
            pos = np.array([x, y - 3.0])
        elif j == config.LOAD_INDEX:
            pos = np.array([x + 3.0, y - 3.0])
        else:
            pos = np.array([x, y]) + away * 1.7
        ax.text(pos[0], pos[1], f"{j + 1}\n({x:.2f}, {y:.2f})", ha="center", va="center",
                fontsize=7, zorder=6, bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.8))

    # supports
    px, py = coords[config.PIN_INDEX]
    rx_, ry_ = coords[config.ROLLER_INDEX]
    ax.fill([px, px - 0.9, px + 0.9], [py, py - 1.4, py - 1.4], fc="#e0e0e0", ec="black", zorder=3)
    ax.fill([rx_, rx_ - 0.9, rx_ + 0.9], [ry_, ry_ - 1.1, ry_ - 1.1], fc="#e0e0e0", ec="black",
            zorder=3)
    ax.plot([rx_ - 1.2, rx_ + 1.2], [ry_ - 1.3, ry_ - 1.3], color="black", lw=1.5, zorder=3)

    # load arrow: starts at the load joint and points down, clear of the members above it
    lx, ly = coords[config.LOAD_INDEX]
    ax.annotate("", xy=(lx, ly - LOAD_ARROW_LENGTH), xytext=(lx, ly - 0.2),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#2e7d32"), zorder=6)
    ax.text(lx - 0.5, ly - LOAD_ARROW_LENGTH - 0.3, _mt(f"P = {s['p_max']:.2f} oz (failure load)"),
            fontsize=9, color="#2e7d32", ha="center", va="top")

    # span dimension line
    ax.annotate("", xy=(px, y_dim), xytext=(rx_, y_dim),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="black"))
    ax.text((px + rx_) / 2, y_dim + 0.4, f"span = {s['span']:.2f} in", ha="center", va="bottom",
            fontsize=9)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x (in)")
    ax.set_ylabel("y (in)")
    ax.grid(True, alpha=0.2)
    ax.legend(handles=[Line2D([0], [0], color=COMPRESSION_COLOUR, lw=3, label="compression (C)"),
                       Line2D([0], [0], color=TENSION_COLOUR, lw=3, label="tension (T)")],
              loc="upper right", fontsize=8, framealpha=0.9)

    # member table
    header = ["ID", "Joints", "Length (in)", "Angle (deg)", "Force (oz)", "T/C", "Capacity (oz)",
              "Utilisation"]
    cells = [[f"m{r['id']}", f"{r['joints'][0]}-{r['joints'][1]}", f"{r['length']:.3f}",
              f"{r['angle']:.1f}", f"{r['force']:.3f}", r["kind"], f"{r['capacity']:.3f}",
              f"{r['utilisation']:.3f}"] for r in rows]
    table = ax_table.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")
    lengths = geometry.member_lengths(coords, members)
    cut_cells = [[e["stick"], ", ".join(f"m{m + 1} ({lengths[m]:.2f})" for m in e["members"])
                  or "(unused)", f"{e['used']:.2f}", f"{e['left']:.2f}"] for e in plan]
    cut_table = ax_cut.table(cellText=cut_cells, colLabels=["Stick", "Members cut from it (in)",
                                                            "Used (in)", "Left (in)"],
                             loc="upper center", cellLoc="center", colWidths=[0.14, 0.58, 0.14, 0.14])
    cut_table.auto_set_font_size(False)
    cut_table.set_fontsize(8)
    cut_table.scale(1.0, 1.25)
    for (row, _), cell in cut_table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")
    ax_cut.text(0.5, 0.0, "Cutting plan: every member is cut from a single stick (no splicing, "
                "no kerf).", transform=ax_cut.transAxes, ha="center", va="top", fontsize=8)
    ax_table.text(0.5, 0.0, f"Forces at P_max = {s['p_max']:.2f} oz; negative = compression"
                  + ("; verified by MATLAB. " if verified else ". ")
                  + "Joint/member numbers are 1-based (joint 1 = pin, 2 = roller, 3 = load joint).",
                  transform=ax_table.transAxes, ha="center", va="top", fontsize=8)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------------------
# Machine-readable copy and the combined writer
# --------------------------------------------------------------------------------------
def design_to_dict(design, name: str = "", verification=None) -> dict:
    s = summary(design)
    data = {
        "name": name,
        "objective": design.objective,
        "index_base": 0,
        "joints": [[float(x), float(y)] for x, y in design.coords],
        "members": [[int(i), int(j)] for i, j in design.candidate.members],
        "roles": {"pin": config.PIN_INDEX, "roller": config.ROLLER_INDEX,
                  "load_joint": config.LOAD_INDEX},
        "metrics": {k: s[k] for k in ("p_max", "p_max_lower", "p_max_uncertainty", "cost",
                                      "p_over_cost", "total_length", "span", "n_joints",
                                      "n_members")},
        "stock_lengths": [float(x) for x in config.STOCK_LENGTHS],
        "cutting_plan": cutting_plan(design),
        "units": {"length": "in", "force": "oz"},
    }
    if verification is not None:
        data["matlab_verification"] = {
            "ok": bool(verification.ok), "rel_tol": verification.rel_tol,
            "checks": [{"name": r["name"], "python": r["python"], "matlab": r["matlab"],
                        "diff": r["diff"], "ok": bool(r["ok"])} for r in verification.rows]}
    return data


def write_outputs(design, name: str, title: str = None, forces=None,
                  figures_dir=None, designs_dir=None, matlab_dir=None, verification=None,
                  submission_file: str = None) -> dict:
    """Write the drawing (PNG), text report, JSON and MATLAB input files; return their paths.

    `forces` (member forces at config.TEST_LOAD, analysis.py sign convention) replaces the Python
    solver's values in the drawing and report, and `verification` (a matlab_io.Comparison) adds the
    MATLAB verification result to the report and JSON. Pass both after a MATLAB run.
    """
    title = title or name
    figures_dir = config.FIGURES_DIR if figures_dir is None else figures_dir
    designs_dir = config.DESIGNS_DIR if designs_dir is None else designs_dir
    designs_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    png = figures_dir / f"{name}.png"
    fig = draw_truss(design, title, forces, png,
                     verified=verification is not None and verification.ok and forces is not None)
    import matplotlib.pyplot as plt
    plt.close(fig)
    paths["figure"] = png

    txt = designs_dir / f"{name}.txt"
    txt.write_text(report_text(design, title, forces, verification))
    paths["report"] = txt

    js = designs_dir / f"{name}.json"
    js.write_text(json.dumps(design_to_dict(design, name, verification), indent=2) + "\n")
    paths["json"] = js

    paths.update({f"matlab_{k}": v for k, v in matlab_io.export_inputs(
        design.coords, design.candidate.members, name, matlab_dir,
        submission_file=submission_file).items()})
    return paths
