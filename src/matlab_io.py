# matlab_io.py
# The bridge between Python designs and the MATLAB analysis files.
#
# Python side of the MATLAB verification: build the six input matrices, run matlab/truss_analysis.m
# in one headless batch (matlab -batch, via matlab/run_verification.m), read the results back and
# compare them with the Python solver.
#
# Input format (shapes and variable names taken from matlab/truss_analysis.m's header and
# matlab/calculate_forces.m; all indices are 1-based in MATLAB, joint k = row k, member m = column m):
#   C   (J x M)   connection matrix, 1 where member (column) touches joint (row)
#   X   (1 x J)   joint x coordinates [in]
#   Y   (1 x J)   joint y coordinates [in]
#   Sx  (J x 3)   x reaction matrix: a 1 at (pin, column 1)
#   Sy  (J x 3)   y reaction matrix: a 1 at (pin, column 2) and (roller, column 3)
#   L   (2J x 1)  external loads: rows 1..J are x loads, rows J+1..2J are y loads [oz]
#
# Unknowns are ordered [member forces..., Sx1 (pin x), Sy1 (pin y), Sy2 (roller y)], the same order
# as analysis.py. Joint order follows the project convention: joint 1 = pin, 2 = roller, 3 = load.
#
# SIGN CONVENTION: L follows calculate_forces.m, which documents downward loads as NEGATIVE entries
# in L ("load = -sum(L)"). truss_analysis.m solves e*F = L, so its F is the negative of analysis.py's
# (its positive = compression, which is also how it prints). MATLAB_FORCE_SIGN records that relation
# in one place.
#
# BUCKLING: truss_analysis.m uses 3391.3 * L^-2.012 - 1.52 oz, i.e. the LOWER-BOUND capacity, so its
# max_load is compared with analysis.py's p_max_lower (not the nominal p_max).

import glob
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import config
from src import geometry

MATLAB_FORCE_SIGN = -1.0      # F_matlab = MATLAB_FORCE_SIGN * F_python, with L negative for down
LOAD_SIGN = -1.0              # sign of a downward load entry in L
MAT_VARIABLES = ("C", "X", "Y", "Sx", "Sy", "L")


# --------------------------------------------------------------------------------------
# Building the six input matrices
# --------------------------------------------------------------------------------------
def build_matrices(coords, members, load: float = config.TEST_LOAD) -> dict:
    """The six MATLAB inputs for a design, as numpy arrays (see the module header for shapes)."""
    c = np.asarray(coords, dtype=float)
    n_j, n_m = len(c), len(members)
    C = np.zeros((n_j, n_m))
    for m, (i, j) in enumerate(members):
        C[i, m] = C[j, m] = 1.0
    Sx = np.zeros((n_j, 3))
    Sy = np.zeros((n_j, 3))
    Sx[config.PIN_INDEX, 0] = 1.0
    Sy[config.PIN_INDEX, 1] = 1.0
    Sy[config.ROLLER_INDEX, 2] = 1.0
    L = np.zeros((2 * n_j, 1))
    L[n_j + config.LOAD_INDEX, 0] = LOAD_SIGN * load
    return {"C": C, "X": c[:, 0].reshape(1, -1), "Y": c[:, 1].reshape(1, -1),
            "Sx": Sx, "Sy": Sy, "L": L}


def round_for_export(coords, step: float = config.COORD_ROUNDING) -> np.ndarray:
    """Coordinates snapped to the export grid (designs from geometry_opt already are)."""
    return np.round(np.round(np.asarray(coords, dtype=float) / step) * step, 6)


def solve_exported(matrices: dict) -> np.ndarray:
    """Python re-implementation of calculate_forces.m's linear algebra, for checking exports.

    Builds the unit-vector matrix e exactly the way that script does and returns F = e \\ L
    (member forces followed by Sx1, Sy1, Sy2) in the MATLAB convention.
    """
    C, X, Y = matrices["C"], matrices["X"].ravel(), matrices["Y"].ravel()
    Sx, Sy, L = matrices["Sx"], matrices["Sy"], matrices["L"].ravel()
    n_j, n_m = C.shape
    e = np.zeros((2 * n_j, n_m + 3))
    e[:n_j, n_m:] = Sx
    e[n_j:, n_m:] = Sy
    for m in range(n_m):
        j1, j2 = np.flatnonzero(C[:, m])
        dx, dy = X[j2] - X[j1], Y[j2] - Y[j1]
        r = np.hypot(dx, dy)
        e[j1, m], e[j2, m] = dx / r, -dx / r
        e[j1 + n_j, m], e[j2 + n_j, m] = dy / r, -dy / r
    return np.linalg.solve(e, L)


# --------------------------------------------------------------------------------------
# Writing files
# --------------------------------------------------------------------------------------
def _matlab_literal(name: str, a: np.ndarray) -> str:
    rows = ["    " + ", ".join(f"{v:g}" for v in row) for row in np.atleast_2d(a)]
    return f"{name} = [\n" + ";\n".join(rows) + "\n];\n"


def write_mat(matrices: dict, path) -> None:
    """Save the six matrices to a .mat file (variable names C, X, Y, Sx, Sy, L)."""
    from scipy.io import savemat
    savemat(str(path), {k: matrices[k] for k in MAT_VARIABLES})


def write_matlab_text(matrices: dict, path, title: str = "") -> None:
    """Save the same matrices as a readable MATLAB script that can be pasted or run directly."""
    n_j, n_m = matrices["C"].shape
    head = [
        f"% MATLAB input for: {title}" if title else "% MATLAB input",
        f"% {n_j} joints, {n_m} members. Joint k = row k (1 = pin, 2 = roller, 3 = load joint),",
        "% member m = column m. Units: inches and oz.",
        "% PLACEHOLDER: the MATLAB verification step is not implemented yet.",
        f"% Sign convention: downward load is {'negative' if LOAD_SIGN < 0 else 'positive'} in L "
        "(as in calculate_forces.m).",
        f"joint_cnt = {n_j};",
        f"member_cnt = {n_m};",
        "",
    ]
    body = [_matlab_literal(k, matrices[k]) for k in ("C", "X", "Y", "Sx", "Sy", "L")]
    with open(path, "w") as f:
        f.write("\n".join(head) + "\n".join(body))


def export_inputs(coords, members, name: str, folder=None, load: float = config.TEST_LOAD) -> dict:
    """Write <name>.mat and <name>.m to results/matlab_io/ and return their paths."""
    folder = config.MATLAB_IO_DIR if folder is None else folder
    folder.mkdir(parents=True, exist_ok=True)
    mats = build_matrices(round_for_export(coords), members, load)
    mat_path, txt_path = folder / f"{name}_input.mat", folder / f"{name}_input.m"
    write_mat(mats, mat_path)
    write_matlab_text(mats, txt_path, title=name)
    return {"mat": mat_path, "text": txt_path}


# --------------------------------------------------------------------------------------
# Running MATLAB
# --------------------------------------------------------------------------------------
class MatlabError(RuntimeError):
    """MATLAB could not be run, failed, or produced no usable output."""


class MatlabNotAvailable(MatlabError):
    """No MATLAB executable was found."""


def find_matlab():
    """Path of the MATLAB launcher, or None: config.MATLAB_EXECUTABLE, then PATH, then /Applications."""
    if config.MATLAB_EXECUTABLE:
        p = Path(config.MATLAB_EXECUTABLE)
        return p if p.exists() else None
    on_path = shutil.which("matlab")
    if on_path:
        return Path(on_path)
    apps = sorted(glob.glob("/Applications/MATLAB_*.app/bin/matlab"))
    return Path(apps[-1]) if apps else None


def matlab_available() -> bool:
    """True if a MATLAB executable was found (this does not check the licence)."""
    return find_matlab() is not None


@dataclass
class MatlabOutput:
    """What run_verification.m saves for one design (MATLAB's own sign convention)."""
    member_forces: np.ndarray       # (M,) positive = compression
    reactions: np.ndarray           # (3,) Sx1, Sy1, Sy2
    member_lengths: np.ndarray      # (M,)
    buckling_forces: np.ndarray     # (M,) capacity - 1.52 oz
    governing: int                  # 1-based member that limits max_load (0 if none)
    ratio: float
    applied_load: float             # oz, the load that was in L
    max_load: float                 # oz, lower-bound failure load
    cost: float
    error_message: str = ""
    mat_path: Path = None
    log_path: Path = None


def output_paths(input_mat) -> tuple:
    """(<name>_output.mat, <name>_output.txt) belonging to a <name>_input.mat file."""
    base = str(input_mat)
    if not base.endswith("_input.mat"):
        raise ValueError(f"expected a file named <name>_input.mat, got {base}")
    return Path(base[:-len("_input.mat")] + "_output.mat"), Path(base[:-len("_input.mat")] + "_output.txt")


def read_output(input_mat) -> MatlabOutput:
    """Load the _output.mat that run_verification.m wrote next to `input_mat`."""
    mat_path, log_path = output_paths(input_mat)
    if not mat_path.exists():
        raise MatlabError(f"MATLAB produced no output for {Path(input_mat).name} (expected {mat_path})")
    d = loadmat(str(mat_path))
    message = str(d["error_message"].ravel()[0]) if d["error_message"].size else ""
    if message:
        raise MatlabError(f"MATLAB failed on {Path(input_mat).name}: {message}")
    flat = lambda k: np.asarray(d[k], dtype=float).ravel()           # noqa: E731
    return MatlabOutput(
        member_forces=flat("member_forces"), reactions=flat("reactions"),
        member_lengths=flat("member_lengths"), buckling_forces=flat("buckling_forces"),
        governing=int(flat("governing")[0]), ratio=float(flat("ratio")[0]),
        applied_load=float(flat("applied_load")[0]), max_load=float(flat("max_load")[0]),
        cost=float(flat("cost")[0]), mat_path=mat_path, log_path=log_path)


def _matlab_string(text) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def run_matlab(inputs, workdir=None, timeout: float = None, executable=None) -> dict:
    """Run matlab/truss_analysis.m on exported input files in ONE headless MATLAB session.

    `inputs` are the <name>_input.mat paths from export_inputs. Returns {input path: MatlabOutput}.
    Each design's printout is saved next to its input as <name>_output.txt, and the session log as
    matlab_run.log in `workdir`. Raises MatlabNotAvailable if MATLAB is not installed and MatlabError
    if the run fails, times out, or any design produces no usable result.
    """
    exe = Path(executable) if executable else find_matlab()
    if exe is None or not Path(exe).exists():
        raise MatlabNotAvailable("MATLAB was not found. Install it, put `matlab` on PATH, or set "
                                 "MATLAB_EXECUTABLE in config.py; or run without --matlab.")
    inputs = [Path(p).resolve() for p in inputs]
    if not inputs:
        return {}
    for p in inputs:
        if not p.exists():
            raise MatlabError(f"input file not found: {p}")
    workdir = Path(workdir) if workdir else inputs[0].parent
    workdir.mkdir(parents=True, exist_ok=True)
    list_file = workdir / "_matlab_batch_list.txt"
    list_file.write_text("\n".join(str(p) for p in inputs) + "\n")
    for p in inputs:                                   # stale results must not be mistaken for new ones
        for stale in output_paths(p):
            stale.unlink(missing_ok=True)

    command = f"addpath({_matlab_string(config.MATLAB_DIR)}); run_verification({_matlab_string(list_file)})"
    timeout = config.MATLAB_TIMEOUT if timeout is None else timeout
    try:
        proc = subprocess.run([str(exe), "-batch", command], capture_output=True, text=True,
                              timeout=timeout, cwd=str(workdir))
    except subprocess.TimeoutExpired as exc:
        raise MatlabError(f"MATLAB did not finish within {timeout:g} s") from exc
    except OSError as exc:
        raise MatlabError(f"could not start MATLAB ({exe}): {exc}") from exc
    (workdir / "matlab_run.log").write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
        raise MatlabError(f"MATLAB exited with status {proc.returncode}:\n{tail}")
    return {p: read_output(p) for p in inputs}


# --------------------------------------------------------------------------------------
# Comparing MATLAB with Python
# --------------------------------------------------------------------------------------
@dataclass
class Comparison:
    rows: list = field(default_factory=list)       # dicts: name, python, matlab, diff, ok
    rel_tol: float = config.MATLAB_REL_TOL

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows)

    def lines(self) -> list:
        out = [f"{'check':<34}{'Python':>14}{'MATLAB':>14}{'difference':>14}  result"]
        for r in self.rows:
            out.append(f"{r['name']:<34}{r['python']:>14}{r['matlab']:>14}{r['diff']:>14}  "
                       f"{'ok' if r['ok'] else 'MISMATCH'}")
        return out


def diagnose_shapes(design, output: MatlabOutput) -> list:
    """Problems with the shapes of MATLAB's outputs (empty list = fine)."""
    n_m = design.candidate.n_members
    problems = []
    for name, got, want in (("member_forces", output.member_forces.shape, (n_m,)),
                            ("reactions", output.reactions.shape, (3,)),
                            ("member_lengths", output.member_lengths.shape, (n_m,))):
        if tuple(got) != want:
            problems.append(f"{name} has shape {tuple(got)}, expected {want}")
    return problems


def diagnose(design, output: MatlabOutput, load: float = config.TEST_LOAD) -> list:
    """Human-readable hints about shape or sign-convention problems (empty list = none found)."""
    problems = diagnose_shapes(design, output)
    if problems:
        return problems
    py = np.asarray(design.evaluation.analysis.forces, dtype=float)
    mapped = matlab_forces_as_python(output, load)
    scale = max(1.0, float(np.max(np.abs(py))))
    if np.max(np.abs(py - mapped)) > config.MATLAB_REL_TOL * scale:
        raw = output.member_forces * (config.TEST_LOAD / load)
        if np.max(np.abs(py - raw)) <= config.MATLAB_REL_TOL * scale:
            problems.append("sign convention: MATLAB forces equal +Python forces, but "
                            f"MATLAB_FORCE_SIGN = {MATLAB_FORCE_SIGN:g} expects the opposite sign")
        else:
            problems.append("member forces differ from the Python solver by more than a sign "
                            "(check the matrices: joint/member order, supports, load joint)")
    return problems


def matlab_forces_as_python(output: MatlabOutput, load: float = config.TEST_LOAD) -> np.ndarray:
    """MATLAB member forces in analysis.py's convention (+ tension) and at config.TEST_LOAD."""
    return output.member_forces / MATLAB_FORCE_SIGN * (config.TEST_LOAD / load)


def compare_results(design, output: MatlabOutput, load: float = config.TEST_LOAD,
                    rel_tol: float = None) -> Comparison:
    """Compare a MATLAB result with the Python solver for the same design.

    Forces and reactions are compared after mapping the sign convention. MATLAB's max_load uses the
    lower-bound capacities, so it is compared with the Python p_max_lower.
    """
    rel_tol = config.MATLAB_REL_TOL if rel_tol is None else rel_tol
    a = design.evaluation.analysis
    scale = config.TEST_LOAD / load
    comp = Comparison(rel_tol=rel_tol)

    def add(name, py, mat, diff, ok, fmt="{:.4f}"):
        comp.rows.append({"name": name, "python": fmt.format(py) if not isinstance(py, str) else py,
                          "matlab": fmt.format(mat) if not isinstance(mat, str) else mat,
                          "diff": f"{diff:.2e}", "ok": bool(ok)})

    shape_problems = diagnose_shapes(design, output)
    if shape_problems:
        add("output shapes", "as expected", "; ".join(shape_problems), float(len(shape_problems)), False)
        return comp

    py_f = np.asarray(a.forces, dtype=float)
    mat_f = matlab_forces_as_python(output, load)
    diff = float(np.max(np.abs(py_f - mat_f)))
    add("member forces, max |diff| (oz/oz)", float(np.max(np.abs(py_f))), float(np.max(np.abs(mat_f))),
        diff, diff <= rel_tol * max(1.0, float(np.max(np.abs(py_f)))))
    py_r = np.asarray(a.reactions, dtype=float)
    mat_r = output.reactions / MATLAB_FORCE_SIGN * scale
    diff = float(np.max(np.abs(py_r - mat_r)))
    add("reactions, max |diff| (oz/oz)", float(np.max(np.abs(py_r))), float(np.max(np.abs(mat_r))),
        diff, diff <= rel_tol * max(1.0, float(np.max(np.abs(py_r)))))
    py_l = geometry.member_lengths(design.coords, design.candidate.members)
    diff = float(np.max(np.abs(py_l - output.member_lengths)))
    add("member lengths, max |diff| (in)", float(py_l.sum()), float(output.member_lengths.sum()),
        diff, diff <= 1e-6)
    diff = abs(a.cost - output.cost)
    add("cost ($)", a.cost, output.cost, diff, diff <= rel_tol * a.cost, "{:.2f}")
    diff = abs(a.p_max_lower - output.max_load)
    add("max load, lower bound (oz)", a.p_max_lower, output.max_load, diff,
        diff <= rel_tol * a.p_max_lower, "{:.3f}")
    # limiting member: a mismatch only counts if the two members are not a near-tie
    py_gov, mat_gov = a.governing_lower + 1, output.governing
    same = py_gov == mat_gov
    if not same and 0 < mat_gov <= len(a.forces):
        f = -a.forces
        low = np.maximum(a.capacities - config.BUCKLING_UNCERTAINTY, 0.0)
        r_py, r_mat = low[py_gov - 1] / f[py_gov - 1], low[mat_gov - 1] / f[mat_gov - 1]
        same = abs(r_py - r_mat) <= rel_tol * r_py
    add("limiting member (m)", f"m{py_gov}", f"m{mat_gov}", float(py_gov != mat_gov), same)
    return comp

