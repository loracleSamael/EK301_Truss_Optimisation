# pipeline.py
# The whole project in one call: enumerate topologies, screen them, optimise the geometry for both
# objectives, check sensitivity to the buckling-fit uncertainty, pick the two trusses and write every
# output. scripts/run_pipeline.py is a thin command-line wrapper around run().
#
# Stages
#   1. enumerate  all (topology, roles) candidates for J_MIN..J_MAX (exhaustive, deterministic)
#   2. screen     random + repaired geometries per candidate (optimiser.py)
#   3. optimise   SLSQP per candidate and objective, stick assignments included (geometry_opt.py)
#   4. sensitivity re-score the best few designs with the lower-bound buckling fit (sensitivity.py)
#   5. select     best design per objective
#   6. write      drawing (PNG), text report, JSON and MATLAB input files (visualise.py)
#
#   7. verify     (--matlab) run the two winners through matlab/truss_analysis.m in one MATLAB
#                 session, compare with the Python solver, and rewrite their outputs with the
#                 MATLAB-confirmed forces and the verification result
#
# Only the two winning trusses produce files: nothing is written for runner-up designs.

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
from src import (geometry, geometry_opt, matlab_io, optimiser, scoring, sensitivity, topology,
                 visualise)

TRUSS_NAMES = {
    scoring.OBJ_MAX_LOAD: ("truss1_max_load", "Truss 1: maximum failure load (cost ignored)"),
    scoring.OBJ_MAX_LOAD_PER_COST: ("truss2_max_load_per_cost",
                                    "Truss 2: maximum failure load / cost"),
}

# --quick: all candidates but far fewer samples, starts and top candidates (about 20 s)
QUICK = {"n_samples": 40, "top_k": 6, "n_starts": 2}


class PipelineError(RuntimeError):
    """The pipeline could not produce a legal design."""


@dataclass
class Options:
    workers: int = max(1, min(8, os.cpu_count() or 1))
    seed: int = config.RANDOM_SEED
    n_samples: int = config.SCREEN_SAMPLES_PER_CANDIDATE
    top_k: int = config.TOP_K
    n_starts: int = config.SLSQP_STARTS
    quick: bool = False
    matlab: bool = config.USE_MATLAB     # verify the two winners with MATLAB
    out_dir: Path = None                 # None: the folders from config.py
    regret: bool = True                  # also measure the regret in the sensitivity check
    quiet: bool = False

    def resolved(self) -> "Options":
        """A copy with the --quick settings applied."""
        if not self.quick:
            return self
        return Options(**{**self.__dict__, **QUICK})


@dataclass
class PipelineResult:
    options: Options
    designs: dict = field(default_factory=dict)          # objective -> winning Design
    ranked: dict = field(default_factory=dict)           # objective -> all legal Designs, best first
    sensitivity: dict = field(default_factory=dict)      # objective -> SensitivityReport
    paths: dict = field(default_factory=dict)            # objective -> {kind: Path}
    names: dict = field(default_factory=dict)            # objective -> file stem
    verification: dict = field(default_factory=dict)     # objective -> matlab_io.Comparison (winner)
    matlab_status: str = "not requested"                 # "passed", "FAILED", "skipped: ...", ...
    stats: dict = field(default_factory=dict)            # counts and timings
    warnings: list = field(default_factory=list)
    summary_path: Path = None

    @property
    def matlab_failed(self) -> bool:
        """True if MATLAB verification was requested but did not run cleanly or found a mismatch."""
        return self.matlab_status.startswith(("FAILED", "skipped"))


def submission_file(objective: str) -> str:
    """File name of a winner's MATLAB input, e.g. TrussDesign1_MaryJoeBob_A1.mat."""
    return matlab_io.submission_filename(list(scoring.OBJECTIVES).index(objective) + 1)


def _dirs(options: Options) -> dict:
    if options.out_dir is None:
        return {"figures_dir": config.FIGURES_DIR, "designs_dir": config.DESIGNS_DIR,
                "matlab_dir": config.MATLAB_IO_DIR}
    root = Path(options.out_dir)
    return {"figures_dir": root / "figures", "designs_dir": root / "designs",
            "matlab_dir": root / "matlab_io"}


def _progress(label: str, log):
    """Progress callback that logs at every 25 %."""
    state = {"next": 25}

    def report(done: int, total: int):
        pct = 100 * done / total
        if pct >= state["next"]:
            log(f"  {label}: {done}/{total} ({int(pct)}%)")
            while state["next"] <= pct:
                state["next"] += 25
    return report


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
def run(options: Options = None, log=print) -> PipelineResult:
    options = (options or Options()).resolved()
    if options.quiet:
        def log(_msg):                      # noqa: F811 - silence all progress output
            return None
    result = PipelineResult(options)
    stats = result.stats
    t_start = time.perf_counter()

    # 1. enumerate -----------------------------------------------------------------------
    t = time.perf_counter()
    candidates = topology.enumerate_candidates()
    stats["candidates"] = len(candidates)
    stats["t_enumerate"] = time.perf_counter() - t
    log(f"Enumerated {len(candidates)} candidates "
        f"(J = {config.J_MIN}..{config.J_MAX}, sticks {' + '.join(f'{x:g}' for x in config.STOCK_LENGTHS)} in).")

    # 2. screen --------------------------------------------------------------------------
    t = time.perf_counter()
    log(f"Screening ({options.n_samples} samples per candidate, {options.workers} worker(s), "
        f"seed {options.seed}) ...")
    screens = optimiser.screen_all(candidates, n_samples=options.n_samples, top_k=options.top_k,
                                   progress=_progress("screened", log), workers=options.workers,
                                   seed=options.seed)
    stats["t_screen"] = time.perf_counter() - t
    stats["screen_feasible"] = sum(1 for s in screens["all"] if s.n_feasible)
    log(f"  {stats['screen_feasible']} of {len(candidates)} candidates have a legal sample.")
    if stats["screen_feasible"] == 0:
        raise PipelineError("Screening found no legal design. Try more samples (--samples).")

    # 3. optimise + 4. sensitivity -------------------------------------------------------
    for objective in scoring.OBJECTIVES:
        t = time.perf_counter()
        log(f"Optimising geometry for '{objective}' (top {len(screens[objective])} candidates, "
            f"{options.n_starts} starts) ...")
        ranked = geometry_opt.optimise_all(
            screens[objective], objective, n_starts=options.n_starts, workers=options.workers,
            progress=_progress("optimised", log))
        stats[f"t_optimise_{objective}"] = time.perf_counter() - t
        if not ranked:
            raise PipelineError(f"No legal design found for '{objective}'. "
                                "Try more samples (--samples) or a larger --top-k.")
        result.ranked[objective] = ranked
        result.designs[objective] = ranked[0]
        log(f"  {len(ranked)} legal designs; best P_max {ranked[0].p_max:.3f} oz, "
            f"cost {ranked[0].cost:.2f}, P/cost {ranked[0].p_over_cost:.4f}")

        t = time.perf_counter()
        report = sensitivity.sensitivity_check(ranked, objective, config.N_FINALISTS,
                                               with_regret=options.regret)
        stats[f"t_sensitivity_{objective}"] = time.perf_counter() - t
        result.sensitivity[objective] = report
        if report.winner_changed:
            result.warnings.append(
                f"{objective}: the winner changes when the buckling capacities are lowered by "
                f"{config.BUCKLING_UNCERTAINTY:g} oz (see the sensitivity section).")

    # 5/6. select and write --------------------------------------------------------------
    dirs = _dirs(options)
    for objective, design in result.designs.items():
        name, title = TRUSS_NAMES[objective]
        result.names[objective] = name
        result.paths[objective] = visualise.write_outputs(
            design, name, title, submission_file=submission_file(objective), **dirs)
    if options.matlab:
        t = time.perf_counter()
        _verify_with_matlab(result, dirs, log)
        stats["t_matlab"] = time.perf_counter() - t

    stats["t_total"] = time.perf_counter() - t_start
    text = format_summary(result)
    result.summary_path = dirs["designs_dir"] / "run_summary.txt"
    result.summary_path.write_text(text)
    log("")
    log(text)
    return result


# --------------------------------------------------------------------------------------
# MATLAB verification
# --------------------------------------------------------------------------------------
def _verify_with_matlab(result: PipelineResult, dirs: dict, log) -> None:
    """Run the two winners through MATLAB and update their outputs. Never raises: a MATLAB problem
    becomes a warning and a status, so the optimisation is not lost. Only the winners' files are
    written (their inputs already exist; MATLAB adds one output pair for each)."""
    if not matlab_io.matlab_available():
        result.matlab_status = "skipped: MATLAB not found"
        result.warnings.append("MATLAB verification skipped: MATLAB was not found (set "
                               "MATLAB_EXECUTABLE in config.py or put `matlab` on PATH).")
        log("MATLAB verification skipped: MATLAB not found.")
        return

    inputs = {objective: result.paths[objective]["matlab_mat"] for objective in result.designs}
    log(f"Verifying the {len(inputs)} winning trusses with MATLAB ...")
    try:
        outputs = matlab_io.run_matlab(list(inputs.values()), dirs["matlab_dir"])
    except matlab_io.MatlabError as exc:
        result.matlab_status = f"FAILED to run: {exc}"
        result.warnings.append(f"MATLAB verification could not be completed: {exc}")
        log(f"MATLAB verification failed to run: {exc}")
        return

    for objective, design in result.designs.items():
        out = outputs[Path(inputs[objective]).resolve()]
        comparison = matlab_io.compare_results(design, out)
        result.verification[objective] = comparison
        name, title = TRUSS_NAMES[objective]
        forces = matlab_io.matlab_forces_as_python(out) if comparison.ok else None
        result.paths[objective] = visualise.write_outputs(
            design, name, title, forces=forces, verification=comparison,
            submission_file=submission_file(objective), **dirs)
        if not comparison.ok:
            problems = "; ".join(matlab_io.diagnose(design, out)) or "see the comparison table"
            result.warnings.append(f"MATLAB verification FAILED for {name}: {problems}")
    ok = [c.ok for c in result.verification.values()]
    result.matlab_status = "passed" if all(ok) else "FAILED: mismatch"
    log(f"  MATLAB verification {result.matlab_status}: {sum(ok)} of {len(ok)} trusses agree.")

# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------
def _design_block(objective: str, design, name: str, paths: dict) -> list:
    lengths = geometry.member_lengths(design.coords, design.candidate.members)
    a = design.evaluation.analysis
    lines = [
        f"{name}  ({objective})",
        f"  P_max = {a.p_max:.2f} +/- {a.p_max_uncertainty:.2f} oz   (lower bound {a.p_max_lower:.2f} oz)",
        f"  cost = ${design.cost:.2f}   P_max/cost = {design.p_over_cost:.4f} oz/$",
        f"  {design.candidate.n_joints} joints, {design.candidate.n_members} members, "
        f"{lengths.sum():.2f} in of member length",
        f"  load joint at x = {design.coords[config.LOAD_INDEX, 0]:.2f} in, "
        f"height {design.coords[config.LOAD_INDEX, 1]:.2f} in; span "
        f"{geometry.span(design.coords):.2f} in",
        "  cutting plan:",
    ]
    for entry in visualise.cutting_plan(design):
        members = ", ".join(f"m{m + 1}" for m in entry["members"]) or "-"
        lines.append(f"    stick {entry['stick']}: {members}  "
                     f"(used {entry['used']:.2f}, left {entry['left']:.2f} in)")
    lines.append("  files:")
    labels = {"figure": "drawing", "report": "report", "json": "json",
              "matlab_mat": "MATLAB input (.mat)", "matlab_text": "MATLAB input (.m)"}
    lines += [f"    {labels.get(k, k)}: {v}" for k, v in paths.items()]
    return lines


def format_summary(result: PipelineResult) -> str:
    o, st = result.options, result.stats
    lines = ["=" * 78, "Truss optimiser: run summary", "=" * 78,
             f"candidates {st['candidates']} (J = {config.J_MIN}..{config.J_MAX}); "
             f"{st['screen_feasible']} had a legal screened sample; seed {o.seed}"
             + ("; QUICK run" if o.quick else ""),
             f"settings: {o.n_samples} samples, top-k {o.top_k}, {o.n_starts} SLSQP starts, "
             f"{o.workers} worker(s)",
             f"time: enumerate {st['t_enumerate']:.1f}s, screen {st['t_screen']:.1f}s, optimise "
             + ", ".join(f"{st[f't_optimise_{obj}']:.1f}s" for obj in scoring.OBJECTIVES)
             + f", total {st['t_total']:.1f}s", ""]
    for objective in scoring.OBJECTIVES:
        lines += _design_block(objective, result.designs[objective], result.names[objective],
                               result.paths[objective])
        lines += [""] + result.sensitivity[objective].summary_lines() + [""]
    lines.append("Rules enforced: span 30-33 in, members 8.5-14 in, load joint 0.5-1.5 in above the "
                 "supports, strip rule, no crossings, nothing below the supports, and every member cut "
                 "from one of the sticks.")
    if o.matlab:
        lines += ["", f"MATLAB verification (matlab/truss_analysis.m): {result.matlab_status}"]
        for objective in scoring.OBJECTIVES:
            cmp = result.verification.get(objective)
            if cmp is None:
                continue
            lines.append(f"  {result.names[objective]}: {'PASSED' if cmp.ok else 'FAILED'}")
            lines += [f"    {line}" for line in cmp.lines()]
    if result.warnings:
        lines += ["", "WARNINGS"] + [f"  - {w}" for w in result.warnings]
    return "\n".join(lines) + "\n"
