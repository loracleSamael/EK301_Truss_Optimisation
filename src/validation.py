# validation.py
# One-off check to run before trusting the optimiser's numbers: do the Python solver and
# matlab/truss_analysis.m agree on many different, legal designs?
#
#   random_valid_designs(n)   n varied designs that satisfy every project rule (not optimised)
#   validate(designs, folder) export them, run MATLAB once, compare each with the Python solver
#
# scripts/validate_against_matlab.py is the command-line wrapper (exit 0 pass, 1 fail, 2 no MATLAB).

import random
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import config
from src import geometry_opt, matlab_io, optimiser, scoring, topology

SAMPLES_PER_VISIT = 24          # random geometries tried for a candidate on each visit
MAX_ROUNDS = 6                  # passes over the candidate list before giving up


# --------------------------------------------------------------------------------------
# Random legal designs
# --------------------------------------------------------------------------------------
def random_valid_designs(n: int = 50, seed: int = config.RANDOM_SEED, per_candidate: int = 3,
                         candidates=None) -> list:
    """n varied, legal (all rules, cuttable from the sticks) designs, rounded to the export grid.

    They come from the screening sampler and repair step, not from the optimiser, so they cover
    many topologies and geometries rather than just the winners. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    pool = list(candidates if candidates is not None else topology.enumerate_candidates())
    rng.shuffle(pool)
    designs, taken = [], Counter()
    for round_no in range(MAX_ROUNDS):
        for cand in pool:
            if len(designs) >= n:
                return designs
            if taken[cand] >= per_candidate:
                continue
            crng = optimiser._rng_for(cand, seed * 1000 + round_no)
            orders = optimiser.construction_orders(cand, crng)
            drawn = [c for c in (optimiser.sample_geometry(cand, orders, crng)
                                 for _ in range(SAMPLES_PER_VISIT)) if c is not None] if orders else []
            if not drawn:
                continue
            batch = optimiser._spread_load_joint(np.stack(drawn), crng)
            for coords in optimiser.repair_batch(cand, batch):
                coords = geometry_opt.round_coords(coords)
                ev = scoring.evaluate(coords, cand.members)
                if not ev.valid:
                    continue
                designs.append(geometry_opt.Design(
                    cand, coords, ev, scoring.OBJ_MAX_LOAD, scoring.score(ev, scoring.OBJ_MAX_LOAD)))
                taken[cand] += 1
                if taken[cand] >= per_candidate or len(designs) >= n:
                    break
    return designs


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
@dataclass
class ValidationReport:
    names: list = field(default_factory=list)
    designs: list = field(default_factory=list)
    comparisons: dict = field(default_factory=dict)       # name -> matlab_io.Comparison
    diagnostics: dict = field(default_factory=dict)       # name -> [hints]
    seconds: float = 0.0
    folder: object = None

    @property
    def failures(self) -> list:
        return [n for n in self.names if not self.comparisons[n].ok]

    @property
    def passed(self) -> bool:
        return bool(self.names) and not self.failures

    def worst_case(self) -> dict:
        """check name -> largest absolute difference seen over all designs."""
        worst = {}
        for cmp in self.comparisons.values():
            for r in cmp.rows:
                try:
                    d = float(r["diff"])
                except ValueError:
                    continue
                worst[r["name"]] = max(worst.get(r["name"], 0.0), d)
        return worst

    def lines(self) -> list:
        by_j = Counter(d.candidate.n_joints for d in self.designs)
        n_topologies = len({d.candidate for d in self.designs})
        lines = [f"Validated {len(self.names)} designs on {n_topologies} different topologies "
                 f"({', '.join(f'J={j}: {c}' for j, c in sorted(by_j.items()))}) "
                 f"against matlab/truss_analysis.m in {self.seconds:.1f} s.",
                 f"Tolerance: relative {config.MATLAB_REL_TOL:g}. Worst absolute differences:"]
        worst = self.worst_case()
        for name, value in worst.items():
            lines.append(f"  {name:<36}{value:>12.3e}")
        if self.failures:
            lines.append(f"{len(self.failures)} design(s) disagree:")
            for n in self.failures:
                lines.append(f"  {n}:")
                lines += [f"    {ln}" for ln in self.comparisons[n].lines()]
                lines += [f"    hint: {h}" for h in self.diagnostics.get(n, [])]
        lines.append("RESULT: " + ("PASS - Python and MATLAB agree on every design."
                                   if self.passed else "FAIL"))
        return lines


def validate(designs, folder, timeout: float = None) -> ValidationReport:
    """Export `designs`, run them through MATLAB in one session and compare each with Python.

    Raises matlab_io.MatlabNotAvailable / MatlabError if MATLAB cannot be run.
    """
    t0 = time.perf_counter()
    report = ValidationReport(folder=folder)
    inputs = {}
    for k, design in enumerate(designs):
        name = f"val_{k:03d}"
        paths = matlab_io.export_inputs(design.coords, design.candidate.members, name, folder)
        inputs[name] = paths["mat"]
        report.names.append(name)
        report.designs.append(design)
    outputs = matlab_io.run_matlab(list(inputs.values()), folder, timeout=timeout)
    for name, design in zip(report.names, report.designs):
        out = outputs[inputs[name].resolve()]
        report.comparisons[name] = matlab_io.compare_results(design, out)
        report.diagnostics[name] = matlab_io.diagnose(design, out)
    report.seconds = time.perf_counter() - t0
    return report
