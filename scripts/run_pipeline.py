# run_pipeline.py
# Entry point that runs the whole project:
#   python scripts/run_pipeline.py            full run (a few minutes)
#   python scripts/run_pipeline.py --quick    small run for a quick look or a smoke test
#   python scripts/run_pipeline.py --matlab   also verify the finalists with MATLAB
#
# Exit status: 0 success, 1 no legal design found, 3 --matlab requested but MATLAB was missing,
# failed to run, or disagreed with the Python solver.
#
# It enumerates the candidate trusses, screens them, optimises the geometry for the two objectives
# (maximum load, maximum load / cost), checks sensitivity to the buckling-fit uncertainty and writes,
# for each of the two trusses, a drawing (PNG), a text report, a JSON file and the MATLAB input files.
# Results go to results/ unless --out-dir is given.

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate, analyse and optimise 2D bridge trusses (maximum load, and maximum "
                    "load / cost).")
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)),
                   help="worker processes (default: min(8, CPU count))")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                   help=f"random seed (default {config.RANDOM_SEED}); the result is reproducible")
    p.add_argument("--samples", type=int, default=config.SCREEN_SAMPLES_PER_CANDIDATE,
                   help="screening samples per candidate")
    p.add_argument("--top-k", type=int, default=config.TOP_K,
                   help="candidates per objective that get the SLSQP optimisation")
    p.add_argument("--starts", type=int, default=config.SLSQP_STARTS,
                   help="SLSQP starts per candidate")
    p.add_argument("--quick", action="store_true",
                   help="small run: fewer samples, starts and finalists")
    p.add_argument("--matlab", action="store_true",
                   help="verify the finalists with MATLAB (matlab/truss_analysis.m); exit status 3 "
                        "if MATLAB is missing, fails to run, or disagrees with the Python solver")
    p.add_argument("--no-regret", action="store_true",
                   help="skip the regret measurement in the sensitivity check")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write results here instead of results/")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    options = pipeline.Options(
        workers=args.workers, seed=args.seed, n_samples=args.samples, top_k=args.top_k,
        n_starts=args.starts, quick=args.quick, matlab=args.matlab, out_dir=args.out_dir,
        regret=not args.no_regret)
    try:
        result = pipeline.run(options)
    except pipeline.PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.matlab_failed:
        print(f"MATLAB verification: {result.matlab_status}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":      # required: the worker processes re-import this module
    sys.exit(main())
