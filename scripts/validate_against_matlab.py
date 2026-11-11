# validate_against_matlab.py
# One-off check to run before trusting the optimiser:
#   python scripts/validate_against_matlab.py            about 50 random legal designs
#
# Generates random legal designs (all project rules, cuttable from the sticks), analyses each with
# the Python solver and with matlab/truss_analysis.m (one MATLAB session), and reports member force,
# reaction, length, cost and max-load differences. It flags shape and sign-convention problems.
#
# Exit status: 0 = pass, 1 = the two disagree (or no designs could be generated), 2 = MATLAB is not
# available or failed to run.

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import matlab_io, validation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare the Python solver with matlab/truss_analysis.m "
                                            "on random legal designs.")
    p.add_argument("-n", "--designs", type=int, default=50, help="number of designs (default 50)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="random seed")
    p.add_argument("--out-dir", type=Path, default=config.MATLAB_IO_DIR / "validation",
                   help="folder for the exported inputs and MATLAB outputs")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not matlab_io.matlab_available():
        print("MATLAB was not found, so nothing can be compared. Install it, put `matlab` on PATH, "
              "or set MATLAB_EXECUTABLE in config.py.", file=sys.stderr)
        return 2
    print(f"Generating {args.designs} random legal designs (seed {args.seed}) ...")
    designs = validation.random_valid_designs(args.designs, args.seed)
    if not designs:
        print("error: could not generate any legal design.", file=sys.stderr)
        return 1
    if len(designs) < args.designs:
        print(f"warning: only {len(designs)} of {args.designs} designs could be generated.")
    print(f"Running {len(designs)} designs through MATLAB (one session, may take a minute) ...")
    try:
        report = validation.validate(designs, args.out_dir)
    except matlab_io.MatlabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\n".join(report.lines()))
    print(f"Files: {args.out_dir}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
