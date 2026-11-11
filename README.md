# Truss Optimiser

A pipeline that generates, analyses and optimises 2D pin-jointed bridge trusses (EK301). It finds
two trusses:

1. **Truss 1, maximum load:** the highest failure load P_max, cost ignored.
2. **Truss 2, maximum load / cost:** the highest P_max per unit cost (joint = $10, member = $1 per inch).

Every candidate is a determinate, fully triangulated, planar truss (M = 2J - 3) on a pin and a roller.
Fast analysis runs in Python during the search. The provided MATLAB analysis
(`matlab/truss_analysis.m`) can then verify the finalists. Each of the two trusses is written out as a
to-scale drawing, a text report, a JSON file and the MATLAB input files.

## Results (seed 301, full settings)

| | Truss 1: max load | Truss 2: max load / cost |
|---|---|---|
| P_max | 62.01 +/- 2.34 oz | 61.59 +/- 2.05 oz |
| Lower-bound P_max (capacities - 1.52 oz) | 57.15 oz | 57.47 oz |
| Cost | $184.28 | $179.36 |
| P_max / cost | 0.3365 oz/$ | **0.3434 oz/$** |
| Joints / members | 7 / 11 | 7 / 11 |
| Member length used | 114.28 in | 109.36 in |
| Span, load joint | 30.00 in, x = 8.50 in, height 0.50 in | 30.00 in, x = 8.92 in, height 0.50 in |

Joint coordinates in inches (joint numbers are 1-based; 1 = pin, 2 = roller, 3 = load joint):

| Joint | Truss 1 | Truss 2 |
|---|---|---|
| 1 | (0.00, 0.00) | (0.00, 0.00) |
| 2 | (30.00, 0.00) | (30.00, 0.00) |
| 3 | (8.50, 0.50) | (8.92, 0.50) |
| 4 | (10.26, 10.63) | (10.73, 9.27) |
| 5 | (18.29, 0.01) | (18.56, 0.00) |
| 6 | (1.49, 8.38) | (2.28, 8.20) |
| 7 | (21.93, 8.31) | (21.97, 7.80) |

Cutting plans (every member from one stick; members numbered m1, m2, ...):

| Stick | Truss 1 | Truss 2 |
|---|---|---|
| A (48 in) | m1, m2, m5, m6, m9 (46.16 in used) | m1, m2, m7, m9, m11 (44.64 in used) |
| B (48 in) | m3, m8, m10, m11 (45.99 in used) | m4, m6, m8, m10 (44.32 in used) |
| C (24 in) | m4, m7 (22.13 in used) | m3, m5 (20.39 in used) |

Notes on these numbers:

- **The stock rule removes 13-member trusses.** A 48 in stick holds at most 5 members (6 x 8.5 = 51 in) and
  the 24 in stick at most 2, so at most 12 members can be cut, and M = 2J - 3 is odd. That limits designs
  to J <= 7. Without the stock rule a 13-member truss reached 63.6 oz; the J = 7 designs above are
  unaffected by the rule.
- **The buckling-fit uncertainty is reported, not optimised.** The search uses the nominal fit. Lowering
  every capacity by 1.52 oz reduces P_max by about 6 to 9 % but does not change the winner or the order of
  the top five. A design re-optimised against the lower bound would gain 3.6 % (Truss 1) and 2.8 %
  (Truss 2) in lower-bound P_max, at a cost of under 1 % of nominal P_max. The pipeline prints this as the
  "regret" of each winner.
- **Verified against MATLAB.** Both winners and their four runner-ups each (10 designs) agree with
  `matlab/truss_analysis.m`: member forces, reactions, lengths and cost to about 1e-15, and the
  lower-bound max load to 9e-4 oz (the MATLAB file uses 3391.3 where Python uses 3391.349). The same holds
  for 50 random legal designs (`scripts/validate_against_matlab.py`).
- **Local optimum.** SLSQP is a local method. The result is the best over all candidate topologies and 8
  starts each, and optimising all 147 candidates instead of the top 40 gave the same winners. It is not a
  proof of global optimality.

## Setup

Python 3.13 was used. From the repository root:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt` (numpy, scipy, matplotlib, networkx, pytest). MATLAB is
optional. It was tested with R2025a. `matlab -batch` needs R2019a or newer and `run_verification.m` uses
`readlines`, so R2020b or newer is required.

## Usage

```
.venv/bin/python scripts/run_pipeline.py              # full run
.venv/bin/python scripts/run_pipeline.py --quick      # small run for a quick look
.venv/bin/python scripts/run_pipeline.py --matlab     # also verify the finalists with MATLAB
.venv/bin/python scripts/validate_against_matlab.py   # compare Python and MATLAB on 50 random designs
.venv/bin/python -m pytest -q                         # run the tests
```

Options of `run_pipeline.py`:

| Flag | Meaning |
|---|---|
| `--quick` | fewer samples (40), starts (2) and finalists (6): about 17 s, same winners here |
| `--matlab` | run the 5 best designs of each objective through MATLAB in one session, compare, and show MATLAB-verified forces in the winners' outputs |
| `--workers N` | worker processes (default: min(8, CPU count)) |
| `--seed N` | random seed (default 301); a run is reproducible for a given seed |
| `--samples N` | screening samples per candidate (default 200) |
| `--top-k N` | candidates per objective that get the SLSQP stage (default 40) |
| `--starts N` | SLSQP starts per candidate (default 8) |
| `--no-regret` | skip the regret measurement in the sensitivity check |
| `--out-dir DIR` | write the results to DIR instead of `results/` |

Exit status of `run_pipeline.py`: 0 success; 1 no legal design found; 3 `--matlab` was requested but MATLAB
is missing, failed to run, or disagreed with the Python solver. `validate_against_matlab.py` exits 0 on
pass, 1 on a disagreement, 2 if MATLAB is unavailable.

### Output files (in `results/`, git-ignored)

For each truss (`truss1_max_load`, `truss2_max_load_per_cost`):

| File | Content |
|---|---|
| `figures/<name>.png` | to-scale drawing: member lengths and forces (red = compression, blue = tension), joint coordinates, supports, load arrow with the failure load, span, member table, cutting plan |
| `designs/<name>.txt` | the same information as text, plus reactions and the MATLAB verification block |
| `designs/<name>.json` | machine-readable copy (0-based indices) |
| `matlab_io/<name>_input.mat` and `.m` | the six MATLAB inputs (`C, X, Y, Sx, Sy, L`) |
| `matlab_io/<name>_output.mat` and `.txt` | MATLAB's results and printout (with `--matlab`) |
| `designs/run_summary.txt` | the run summary that is also printed |

Runner-up designs sent to MATLAB are written as `<name>_finalist<k>_input.mat` etc., and the MATLAB
session log is `matlab_io/matlab_run.log`.

## How it works

1. **Enumerate** (`src/topology.py`): every planar truss built by Henneberg type-I growth for J = 4..7
   (a new joint attached to two existing ones), de-duplicated by isomorphism, with the pin, roller and
   load joint assigned in every distinct way (147 candidates).
2. **Screen** (`src/optimiser.py`): random planar geometries per candidate, repaired by a batched
   gradient descent towards the rules, plus hand-built Warren, Pratt and Howe layouts. The best geometry
   per candidate and objective is kept.
3. **Optimise** (`src/geometry_opt.py`): SLSQP on the continuous geometry for the best candidates. For a
   fixed truss the failure load is a minimum over compression members, written as: maximise t subject to
   `capacity_i(L_i) - t * max(0, -F_i) >= 0`. The stick limits are linear constraints for a chosen
   member-to-stick assignment (`src/packing.py` supplies the assignments), and each result is re-packed
   and re-solved. Final designs are rounded to 0.01 in and polished on that grid.
4. **Sensitivity** (`src/sensitivity.py`): the finalists are re-scored with capacities lowered by 1.52 oz.
5. **Write** (`src/visualise.py`, `src/matlab_io.py`): drawing, report, JSON and MATLAB inputs.
6. **Verify** (`--matlab`): `matlab -batch` runs `matlab/run_verification.m`, which calls
   `truss_analysis.m` for each design. `src/matlab_io.py` compares the results with the Python solver.

## Rules and assumptions

All constants are in `config.py`. Lengths are in inches and forces in ounces.

- Pin at (0, 0), roller at (S, 0) with S from 30 to 33 in. Supports are on the same level and no joint or
  member is below it.
- Member length from 8.5 to 14 in. No crossing members.
- The load joint (joint 3) is 0.5 to 1.5 in above the supports. Its x position is not constrained.
- Strip rule: no joint within +/- 0.5 in of the load joint's x while below it. The supports count as
  joints. Members may pass underneath. The 0.5 in is a conservative value agreed with the user.
- Material: two 48 in sticks and one 24 in stick. Every member is cut from a single stick (no splicing)
  and uses exactly its joint-to-joint length (no allowance, no kerf). Leftovers are not reused.
- Cost = $10 per joint + $1 per inch of member length used (not stock bought).
- Failure is buckling of compression members only: capacity = 3391.349 * L^-2.012 oz. Tension members
  do not fail. The fit has an uncertainty of +/- 1.52 oz per member, reported but not optimised.
- Only trusses reachable by Henneberg type-I growth are considered. A rigid planar truss that needs a
  type-II (edge split) step is not generated.
- Joint index convention in the code: 0 = pin, 1 = roller, 2 = load joint, 3+ = free joints. Reports and
  MATLAB files number joints and members from 1; the JSON files use the code's 0-based indices.

## MATLAB

- `matlab/calculate_forces.m` is the team's original script and is untouched.
- `matlab/truss_analysis.m` is that script's logic as a function
  `[member_forces, reactions, info] = truss_analysis(C, Sx, Sy, X, Y, L)`. Changes are marked `CHANGED:` in
  the file: inputs come from the arguments; outputs are returned in addition to the printout; the max-load
  loop considers compression members only (the original could pick a tension member as the limit); cost
  is $10 per joint + $1 per inch; `member_lengths` and `buckling_forces` are column vectors.
- Kept as in the original: the `- 1.52 oz` in the capacity (so MATLAB reports the lower-bound max load,
  which the Python side compares with its own lower bound) and the sign convention (downward load
  negative in `L`, so MATLAB's positive force is compression and Python's is tension).
- `matlab/run_verification.m` is the batch driver that the Python code calls.
- The MATLAB launcher is found on `PATH` or in `/Applications/MATLAB_*.app/bin/matlab`. To use another,
  set `MATLAB_EXECUTABLE` in `config.py`.

## Repository layout

```
config.py                  all constants and settings
matlab/                    calculate_forces.m (original), truss_analysis.m, run_verification.m
scripts/run_pipeline.py    the pipeline command line
scripts/validate_against_matlab.py   Python-vs-MATLAB check on random designs
src/analysis.py            equilibrium solver, P_max, uncertainty, cost
src/geometry.py            coordinates, lengths, crossings, strip rule, mirror form
src/packing.py             stock-stick packing (exact search) and cutting plans
src/scoring.py             constraint violations and objective scores
src/topology.py            exhaustive topology enumeration
src/optimiser.py           screening stage
src/geometry_opt.py        SLSQP stage, rounding and grid polish
src/sensitivity.py         sensitivity to the buckling-fit uncertainty
src/matlab_io.py           MATLAB inputs, running MATLAB, comparison
src/validation.py          random legal designs and the Python-vs-MATLAB validation
src/pipeline.py            all stages in one call
src/visualise.py           drawing, report, JSON
tests/                     215 tests
results/                   generated output (git-ignored)
```

## Run times (8 CPUs)

| Run | Time |
|---|---|
| Full run, `--matlab` included | about 95 s (screening 42 s, SLSQP 23 s + 22 s, MATLAB a few seconds) |
| `--quick` | about 17 s |
| `scripts/validate_against_matlab.py` | about 13 s |
| Test suite (includes live MATLAB tests, skipped without MATLAB) | about 100 s |

On macOS the worker processes re-import the main script, so any script that calls the pipeline with
`--workers` above 1 must be guarded by `if __name__ == "__main__":` (the provided scripts are).
