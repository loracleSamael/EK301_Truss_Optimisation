# config.py
# Single source of truth for all problem constants and settings. Nothing else should hard-code these.
#
# Units: lengths in inches, forces/loads in ounces (oz).

import math
from pathlib import Path

# --------------------------------------------------------------------------------------
# Geometry limits
# --------------------------------------------------------------------------------------
SPAN_MIN = 30.0             # in, pin-to-roller distance (lower bound)
SPAN_MAX = 33.0             # in, pin-to-roller distance (upper bound)
MEMBER_LENGTH_MIN = 8.5     # in
MEMBER_LENGTH_MAX = 14.0    # in
LOAD_HEIGHT_MIN = 0.5       # in, load joint height above the supports (lower bound)
LOAD_HEIGHT_MAX = 1.5       # in, load joint height above the supports (upper bound)
                            # The load joint's x position is unconstrained along the span.
STRIP_TOLERANCE = 0.5       # in, horizontal half-width of the vertical strip under the load joint
                            # in which no JOINT may sit (conservative value, agreed with user).
                            # Members are allowed to pass underneath the load joint.
COORD_ROUNDING = 0.01       # in, coordinates are rounded to this before final scoring/export

# --------------------------------------------------------------------------------------
# Material budget and cost model
# --------------------------------------------------------------------------------------
# The material is three sticks. Every member is cut from ONE stick (no splicing) and uses exactly its
# joint-to-joint length (no allowance, no kerf). Cost is based on member length used, not stock.
STOCK_LENGTHS = (48.0, 48.0, 24.0)   # in
MATERIAL_BUDGET = float(sum(STOCK_LENGTHS))   # in, total (120)
STOCK_MARGIN = 0.1          # in, safety margin per stick used by the geometry optimiser so that
                            # rounding coordinates to COORD_ROUNDING cannot overflow a stick
PACKING_ASSIGNMENTS_PER_START = 3   # member-to-stick assignments tried per optimiser start
MAX_EXACT_PACKING_MEMBERS = 12      # exact packing search up to this many members
COST_PER_JOINT = 10.0
COST_PER_LENGTH = 1.0       # per inch of member

# --------------------------------------------------------------------------------------
# Buckling fit: F = BUCKLING_A * L ** BUCKLING_B  (F in oz, L in in)
# --------------------------------------------------------------------------------------
BUCKLING_A = 3391.349
BUCKLING_B = -2.012
# The fit has an uncertainty of +/- BUCKLING_UNCERTAINTY oz on each member's capacity. The optimiser
# uses the NOMINAL fit above (no offset). The uncertainty is reported next to P_max, and a lower
# bound P_max uses (capacity - uncertainty). matlab/truss_analysis.m also subtracts it.
BUCKLING_UNCERTAINTY = 1.52   # oz

# --------------------------------------------------------------------------------------
# Joint index convention
# --------------------------------------------------------------------------------------
PIN_INDEX = 0               # pinned support (reacts in x and y)
ROLLER_INDEX = 1            # roller support (reacts in y only)
LOAD_INDEX = 2              # joint carrying the downward load
FIRST_FREE_INDEX = 3        # joints 3+ are free joints
TEST_LOAD = 1.0             # oz, unit load used for the linear analysis (P_max scales from it)

# --------------------------------------------------------------------------------------
# Joint / member counts (derived from the stock sticks)
# --------------------------------------------------------------------------------------
# Determinate, fully triangulated truss: M = 2J - 3.
# Upper bound: every member is at least MEMBER_LENGTH_MIN long and comes from a single stick, so a
# stick of length s holds at most floor(s / MEMBER_LENGTH_MIN) members (48 in: 5, 24 in: 2).
# M = 2J - 3 is always odd, so J_MAX is the largest J whose M still fits.
STOCK_MEMBER_LIMIT = sum(math.floor(s / MEMBER_LENGTH_MIN) for s in STOCK_LENGTHS)   # 12
J_MAX = (STOCK_MEMBER_LIMIT + 3) // 2
M_MAX = 2 * J_MAX - 3
# Lower bound: the pin and roller are at least SPAN_MIN apart and no member is longer than
# MEMBER_LENGTH_MAX, so any path between them needs at least this many members.
MIN_HOPS_PIN_TO_ROLLER = math.ceil(SPAN_MIN / MEMBER_LENGTH_MAX)
J_MIN = MIN_HOPS_PIN_TO_ROLLER + 1


def members_for_joints(j: int) -> int:
    """Number of members in a determinate truss with j joints (M = 2J - 3)."""
    return 2 * j - 3


# --------------------------------------------------------------------------------------
# Optimiser settings
# --------------------------------------------------------------------------------------
RANDOM_SEED = 301
SCREEN_SAMPLES_PER_CANDIDATE = 200   # random geometries tried per (topology, roles) candidate
TOP_K = 40                           # candidates kept after screening, per objective
SLSQP_STARTS = 8                     # multi-start count per candidate in the geometry optimiser
SLSQP_MAX_ITER = 200
N_FINALISTS = 5                      # designs per objective sent to MATLAB (later)

# --------------------------------------------------------------------------------------
# Scoring / penalties (scores are minimised by the optimiser)
# --------------------------------------------------------------------------------------
FEASIBILITY_TOL = 1e-9               # total constraint violation at or below this counts as feasible
STRIP_VIOLATION_FLOOR = 1e-3         # minimum violation for a joint inside the strip (in)
CROSSING_VIOLATION = 1.0             # violation added per crossing member pair
PENALTY_INFEASIBLE = 1e3             # offset for geometry-infeasible designs (worse than any feasible)
PENALTY_SINGULAR = 1e4               # offset for singular / non-determinate designs
COST_TIEBREAK = 1e-6                 # weight on cost to break ties in the max-load objective

# --------------------------------------------------------------------------------------
# Run mode
# --------------------------------------------------------------------------------------
USE_MATLAB = False                   # default for the pipeline; the --matlab flag turns it on
MATLAB_EXECUTABLE = None             # path to the MATLAB launcher; None = auto-detect (PATH, then
                                     # /Applications/MATLAB_*.app/bin/matlab)
SUBMISSION_TEAM = "XingyangAshleyLeonardo"   # team part of the submission file name, written like the
                                             # example TrussDesign1_MaryJoeBob_A1.mat (first names)
SUBMISSION_SECTION = "A3"            # section part of that name (EK301 section A3, from the MATLAB header)
MATLAB_TIMEOUT = 900                 # seconds allowed for one batch MATLAB run (startup included)
MATLAB_REL_TOL = 1e-3                # relative tolerance for the Python-vs-MATLAB comparison
                                     # (MATLAB uses 3391.3, Python 3391.349: about 1.4e-5 apart)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MATLAB_DIR = ROOT / "matlab"
MATLAB_FILE = MATLAB_DIR / "truss_analysis.m"
RESULTS_DIR = ROOT / "results"
DESIGNS_DIR = RESULTS_DIR / "designs"
FIGURES_DIR = RESULTS_DIR / "figures"
MATLAB_IO_DIR = RESULTS_DIR / "matlab_io"
