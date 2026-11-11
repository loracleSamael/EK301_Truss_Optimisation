# geometry_opt.py
# Stage 2 of the search: continuous geometry optimisation for one candidate (topology + roles).
#
# For a fixed topology the failure load is P_max = min over compression members of
# capacity_i(L_i) / |F_i|. That min is written in epigraph form so SLSQP can handle it:
#
#   variables   v = [span, x_load, h_load, x3, y3, x4, y4, ..., t]        (t = failure load)
#   maximise    t                          (objective 1, cost only breaks ties)
#               t / cost                   (objective 2)
#   subject to  capacity_i(L_i) - t * max(0, -F_i) >= 0       for every member i
#               L_min + d <= L_i <= L_max - d                  member lengths
#               sum(L_i on stick s) <= stick_s - d_s           stock: each member is cut from one of
#                                                               the 48 + 48 + 24 in sticks. The member-to-
#                                                               stick assignment is FIXED per run (a
#                                                               discrete outer choice from packing.py), so
#                                                               these are linear; the best assignment of
#                                                               each result is tried again (repack rounds)
#               bounds on span, load height, y >= 0 and x ranges
#               side * (x_j - x_load) >= tol + d_s              strip rule for joints that start
#                                                               below the load joint
#
# Crossings are not in the problem; they are checked afterwards and offenders are discarded.
# The small margins d keep the answer legal after rounding to COORD_ROUNDING. A greedy grid polish
# on the rounded coordinates then recovers most of what the margins cost.

import random
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

import config
from src import analysis, geometry, packing, scoring
from src import optimiser as screening
from src import topology as tp

LENGTH_MARGIN = 0.02            # in, inside each member length limit
STRIP_MARGIN = 0.02             # in, outside the strip
REPACK_ROUNDS = 2               # re-solves with the best assignment of the previous result
POLISH_TOP = 3                  # distinct designs per candidate that get the grid polish
X_RANGE = (-config.MEMBER_LENGTH_MAX, config.SPAN_MAX + config.MEMBER_LENGTH_MAX)
Y_MAX = 3 * config.MEMBER_LENGTH_MAX
T_MAX = 1e4                     # keeps t finite if a design has no compression member
START_JITTER = 1.0              # in, std dev of the perturbation that creates extra starts
POLISH_PASSES = 40              # max sweeps of the +/-COORD_ROUNDING grid polish


@dataclass
class Design:
    """A finished design: rounded coordinates plus its evaluation under the project rules."""
    candidate: tp.Candidate
    coords: np.ndarray
    evaluation: scoring.Evaluation
    objective: str
    score: float
    info: dict = field(default_factory=dict)

    @property
    def p_max(self) -> float:
        return self.evaluation.p_max

    @property
    def cost(self) -> float:
        return self.evaluation.cost

    @property
    def p_over_cost(self) -> float:
        return self.evaluation.p_over_cost


# --------------------------------------------------------------------------------------
# The SLSQP problem
# --------------------------------------------------------------------------------------
class _Problem:
    """Objective, constraints and bounds for one candidate and one objective."""

    def __init__(self, candidate: tp.Candidate, objective: str, start_coords: np.ndarray,
                 assignment=None, offset: float = 0.0):
        self.cand = candidate
        self.objective = objective
        self.offset = float(offset)         # oz subtracted from every capacity (sensitivity runs)
        self.members = np.asarray(candidate.members, dtype=int)
        self.n_members = len(self.members)
        self.n_joints = candidate.n_joints
        self.n_params = 3 + 2 * (self.n_joints - config.FIRST_FREE_INDEX)
        # Stock rule for a FIXED assignment of members to sticks: the members on a stick may not use
        # more than its length. These are linear in the member lengths, which suits SLSQP.
        if assignment is None:
            assignment = packing.min_overflow(
                geometry.member_lengths(start_coords, self.members)).assignment
        self.assignment = np.asarray(assignment, dtype=int)
        sticks = [s for s in range(len(config.STOCK_LENGTHS)) if np.any(self.assignment == s)]
        self.stick_matrix = np.array([(self.assignment == s).astype(float) for s in sticks])
        self.stick_room = np.array([config.STOCK_LENGTHS[s] - config.STOCK_MARGIN for s in sticks])
        # strip rule: joints that start below the load joint keep their side of it
        xl, yl = start_coords[config.LOAD_INDEX]
        self.strip = [(j, 1.0 if start_coords[j, 0] >= xl else -1.0)
                      for j in range(self.n_joints)
                      if j != config.LOAD_INDEX and start_coords[j, 1] < yl]
        self._key = None
        self._val = None

    # -- shared evaluation, cached so objective and constraints reuse one solve ---------------
    def _eval(self, v):
        key = v.tobytes()
        if key == self._key:
            return self._val
        coords = geometry.params_to_coords(v[:-1])
        vec = coords[self.members[:, 1]] - coords[self.members[:, 0]]
        length = np.maximum(np.hypot(vec[:, 0], vec[:, 1]), 1e-9)
        A = analysis.equilibrium_matrix(coords, self.members)
        b = np.zeros(2 * self.n_joints)
        b[2 * config.LOAD_INDEX + 1] = config.TEST_LOAD
        try:
            x = np.linalg.solve(A, b)
            force = x[:self.n_members]
            if not np.all(np.isfinite(force)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            force = np.full(self.n_members, -1e6)            # singular: make every cap constraint fail
        self._key, self._val = key, (coords, length, force)
        return self._val

    # -- objective ---------------------------------------------------------------------------
    def fun(self, v):
        _, length, _ = self._eval(v)
        t = v[-1]
        cost = analysis.total_cost(self.n_joints, length.sum())
        if self.objective == scoring.OBJ_MAX_LOAD:
            return -t + config.COST_TIEBREAK * cost
        return -t / cost

    # -- constraints (all >= 0) --------------------------------------------------------------
    def cons(self, v):
        coords, length, force = self._eval(v)
        t = v[-1]
        cap = analysis.buckling_capacity(length) - self.offset
        compression = np.maximum(-force, 0.0) / config.TEST_LOAD
        out = [
            cap - t * compression,                                        # buckling
            length - (config.MEMBER_LENGTH_MIN + LENGTH_MARGIN),         # long enough
            (config.MEMBER_LENGTH_MAX - LENGTH_MARGIN) - length,         # short enough
            self.stick_room - self.stick_matrix @ length,                # each stick's length
        ]
        if self.strip:
            xl = coords[config.LOAD_INDEX, 0]
            out.append([s * (coords[j, 0] - xl) - (config.STRIP_TOLERANCE + STRIP_MARGIN)
                        for j, s in self.strip])
        return np.concatenate([np.atleast_1d(np.asarray(o, dtype=float)) for o in out])

    def bounds(self):
        b = [(config.SPAN_MIN, config.SPAN_MAX), X_RANGE,
             (config.LOAD_HEIGHT_MIN, config.LOAD_HEIGHT_MAX)]
        b += [X_RANGE, (0.0, Y_MAX)] * (self.n_joints - config.FIRST_FREE_INDEX)
        b.append((0.0, T_MAX))
        return b


def _start_vector(coords: np.ndarray, problem: _Problem) -> np.ndarray:
    p = geometry.coords_to_params(coords)
    p[0] = np.clip(p[0], config.SPAN_MIN, config.SPAN_MAX)
    p[2] = np.clip(p[2], config.LOAD_HEIGHT_MIN, config.LOAD_HEIGHT_MAX)
    v = np.append(p, 1.0)
    res = analysis.analyse(geometry.params_to_coords(p), problem.members)
    v[-1] = min(max(res.p_max, 1.0), T_MAX) if res.valid else 1.0
    return v


def optimise_geometry(candidate: tp.Candidate, start_coords: np.ndarray, objective: str,
                      max_iter: int = config.SLSQP_MAX_ITER, assignment=None, offset: float = 0.0):
    """Run SLSQP from one starting geometry. Returns (coords, scipy result).

    `assignment` fixes which stick each member is cut from (default: the best packing of the start
    lengths). The stock limits are then linear constraints, so any result is cuttable by construction.
    `offset` (oz) is subtracted from every member's buckling capacity; the pipeline only uses it for
    the sensitivity check, never for the main optimisation.
    """
    problem = _Problem(candidate, objective, start_coords, assignment, offset)
    v0 = _start_vector(start_coords, problem)
    res = minimize(problem.fun, v0, method="SLSQP", bounds=problem.bounds(),
                   constraints=[{"type": "ineq", "fun": problem.cons}],
                   options={"maxiter": max_iter, "ftol": 1e-10})
    coords = geometry.params_to_coords(res.x[:-1])
    return coords, res


# --------------------------------------------------------------------------------------
# Rounding and grid polish
# --------------------------------------------------------------------------------------
def round_coords(coords: np.ndarray, step: float = config.COORD_ROUNDING) -> np.ndarray:
    """Round to the export grid; the pin stays at the origin and the roller on y = 0."""
    out = np.round(np.asarray(coords, dtype=float) / step) * step
    out = np.round(out, 6)                                   # strip float noise such as 0.30000000000000004
    out[config.PIN_INDEX] = (0.0, 0.0)
    out[config.ROLLER_INDEX, 1] = 0.0
    return out


def _score(coords, members, objective):
    ev = scoring.evaluate(coords, members)
    return scoring.score(ev, objective), ev


def grid_polish(candidate: tp.Candidate, coords: np.ndarray, objective: str,
                step: float = config.COORD_ROUNDING, passes: int = POLISH_PASSES):
    """Greedy +/- step moves on single coordinates of the rounded design while the score improves.

    Recovers what the safety margins and rounding cost. Returns (coords, score, evaluation).
    """
    best = np.array(coords, dtype=float)
    best_score, best_ev = _score(best, candidate.members, objective)
    movable = [(j, k) for j in range(candidate.n_joints) for k in (0, 1)
               if j != config.PIN_INDEX and not (j == config.ROLLER_INDEX and k == 1)]
    for _ in range(passes):
        improved = False
        for j, k in movable:
            for sign in (1.0, -1.0):
                trial = best.copy()
                trial[j, k] = round(trial[j, k] + sign * step, 6)
                s, ev = _score(trial, candidate.members, objective)
                if s < best_score - 1e-12:
                    best, best_score, best_ev, improved = trial, s, ev, True
                    break
        if not improved:
            break
    return best, best_score, best_ev


def finalise(candidate: tp.Candidate, coords: np.ndarray, objective: str, polish: bool = True):
    """Round a raw SLSQP geometry, re-score it, optionally polish it. None if it is not legal."""
    rounded = round_coords(coords)
    score, ev = _score(rounded, candidate.members, objective)
    if not ev.valid:
        return None
    if polish:
        rounded, score, ev = grid_polish(candidate, rounded, objective)
    return Design(candidate, rounded, ev, objective, score)


# --------------------------------------------------------------------------------------
# Multi-start per candidate
# --------------------------------------------------------------------------------------
def _rng_for(candidate: tp.Candidate, objective: str, seed: int = None) -> random.Random:
    rng = screening._rng_for(candidate, seed)
    return random.Random(f"{rng.random()}|{objective}")


def start_points(screen: screening.Screen, objective: str, n_starts: int, rng: random.Random):
    """Starting geometries for SLSQP: the screened bests, then jittered and repaired copies.

    Candidates with no legal screened sample fall back to fresh random constructions.
    """
    cand = screen.candidate
    bases = []
    for obj in [objective] + [o for o in scoring.OBJECTIVES if o != objective]:
        co = screen.best_coords(obj)
        if co is not None and not any(np.allclose(co, b) for b in bases):
            bases.append(co)
    if not bases:
        orders = screening.construction_orders(cand, rng)
        drawn = [c for c in (screening.sample_geometry(cand, orders, rng) for _ in range(40))
                 if c is not None] if orders else []
        if drawn:
            batch = screening._spread_load_joint(np.stack(drawn), rng)
            bases = list(screening.repair_batch(cand, batch)[:2])
    if not bases:
        return []
    starts = list(bases[:n_starts])
    n_extra = n_starts - len(starts)
    if n_extra > 0:
        noise = np.array([[rng.gauss(0.0, START_JITTER) for _ in range(2 * cand.n_joints)]
                          for _ in range(n_extra)]).reshape(n_extra, cand.n_joints, 2)
        noise[:, [config.PIN_INDEX, config.ROLLER_INDEX], :] = 0.0
        parents = np.stack([bases[i % len(bases)] for i in range(n_extra)])
        starts.extend(screening.repair_batch(cand, parents + noise))
    return starts


def _lengths(candidate: tp.Candidate, coords) -> np.ndarray:
    return geometry.member_lengths(coords, candidate.members)


def solve_with_repack(candidate: tp.Candidate, start: np.ndarray, objective: str, assignment,
                      rounds: int = REPACK_ROUNDS, max_iter: int = config.SLSQP_MAX_ITER):
    """SLSQP with a fixed member-to-stick assignment, then re-solve with the best assignment of the
    result (the first assignment is only a guess) until it stops changing. Best legal Design or None.
    """
    best, tried = None, set()
    cur_start, cur_assign = start, assignment
    for _ in range(rounds + 1):
        key = tuple(int(a) for a in cur_assign)
        if key in tried:
            break
        tried.add(key)
        try:
            coords, _ = optimise_geometry(candidate, cur_start, objective, max_iter, cur_assign)
        except (ValueError, np.linalg.LinAlgError):
            break
        design = finalise(candidate, coords, objective, polish=False)
        if design is not None and (best is None or design.score < best.score):
            best = design
        cur_start = design.coords if design is not None else coords
        cur_assign = packing.min_overflow(_lengths(candidate, cur_start)).assignment
    return best


def _grid_polished(design: Design, extra_info: dict) -> Design:
    coords, score, ev = grid_polish(design.candidate, design.coords, design.objective)
    return Design(design.candidate, coords, ev, design.objective, score, {**design.info, **extra_info})


def optimise_candidate(screen: screening.Screen, objective: str,
                       n_starts: int = config.SLSQP_STARTS):
    """Best legal Design over several starts and stick assignments, or None if none is legal."""
    cand = screen.candidate
    rng = _rng_for(cand, objective, screen.seed)
    designs, n_runs = [], 0
    for start in start_points(screen, objective, n_starts, rng):
        for pr in packing.top_assignments(_lengths(cand, start),
                                          config.PACKING_ASSIGNMENTS_PER_START):
            n_runs += 1
            design = solve_with_repack(cand, start, objective, pr.assignment)
            if design is not None:
                designs.append(design)
    if not designs:
        return None
    # The grid polish changes the ranking slightly (rounding and margin losses differ per design),
    # so polish the few best distinct designs and keep the best afterwards.
    designs.sort(key=lambda d: d.score)
    distinct = []
    for d in designs:
        if not any(np.allclose(d.coords, o.coords) for o in distinct):
            distinct.append(d)
        if len(distinct) == POLISH_TOP:
            break
    info = {"legal_runs": len(designs), "runs": n_runs, "starts": n_starts}
    return min((_grid_polished(d, info) for d in distinct), key=lambda d: d.score)


def polish_design(design: Design) -> Design:
    """One more round of SLSQP (more iterations, several assignments) from a finished design."""
    cand, best = design.candidate, design
    for pr in packing.top_assignments(_lengths(cand, design.coords),
                                      config.PACKING_ASSIGNMENTS_PER_START):
        again = solve_with_repack(cand, design.coords, design.objective, pr.assignment,
                                  max_iter=4 * config.SLSQP_MAX_ITER)
        if again is not None and again.score < best.score - 1e-12:
            best = again
    if best is design:
        return design
    return _grid_polished(best, {**design.info, "polished": True})


def _candidate_job(args):
    screen, objective, n_starts = args
    return optimise_candidate(screen, objective, n_starts)


def optimise_all(screens, objective: str, n_starts: int = config.SLSQP_STARTS,
                 n_polish: int = config.N_FINALISTS, workers: int = 1, progress=None) -> list:
    """Optimise every screened candidate for one objective; return Designs, best first.

    The best n_polish designs get an extra SLSQP polish before the list is returned.
    """
    jobs = [(s, objective, n_starts) for s in screens]
    designs = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, d in enumerate(pool.map(_candidate_job, jobs, chunksize=2)):
                if d is not None:
                    designs.append(d)
                if progress is not None:
                    progress(i + 1, len(jobs))
    else:
        for i, job in enumerate(jobs):
            d = _candidate_job(job)
            if d is not None:
                designs.append(d)
            if progress is not None:
                progress(i + 1, len(jobs))
    designs.sort(key=lambda d: d.score)
    for i in range(min(n_polish, len(designs))):
        designs[i] = polish_design(designs[i])
    designs.sort(key=lambda d: d.score)
    return designs
