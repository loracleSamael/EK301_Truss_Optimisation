# optimiser.py
# The search that improves designs. Stage 1 (this section): screening.
#
# Screening gives every (topology, roles) candidate from topology.py a cheap look:
#   - sample random geometries by building each truss the way it was grown (Henneberg order), so
#     every member length is in range by construction,
#   - fit the sample to the supports (pin at the origin, roller at (S, 0), S in the span range),
#   - add hand-built Warren / Pratt / Howe layouts as extra samples where a candidate matches,
#   - score each sample with scoring.py and remember the best one per objective.
# The best candidates are then handed to the geometry optimiser (stage 2).
#
# Randomness is deterministic: each candidate gets its own random stream derived from
# config.RANDOM_SEED and the candidate itself, so results do not depend on evaluation order.

import math
import random
import zlib
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

import config
from src import geometry, scoring
from src import topology as tp

N_ORDERS = 6                        # construction orders tried per candidate
MAX_PLACEMENT_TRIES = 30            # attempts to find legal lengths when placing one joint
REFINE_ROUNDS = 2                   # perturb-and-repair rounds on the least-violating samples
REFINE_PARENTS = 8                  # samples perturbed per round
REFINE_FRACTION = 0.5               # children per round, as a fraction of n_samples
REFINE_SIGMA = 1.5                  # std dev of the perturbation of each coordinate (in)
MIN_SUPPORT_DISTANCE = 3.0          # reject constructions whose supports nearly coincide
HEIGHT_STRETCH = (0.6, 1.4)         # random stretch across the support line, relative to along it
MAX_SHEAR = 0.3                     # random shear, relative to the along-line stretch

# Repair step (batched gradient descent on a smooth rule-violation energy)
REPAIR_ITERS = 150
REPAIR_LR = 0.08
REPAIR_MAX_STEP = 1.0               # largest move of one coordinate per iteration (in)
REPAIR_MARGIN = 0.05                # aim this far inside each limit so rounding stays legal
REPAIR_BUDGET_MARGIN = 1.0          # aim for total length <= budget - margin


# --------------------------------------------------------------------------------------
# Construction orders
# --------------------------------------------------------------------------------------
def _adjacency(candidate: tp.Candidate) -> dict:
    adj = {n: set() for n in range(candidate.n_joints)}
    for u, v in candidate.members:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_order(adj: dict, rng: random.Random):
    """Randomised search for a Henneberg-I order.

    Returns (base_edge, steps) where steps is a list of (joint, anchor1, anchor2) in the order the
    joints are added, or None if this random choice dead-ends (the caller retries).
    """
    adj = {k: set(v) for k, v in adj.items()}
    removed = []
    while len(adj) > 2:
        degree2 = [v for v, nb in adj.items() if len(nb) == 2]
        rng.shuffle(degree2)
        for v in degree2:
            u, w = sorted(adj[v])
            # Removing v must leave a graph that can still be dismantled; try and backtrack.
            sub = {k: (nb - {v}) for k, nb in adj.items() if k != v}
            rest = _find_order(sub, rng) if len(sub) > 2 else (tuple(sorted(sub)), [])
            if rest is not None:
                base, steps = rest
                return base, steps + [(v, u, w)]
        return None
    return tuple(sorted(adj)), []


def construction_orders(candidate: tp.Candidate, rng: random.Random, n: int = N_ORDERS) -> list:
    """Up to n distinct valid construction orders for the candidate's graph."""
    adj = _adjacency(candidate)
    orders, seen = [], set()
    for _ in range(4 * n):
        order = _find_order(adj, rng)
        if order is None:
            continue
        key = (order[0], tuple(order[1]))
        if key not in seen:
            seen.add(key)
            orders.append(order)
        if len(orders) == n:
            break
    return orders


# --------------------------------------------------------------------------------------
# Random geometry
# --------------------------------------------------------------------------------------
def _blocked(p, u, w, pos: dict, edges: list) -> bool:
    """True if joining new point p to joints u and w would cross, touch or fold onto a member."""
    for a, b in edges:
        for end in (u, w):
            if end in (a, b):
                # shares a joint with the new member: only a problem if they fold back on each other
                other = b if end == a else a
                s = pos[end]
                v1 = (p[0] - s[0], p[1] - s[1])
                v2 = (pos[other][0] - s[0], pos[other][1] - s[1])
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if abs(cross) < 1e-9 and v1[0] * v2[0] + v1[1] * v2[1] > 0:
                    return True
            elif geometry.segments_cross(pos[end], p, pos[a], pos[b]):
                return True
    return False


def _place(pu, pw, rng: random.Random, pos: dict = None, edges: list = None, ends=None):
    """A point at distance l1 from pu and l2 from pw, both lengths in the legal member range.

    If pos/edges/ends are given, placements that would make the new members cross existing ones
    are rejected, so the construction stays planar.
    """
    dx, dy = pw[0] - pu[0], pw[1] - pu[1]
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return None
    ex, ey = dx / d, dy / d
    lo, hi = config.MEMBER_LENGTH_MIN, config.MEMBER_LENGTH_MAX
    for _ in range(MAX_PLACEMENT_TRIES):
        l1, l2 = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if abs(l1 - l2) <= d <= l1 + l2:
            a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
            h = math.sqrt(max(l1 * l1 - a * a, 0.0))
            side = 1.0 if rng.random() < 0.5 else -1.0
            p = (pu[0] + a * ex - side * h * ey, pu[1] + a * ey + side * h * ex)
            if pos is not None and _blocked(p, ends[0], ends[1], pos, edges):
                continue
            return p
    return None


def _fit_to_supports(pos: dict, n: int, rng: random.Random):
    """Affine-map the construction so pin = (0, 0) and roller = (S, 0) with S drawn from the span
    range, then flip so that the truss lies mostly above the supports.

    An affine map can place the supports exactly wherever the construction put them and keeps the
    layout planar. Member lengths change with it; the repair step pulls them back into range.
    """
    px, py = pos[config.PIN_INDEX]
    rx, ry = pos[config.ROLLER_INDEX]
    d = math.hypot(rx - px, ry - py)
    if d < MIN_SUPPORT_DISTANCE:
        return None
    ex, ey = (rx - px) / d, (ry - py) / d
    s = rng.uniform(config.SPAN_MIN, config.SPAN_MAX) / d      # stretch along the support line
    k = s * rng.uniform(*HEIGHT_STRETCH)                       # stretch across it
    shear = s * rng.uniform(-MAX_SHEAR, MAX_SHEAR)
    coords = np.empty((n, 2))
    for j in range(n):
        x, y = pos[j][0] - px, pos[j][1] - py
        a, b = x * ex + y * ey, -x * ey + y * ex               # along / across the support line
        coords[j] = (s * a + shear * b, k * b)
    coords[config.ROLLER_INDEX, 1] = 0.0
    if coords[:, 1].sum() < 0:
        coords[:, 1] *= -1.0
    return coords


def sample_geometry(candidate: tp.Candidate, orders: list, rng: random.Random):
    """One random geometry for the candidate (a (J, 2) array) or None if the draw failed."""
    base, steps = orders[rng.randrange(len(orders))]
    length = rng.uniform(config.MEMBER_LENGTH_MIN, config.MEMBER_LENGTH_MAX)
    pos = {base[0]: (0.0, 0.0), base[1]: (length, 0.0)}
    edges = [base]
    for v, u, w in steps:
        p = _place(pos[u], pos[w], rng, pos, edges, (u, w))
        if p is None:
            return None
        pos[v] = p
        edges += [(v, u), (v, w)]
    return _fit_to_supports(pos, candidate.n_joints, rng)


def _incidence(candidate: tp.Candidate) -> np.ndarray:
    """(M, J) matrix D with D @ coords = vector from first to second joint of each member."""
    d = np.zeros((candidate.n_members, candidate.n_joints))
    for k, (i, j) in enumerate(candidate.members):
        d[k, i], d[k, j] = -1.0, 1.0
    return d


def repair_batch(candidate: tp.Candidate, batch: np.ndarray,
                 iters: int = REPAIR_ITERS, lr: float = REPAIR_LR) -> np.ndarray:
    """Pull a batch of geometries (B, J, 2) towards the smooth rules by gradient descent.

    Energy per sample: squared violation of member lengths, the material budget, joint heights
    (y >= 0) and the load joint height window. The pin and roller stay fixed. Crossings and the
    strip rule are not differentiable and are checked afterwards by scoring.
    """
    X = batch.copy()
    D = _incidence(candidate)
    lo = config.MEMBER_LENGTH_MIN + REPAIR_MARGIN
    hi = config.MEMBER_LENGTH_MAX - REPAIR_MARGIN
    budget = config.MATERIAL_BUDGET - REPAIR_BUDGET_MARGIN
    h_lo = config.LOAD_HEIGHT_MIN + REPAIR_MARGIN
    h_hi = config.LOAD_HEIGHT_MAX - REPAIR_MARGIN
    free = np.ones(candidate.n_joints, dtype=bool)
    free[[config.PIN_INDEX, config.ROLLER_INDEX]] = False
    for _ in range(iters):
        vec = np.einsum("mj,bjk->bmk", D, X)
        length = np.maximum(np.hypot(vec[..., 0], vec[..., 1]), 1e-9)
        unit = vec / length[..., None]
        # dE/dL for each member: length limits plus the shared material-budget term
        g = 2.0 * (np.maximum(length - hi, 0.0) - np.maximum(lo - length, 0.0))
        excess = np.maximum(length.sum(axis=1) - budget, 0.0)
        g = g + 2.0 * excess[:, None] / candidate.n_members      # damped: shared by all members
        grad = np.einsum("mj,bmk->bjk", D, g[..., None] * unit)
        y = X[..., 1]
        grad[..., 1] += 2.0 * np.minimum(y, 0.0)                         # keep y >= 0
        yl = y[:, config.LOAD_INDEX]
        grad[:, config.LOAD_INDEX, 1] += 2.0 * (np.maximum(yl - h_hi, 0.0)
                                                - np.maximum(h_lo - yl, 0.0))
        grad[:, ~free, :] = 0.0
        X -= np.clip(lr * grad, -REPAIR_MAX_STEP, REPAIR_MAX_STEP)
    return X


def _spread_load_joint(batch: np.ndarray, rng: random.Random):
    """Give each sample's load joint a height inside the legal window (repair starts closer)."""
    for b in range(len(batch)):
        batch[b, config.LOAD_INDEX, 1] = rng.uniform(config.LOAD_HEIGHT_MIN,
                                                     config.LOAD_HEIGHT_MAX)
    return batch


# --------------------------------------------------------------------------------------
# Seed layouts (classic bridge shapes) matched onto candidates
# --------------------------------------------------------------------------------------
def _seed(params, members, name):
    return {"name": name, "coords": geometry.params_to_coords(params), "members": list(members)}


def seed_layouts() -> list:
    """Hand-built Warren, Pratt and Howe trusses in the project joint convention."""
    # Warren, 3 bottom panels of ~10 in with top joints between them. 7 joints.
    warren = _seed([30, 20, 1, 10, 0, 5, 8, 15, 8, 25, 8],
                   [(0, 3), (3, 2), (2, 1), (4, 5), (5, 6), (0, 4), (4, 3), (3, 5), (5, 2),
                    (2, 6), (6, 1)], "warren")
    # Pratt / Howe: three panels with verticals over the interior bottom joints and a single
    # diagonal in the middle panel. 6 joints.
    base = [30, 20, 1, 10, 0, 10, 9.5, 20, 9.7]
    common = [(0, 3), (3, 2), (2, 1), (4, 5), (0, 4), (1, 5), (3, 4), (2, 5)]
    pratt = _seed(base, common + [(4, 2)], "pratt")
    howe = _seed(base, common + [(3, 5)], "howe")
    return [warren, pratt, howe]


def _seed_graph(seed: dict) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(seed["coords"])))
    g.add_edges_from(seed["members"])
    roles = {config.PIN_INDEX: "end", config.ROLLER_INDEX: "end", config.LOAD_INDEX: "load"}
    for n in g.nodes:
        g.nodes[n]["role"] = roles.get(n, "free")
    return g


_SEEDS = None


def seed_geometries(candidate: tp.Candidate) -> list:
    """Coordinates (in the candidate's labels) of every seed layout isomorphic to the candidate."""
    global _SEEDS
    if _SEEDS is None:
        _SEEDS = [(s, _seed_graph(s)) for s in seed_layouts()]
    cand_graph = tp.candidate_graph(candidate)
    out = []
    for seed, sgraph in _SEEDS:
        if sgraph.number_of_nodes() != candidate.n_joints:
            continue
        matcher = GraphMatcher(cand_graph, sgraph, node_match=lambda a, b: a["role"] == b["role"])
        if not matcher.is_isomorphic():
            continue
        coords = np.empty_like(seed["coords"])
        for c_node, s_node in matcher.mapping.items():
            coords[c_node] = seed["coords"][s_node]
        if coords[config.PIN_INDEX, 0] > 1e-9:      # pin and roller were swapped: mirror
            coords[:, 0] = geometry.span(seed["coords"]) - coords[:, 0]
        out.append(coords)
    return out


# --------------------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------------------
@dataclass
class Screen:
    """Best result for one candidate, per objective."""
    candidate: tp.Candidate
    seed: int = None                    # random seed used (also reused by the SLSQP stage)
    n_samples: int = 0                  # geometries scored
    n_feasible: int = 0
    best_violation: float = math.inf    # smallest violation seen (of any sample)
    best: dict = field(default_factory=dict)    # objective -> (score, coords, Evaluation)

    def rank_key(self, objective: str) -> tuple:
        """Sort key: candidates with a legal sample first (best score), then least violation."""
        if objective in self.best:
            return (0, self.best[objective][0])
        return (1, self.best_violation)

    def best_coords(self, objective: str):
        return self.best[objective][1] if objective in self.best else None


def _rng_for(candidate: tp.Candidate, seed: int = None) -> random.Random:
    """Deterministic random stream for one candidate (independent of evaluation order)."""
    seed = config.RANDOM_SEED if seed is None else seed
    key = f"{seed}|{candidate.n_joints}|{candidate.members}"
    return random.Random(zlib.crc32(key.encode()))


def screen_candidate(candidate: tp.Candidate,
                     n_samples: int = config.SCREEN_SAMPLES_PER_CANDIDATE,
                     seed: int = None) -> Screen:
    """Score random + seed geometries for one candidate."""
    seed = config.RANDOM_SEED if seed is None else seed
    rng = _rng_for(candidate, seed)
    res = Screen(candidate, seed=seed)
    orders = construction_orders(candidate, rng)
    geometries = seed_geometries(candidate)
    drawn = []
    if orders:
        for _ in range(n_samples):
            c = sample_geometry(candidate, orders, rng)
            if c is not None:
                drawn.append(c)
    if drawn:
        batch = _spread_load_joint(np.stack(drawn), rng)
        geometries.extend(repair_batch(candidate, batch))
    pool = []                                               # (violation, coords) of every sample

    def consider(coords):
        res.n_samples += 1
        viol = scoring.constraint_violations(coords, candidate.members)
        total = float(sum(viol.values()))
        res.best_violation = min(res.best_violation, total)
        pool.append((total, coords))
        if total > config.FEASIBILITY_TOL:
            return                                          # skip the solve for illegal samples
        ev = scoring.evaluate(coords, candidate.members)
        if not ev.valid:
            return
        res.n_feasible += 1
        for obj in scoring.OBJECTIVES:
            s = scoring.score(ev, obj)
            if obj not in res.best or s < res.best[obj][0]:
                res.best[obj] = (s, coords.copy(), ev)

    for coords in geometries:
        consider(coords)

    # Refinement: jiggle and re-repair the least-violating samples (helps most with crossings).
    for _ in range(REFINE_ROUNDS if n_samples > 0 and pool else 0):
        pool.sort(key=lambda item: item[0])
        parents = [c for _, c in pool[:REFINE_PARENTS]]
        k = max(1, int(n_samples * REFINE_FRACTION) // len(parents))
        kids = []
        for parent in parents:
            noise = np.array([[rng.gauss(0.0, REFINE_SIGMA) for _ in range(2)]
                              for _ in range(candidate.n_joints * k)]).reshape(k, -1, 2)
            noise[:, [config.PIN_INDEX, config.ROLLER_INDEX], :] = 0.0
            kids.append(parent[None] + noise)
        for coords in repair_batch(candidate, np.concatenate(kids)):
            consider(coords)
    return res


def _screen_job(args):
    candidate, n_samples, seed = args
    return screen_candidate(candidate, n_samples, seed)


def screen_all(candidates=None, n_samples: int = config.SCREEN_SAMPLES_PER_CANDIDATE,
               top_k: int = config.TOP_K, progress=None, workers: int = 1, seed: int = None) -> dict:
    """Screen every candidate and return {objective: [Screen, ...]} with the best top_k each.

    Also returns all Screens under the key "all" for reporting. Each candidate has its own random
    stream, so the result is identical for any number of workers. workers > 1 uses processes (a
    script calling this must be guarded by `if __name__ == "__main__"`).
    """
    if candidates is None:
        candidates = tp.enumerate_candidates()
    seed = config.RANDOM_SEED if seed is None else seed       # workers do not see later changes
    jobs = [(c, n_samples, seed) for c in candidates]
    screens = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, s in enumerate(pool.map(_screen_job, jobs, chunksize=8)):
                screens.append(s)
                if progress is not None:
                    progress(i + 1, len(jobs))
    else:
        for i, job in enumerate(jobs):
            screens.append(_screen_job(job))
            if progress is not None:
                progress(i + 1, len(jobs))
    out = {"all": screens}
    for obj in scoring.OBJECTIVES:
        out[obj] = sorted(screens, key=lambda s: s.rank_key(obj))[:top_k]
    return out
