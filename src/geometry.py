# geometry.py
# Pure geometry helpers with no knowledge of loads or scoring.
#
# Conventions:
#   coords  : (J, 2) float array of joint positions, row index = joint index
#             (0 = pin at (0, 0), 1 = roller at (span, 0), 2 = load joint, 3+ = free joints)
#   members : sequence of (i, j) joint index pairs
#   params  : optimiser vector [span, x_load, h_load, x3, y3, x4, y4, ...]

import numpy as np

import config
from src import packing

_EPS = 1e-9


# --------------------------------------------------------------------------------------
# Parameter vector <-> coordinates
# --------------------------------------------------------------------------------------
def params_to_coords(params) -> np.ndarray:
    """Convert [span, x_load, h_load, x3, y3, ...] into a (J, 2) coordinate array."""
    p = np.asarray(params, dtype=float)
    if p.size < 3 or (p.size - 3) % 2 != 0:
        raise ValueError("params must have length 3 + 2 * (number of free joints)")
    n_free = (p.size - 3) // 2
    coords = np.zeros((3 + n_free, 2))
    coords[config.PIN_INDEX] = (0.0, 0.0)
    coords[config.ROLLER_INDEX] = (p[0], 0.0)
    coords[config.LOAD_INDEX] = (p[1], p[2])
    coords[config.FIRST_FREE_INDEX:] = p[3:].reshape(n_free, 2)
    return coords


def coords_to_params(coords) -> np.ndarray:
    """Inverse of params_to_coords (assumes the pin is at the origin and the roller at y = 0)."""
    c = np.asarray(coords, dtype=float)
    head = [c[config.ROLLER_INDEX, 0], c[config.LOAD_INDEX, 0], c[config.LOAD_INDEX, 1]]
    return np.concatenate([head, c[config.FIRST_FREE_INDEX:].ravel()])


def span(coords) -> float:
    """Horizontal distance from pin to roller."""
    c = np.asarray(coords, dtype=float)
    return float(c[config.ROLLER_INDEX, 0] - c[config.PIN_INDEX, 0])


# --------------------------------------------------------------------------------------
# Member lengths and angles
# --------------------------------------------------------------------------------------
def member_vectors(coords, members) -> np.ndarray:
    """(M, 2) vectors from the first to the second joint of each member."""
    c = np.asarray(coords, dtype=float)
    m = np.asarray(members, dtype=int)
    return c[m[:, 1]] - c[m[:, 0]]


def member_lengths(coords, members) -> np.ndarray:
    return np.hypot(*member_vectors(coords, members).T)


def member_angles(coords, members, degrees: bool = True) -> np.ndarray:
    """Angle of each member from the +x axis, folded into [0, 180) (members are undirected)."""
    v = member_vectors(coords, members)
    ang = np.arctan2(v[:, 1], v[:, 0]) % np.pi
    return np.degrees(ang) if degrees else ang


# --------------------------------------------------------------------------------------
# Crossing / planarity
# --------------------------------------------------------------------------------------
def _orient(a, b, c) -> float:
    """Twice the signed area of triangle abc (>0 counter-clockwise)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p) -> bool:
    """True if p, known to be collinear with ab, lies within the segment's bounding box."""
    return (min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
            and min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS)


def segments_cross(p1, p2, p3, p4) -> bool:
    """True if segments p1p2 and p3p4 intersect anywhere, including touching or overlapping.

    Callers handle members that legitimately share an endpoint (see members_cross).
    """
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS)) and \
       ((d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS)):
        return True
    if abs(d1) <= _EPS and _on_segment(p3, p4, p1):
        return True
    if abs(d2) <= _EPS and _on_segment(p3, p4, p2):
        return True
    if abs(d3) <= _EPS and _on_segment(p1, p2, p3):
        return True
    if abs(d4) <= _EPS and _on_segment(p1, p2, p4):
        return True
    return False


def members_cross(coords, m1, m2) -> bool:
    """True if two members cross, touch or overlap anywhere other than a shared end joint."""
    c = np.asarray(coords, dtype=float)
    shared = set(m1) & set(m2)
    if len(shared) == 2:          # duplicate member
        return True
    if len(shared) == 1:
        # Only a problem if the members fold back on each other (collinear, same direction).
        s = shared.pop()
        a = m1[0] if m1[1] == s else m1[1]
        b = m2[0] if m2[1] == s else m2[1]
        va, vb = c[a] - c[s], c[b] - c[s]
        cross = va[0] * vb[1] - va[1] * vb[0]
        scale = np.hypot(*va) * np.hypot(*vb)
        return bool(abs(cross) <= _EPS * max(scale, 1.0) and np.dot(va, vb) > 0)
    return segments_cross(c[m1[0]], c[m1[1]], c[m2[0]], c[m2[1]])


def count_crossings(coords, members) -> int:
    """Number of member pairs that cross, touch or overlap (0 for a planar layout)."""
    ms = [tuple(int(k) for k in m) for m in members]
    n = 0
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            if members_cross(coords, ms[i], ms[j]):
                n += 1
    return n


# --------------------------------------------------------------------------------------
# Strip rule
# --------------------------------------------------------------------------------------
def joints_in_strip(coords, tol: float = config.STRIP_TOLERANCE) -> list:
    """Indices of joints directly below the load joint.

    A joint is in the strip if its x is within +/- tol of the load joint's x and it is lower than
    the load joint. Members passing underneath the load joint are allowed, so only joints count.
    """
    c = np.asarray(coords, dtype=float)
    xl, yl = c[config.LOAD_INDEX]
    out = []
    for j in range(len(c)):
        if j == config.LOAD_INDEX:
            continue
        if abs(c[j, 0] - xl) <= tol + _EPS and c[j, 1] < yl - _EPS:
            out.append(j)
    return out


# --------------------------------------------------------------------------------------
# Mirror / canonical form
# --------------------------------------------------------------------------------------
def mirror_design(coords, members):
    """Mirror a design about the vertical midline of the span (x -> span - x).

    The pin and roller swap roles, so joint labels 0 and 1 are exchanged to keep the pin at the
    origin. Under a vertical load the member forces are identical to the original's.
    """
    c = np.asarray(coords, dtype=float)
    s = span(c)
    m = c.copy()
    m[:, 0] = s - m[:, 0]
    m[[config.PIN_INDEX, config.ROLLER_INDEX]] = m[[config.ROLLER_INDEX, config.PIN_INDEX]]
    swap = {config.PIN_INDEX: config.ROLLER_INDEX, config.ROLLER_INDEX: config.PIN_INDEX}
    new_members = [(swap.get(int(a), int(a)), swap.get(int(b), int(b))) for a, b in members]
    return m, new_members


def _design_key(coords, members, step: float):
    q = np.round(np.asarray(coords, dtype=float) / step).astype(int)
    mem = tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in members))
    return (tuple(map(tuple, q.tolist())), mem)


def canonical_key(coords, members, step: float = config.COORD_ROUNDING):
    """Hashable key equal for a design and its mirror image (coordinates rounded to `step`)."""
    mc, mm = mirror_design(coords, members)
    return min(_design_key(coords, members, step), _design_key(mc, mm, step))


# --------------------------------------------------------------------------------------
# Quick report (full constraint scoring lives in scoring.py)
# --------------------------------------------------------------------------------------
def geometry_report(coords, members) -> dict:
    """Pure-geometry rule check: counts and lists only, no penalties."""
    c = np.asarray(coords, dtype=float)
    lengths = member_lengths(c, members)
    h = c[config.LOAD_INDEX, 1] - c[config.PIN_INDEX, 1]
    return {
        "span": span(c),
        "span_ok": config.SPAN_MIN <= span(c) <= config.SPAN_MAX,
        "load_height": float(h),
        "load_height_ok": config.LOAD_HEIGHT_MIN <= h <= config.LOAD_HEIGHT_MAX,
        "bad_length_members": [i for i, L in enumerate(lengths)
                               if not config.MEMBER_LENGTH_MIN <= L <= config.MEMBER_LENGTH_MAX],
        "total_length": float(lengths.sum()),
        "over_budget": bool(not packing.is_packable(lengths)),   # cannot be cut from the sticks
        "joints_below_supports": [j for j in range(len(c)) if c[j, 1] < -_EPS],
        "joints_in_strip": joints_in_strip(c),
        "crossings": count_crossings(c, members),
    }
