# topology.py
# Enumerates the member layouts (which joints are connected) worth optimising.
#
# Approach (exhaustive, no randomness):
#   1. Henneberg type-I growth: start from one edge and repeatedly add a joint connected to any two
#      existing joints. Every result has M = 2J - 3 and is rigid by construction.
#   2. Keep only planar graphs (planarity is hereditary, so non-planar graphs are dropped at every
#      level) and remove isomorphic duplicates.
#   3. For each graph pick which joint is the pin, the roller and the load joint. Assignments that
#      are equivalent under a graph automorphism are merged. Pin and roller are an unordered pair:
#      the mirror image of a design is the same design (geometry.mirror_design).
#   4. Drop assignments that cannot meet the geometry rules (pin and roller too few hops apart).
#
# A Candidate is a graph relabelled to the project's joint convention:
#   0 = pin, 1 = roller, 2 = load joint, 3+ = free joints.
#
# Assumption: only Henneberg type-I trusses are generated. Rigid planar trusses that need a
# type-II (edge split) step are not covered.

import itertools
from dataclasses import dataclass
from functools import lru_cache

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

import config
from src import packing


@dataclass(frozen=True)
class Candidate:
    n_joints: int
    members: tuple          # sorted tuple of (i, j) with i < j, in project joint labels
    graph_id: int           # index of the unlabelled graph within its joint count

    @property
    def n_members(self) -> int:
        return len(self.members)


# --------------------------------------------------------------------------------------
# Graph enumeration
# --------------------------------------------------------------------------------------
def is_planar(graph: nx.Graph) -> bool:
    return nx.check_planarity(graph)[0]


def _extensions(graph: nx.Graph):
    """All graphs obtained by adding one joint connected to two distinct existing joints."""
    new = graph.number_of_nodes()
    for u, v in itertools.combinations(graph.nodes, 2):
        g = graph.copy()
        g.add_edges_from([(new, u), (new, v)])
        yield g


def _invariant(graph: nx.Graph) -> tuple:
    """Cheap isomorphism invariant: sorted (degree, sorted neighbour degrees) per joint."""
    deg = dict(graph.degree)
    return tuple(sorted((deg[n], tuple(sorted(deg[m] for m in graph[n]))) for n in graph.nodes))


def _dedupe(graphs):
    """Remove isomorphic duplicates (bucketed by a cheap invariant, then exact check)."""
    buckets = {}
    unique = []
    for g in graphs:
        key = _invariant(g)
        bucket = buckets.setdefault(key, [])
        if any(nx.is_isomorphic(g, other) for other in bucket):
            continue
        bucket.append(g)
        unique.append(g)
    return unique


@lru_cache(maxsize=None)
def enumerate_graphs(n_joints: int) -> tuple:
    """All planar Henneberg type-I graphs on n_joints joints, one per isomorphism class."""
    if n_joints < 2:
        raise ValueError("need at least 2 joints")
    if n_joints == 2:
        g = nx.Graph()
        g.add_edge(0, 1)
        return (g,)
    grown = (e for g in enumerate_graphs(n_joints - 1) for e in _extensions(g))
    return tuple(_dedupe(g for g in grown if is_planar(g)))


# --------------------------------------------------------------------------------------
# Role assignment (pin / roller / load joint)
# --------------------------------------------------------------------------------------
def _automorphisms(graph: nx.Graph) -> list:
    return list(GraphMatcher(graph, graph).isomorphisms_iter())


def _canonical_roles(roles, automorphisms) -> tuple:
    """Smallest equivalent (pin/roller pair, load joint) over all automorphisms."""
    (a, b), load = roles
    return min((tuple(sorted((s[a], s[b]))), s[load]) for s in automorphisms)


def min_pin_roller_hops() -> int:
    """Fewest members on any path from pin to roller, given the span and max member length."""
    return config.MIN_HOPS_PIN_TO_ROLLER


def _relabel(graph: nx.Graph, pin: int, roller: int, load: int) -> tuple:
    rest = [n for n in sorted(graph.nodes) if n not in (pin, roller, load)]
    order = [pin, roller, load] + rest
    new = {old: i for i, old in enumerate(order)}
    return tuple(sorted(tuple(sorted((new[u], new[v]))) for u, v in graph.edges))


def role_assignments(graph: nx.Graph) -> list:
    """Distinct ((pin, roller), load) assignments, merged under graph automorphisms and filtered
    by the minimum number of hops between pin and roller."""
    autos = _automorphisms(graph)
    dist = dict(nx.all_pairs_shortest_path_length(graph))
    seen = set()
    out = []
    for a, b in itertools.combinations(graph.nodes, 2):
        if dist[a][b] < min_pin_roller_hops():
            continue
        for load in graph.nodes:
            if load in (a, b):
                continue
            key = _canonical_roles(((a, b), load), autos)
            if key in seen:
                continue
            seen.add(key)
            out.append(((a, b), load))
    return out


def candidates_for_joints(n_joints: int) -> list:
    """Every (topology, roles) candidate for a given joint count, relabelled to project indices."""
    cands = []
    for gid, graph in enumerate(enumerate_graphs(n_joints)):
        for (pin, roller), load in role_assignments(graph):
            members = _relabel(graph, pin, roller, load)
            if members:
                cands.append(Candidate(n_joints, members, gid))
    return cands


def enumerate_candidates(j_min: int = config.J_MIN, j_max: int = config.J_MAX) -> list:
    """All candidates for joint counts j_min..j_max inclusive."""
    out = []
    for j in range(j_min, j_max + 1):
        out.extend(candidates_for_joints(j))
    return out


def candidate_counts(j_min: int = config.J_MIN, j_max: int = config.J_MAX) -> dict:
    """{J: (graphs, candidates)} summary used for reporting."""
    return {j: (len(enumerate_graphs(j)), len(candidates_for_joints(j)))
            for j in range(j_min, j_max + 1)}


# --------------------------------------------------------------------------------------
# Helpers for later stages
# --------------------------------------------------------------------------------------
def candidate_graph(c: Candidate) -> nx.Graph:
    """The labelled graph of a candidate, with its role stored as a node attribute."""
    g = nx.Graph()
    g.add_nodes_from(range(c.n_joints))
    g.add_edges_from(c.members)
    roles = {config.PIN_INDEX: "end", config.ROLLER_INDEX: "end", config.LOAD_INDEX: "load"}
    for n in g.nodes:
        g.nodes[n]["role"] = roles.get(n, "free")
    return g


def min_total_length(c: Candidate) -> float:
    """Cheapest conceivable material use: every member at the minimum length."""
    return c.n_members * config.MEMBER_LENGTH_MIN


def fits_material_budget(c: Candidate) -> bool:
    """True if the members could be cut from the stock sticks even at the minimum length."""
    if c.n_members > config.STOCK_MEMBER_LIMIT:
        return False
    return packing.is_packable([config.MEMBER_LENGTH_MIN] * c.n_members)

