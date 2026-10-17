"""Independent validity checks for custom instructions (CIs) on the ORIGINAL DDG.

Every function here reads the annotated, uncontracted graph returned by
:func:`ci_ident.ddg.graph.load_ddg` and never mutates it.  The checks share no
code with the MILP models in :mod:`.enumerator` and :mod:`.selector`, so they
can be used to verify those models' output.

Semantics (the repository's, identical to the RL environment's):

* ``In(S)``  -- the distinct nodes outside ``S`` with an edge into ``S``;
* ``Out(S)`` -- the nodes of ``S`` with a successor outside ``S``, or that are
  sinks of the DDG (their value is live-out of the block);
* convexity -- no node outside ``S`` lies on a directed path between two
  nodes of ``S``;
* connectivity -- ``S`` induces a weakly connected subgraph;
* admissibility -- every node of ``S`` has ``ci_eligible == True``;
* set-level acyclicity -- contracting every CI of a selection into one
  super-node leaves a DAG, i.e. the selected CIs can be scheduled.

Cost model: every node has a software latency of one cycle, so a CI ``S``
saves ``|S| - cycles(cp(S))`` cycles, where ``cp(S)`` is the longest path of
hardware latencies inside the subgraph induced by ``S`` and
:func:`ci_ident.costs.cycles` is ``ceil`` up to :data:`ci_ident.costs.EPS`
(the rounding the RL environment uses).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ci_ident.costs import cycles

Node = Hashable


# ----------------------------------------------------------------- single CI


def io_sets(g: nx.DiGraph, nodes: Iterable[Node]) -> Tuple[Set[Node], Set[Node]]:
    """``(In(S), Out(S))`` on ``g`` (see the module docstring)."""
    S = set(nodes)
    inputs: Set[Node] = set()
    outputs: Set[Node] = set()
    for v in S:
        for u in g.predecessors(v):
            if u not in S:
                inputs.add(u)
        succ = list(g.successors(v))
        if not succ or any(w not in S for w in succ):
            outputs.add(v)
    return inputs, outputs


def io_counts(g: nx.DiGraph, nodes: Iterable[Node]) -> Tuple[int, int]:
    """``(|In(S)|, |Out(S)|)``."""
    i, o = io_sets(g, nodes)
    return len(i), len(o)


def is_convex(g: nx.DiGraph, nodes: Iterable[Node]) -> bool:
    """True iff no node outside ``S`` is both reachable from ``S`` and reaches ``S``."""
    S = set(nodes)
    if len(S) <= 1:
        return True
    # outside nodes reachable from S
    fwd: Set[Node] = set()
    dq = deque(S)
    while dq:
        u = dq.popleft()
        for w in g.successors(u):
            if w not in S and w not in fwd:
                fwd.add(w)
                dq.append(w)
    if not fwd:
        return True
    # does any of them reach S again?
    dq = deque(fwd)
    seen = set(fwd)
    while dq:
        u = dq.popleft()
        for w in g.successors(u):
            if w in S:
                return False
            if w not in seen:
                seen.add(w)
                dq.append(w)
    return True


def is_weakly_connected(g: nx.DiGraph, nodes: Iterable[Node]) -> bool:
    S = set(nodes)
    if len(S) <= 1:
        return True
    start = next(iter(S))
    seen = {start}
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for w in list(g.successors(u)) + list(g.predecessors(u)):
            if w in S and w not in seen:
                seen.add(w)
                dq.append(w)
    return len(seen) == len(S)


def is_admissible(g: nx.DiGraph, nodes: Iterable[Node]) -> bool:
    return all(n in g and bool(g.nodes[n].get("ci_eligible", False)) for n in nodes)


def critical_path(g: nx.DiGraph, nodes: Iterable[Node]) -> float:
    """Longest path of ``hw_latency`` inside the subgraph induced by ``nodes``."""
    S = set(nodes)
    if not S:
        return 0.0
    H = g.subgraph(S)
    dist: Dict[Node, float] = {}
    for v in nx.topological_sort(H):
        best = 0.0
        for p in H.predecessors(v):
            if dist[p] > best:
                best = dist[p]
        dist[v] = best + float(g.nodes[v]["hw_latency"])
    return max(dist.values())


def saving(g: nx.DiGraph, nodes: Sequence[Node]) -> int:
    """Cycles saved by executing ``nodes`` as one CI: ``|S| - cycles(cp(S))``."""
    S = set(nodes)
    if not S:
        return 0
    sw = sum(float(g.nodes[n]["sw_latency"]) for n in S)
    return int(round(sw)) - cycles(critical_path(g, S))


def ci_area(g: nx.DiGraph, nodes: Iterable[Node]) -> float:
    return float(sum(float(g.nodes[n]["area"]) for n in set(nodes)))


def check_ci(g: nx.DiGraph, nodes: Sequence[Node], max_in: int, max_out: int,
             connected: bool = True) -> List[str]:
    """Violated properties of one CI (empty list = valid)."""
    S = set(nodes)
    reasons: List[str] = []
    if not S:
        return ["empty"]
    if len(S) != len(nodes):
        reasons.append("duplicate_nodes")
    if any(n not in g for n in S):
        return reasons + ["unknown_node"]
    if not is_admissible(g, S):
        reasons.append("inadmissible")
    n_in, n_out = io_counts(g, S)
    if n_in > max_in:
        reasons.append("inputs")
    if n_out > max_out:
        reasons.append("outputs")
    if not is_convex(g, S):
        reasons.append("nonconvex")
    if connected and not is_weakly_connected(g, S):
        reasons.append("disconnected")
    return reasons


# ------------------------------------------------------------ sets of CIs


def contract(g: nx.DiGraph, cis: Sequence[Iterable[Node]]) -> nx.DiGraph:
    """Simple digraph in which every CI of ``cis`` is one super-node
    ``("__ci__", k)``.  The CIs must be pairwise disjoint."""
    owner: Dict[Node, int] = {}
    for k, S in enumerate(cis):
        for n in S:
            if n in owner:
                raise ValueError(f"node {n!r} is in CI {owner[n]} and CI {k}")
            owner[n] = k

    def rep(n):
        k = owner.get(n)
        return n if k is None else ("__ci__", k)

    H = nx.DiGraph()
    H.add_nodes_from(rep(n) for n in g.nodes())
    for u, v in g.edges():
        ru, rv = rep(u), rep(v)
        if ru != rv:
            H.add_edge(ru, rv)
    return H


def find_contracted_cycle(g: nx.DiGraph, cis: Sequence[Iterable[Node]]) -> Optional[List[int]]:
    """Indices of the CIs on one cycle of the contracted graph, or ``None``
    if the contracted graph is a DAG."""
    if not cis:
        return None
    H = contract(g, cis)
    try:
        cyc = nx.find_cycle(H)
    except nx.NetworkXNoCycle:
        return None
    ks = sorted({n[1] for e in cyc for n in e[:2]
                 if isinstance(n, tuple) and len(n) == 2 and n[0] == "__ci__"})
    return ks


def set_is_acyclic(g: nx.DiGraph, cis: Sequence[Iterable[Node]]) -> bool:
    """Set-level acyclicity: the graph with every CI contracted is a DAG."""
    return find_contracted_cycle(g, cis) is None


def descendant_masks(g: nx.DiGraph) -> Tuple[Dict[Node, int], Dict[Node, int]]:
    """``(bit, desc)``: one bit per node and, per node, the bitmask of all
    nodes reachable from it (itself excluded)."""
    order = list(nx.topological_sort(g))
    bit = {n: 1 << i for i, n in enumerate(order)}
    desc: Dict[Node, int] = {}
    for u in reversed(order):
        m = 0
        for w in g.successors(u):
            m |= bit[w] | desc[w]
        desc[u] = m
    return bit, desc


def mutually_dependent_pairs(g: nx.DiGraph, cis: Sequence[Iterable[Node]]) -> List[Tuple[int, int]]:
    """Pairs ``(i, j)``, ``i < j``, of node-disjoint CIs such that ``i``
    reaches ``j`` and ``j`` reaches ``i`` on ``g``.  For convex CIs every such
    path leaves the source CI before entering the other, so these are exactly
    the pairs whose joint contraction creates a cycle."""
    bit, desc = descendant_masks(g)
    masks, reach = [], []
    for S in cis:
        m = r = 0
        for n in S:
            m |= bit[n]
            r |= desc[n]
        masks.append(m)
        reach.append(r)
    out = []
    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            if masks[i] & masks[j]:
                continue
            if (reach[i] & masks[j]) and (reach[j] & masks[i]):
                out.append((i, j))
    return out


def check_selection(g: nx.DiGraph, cis: Sequence[Sequence[Node]], max_in: int, max_out: int,
                    connected: bool = True) -> Dict[str, object]:
    """Full check of a selection: per-CI validity, node-disjointness and
    set-level acyclicity.  ``ok`` is True iff everything holds."""
    per_ci = {k: r for k, S in enumerate(cis) if (r := check_ci(g, S, max_in, max_out, connected))}
    seen: Dict[Node, int] = {}
    overlaps = []
    for k, S in enumerate(cis):
        for n in S:
            if n in seen:
                overlaps.append((seen[n], k, n))
            else:
                seen[n] = k
    cycle = None if overlaps else find_contracted_cycle(g, cis)
    return {
        "ok": not per_ci and not overlaps and cycle is None,
        "invalid_cis": per_ci,
        "overlaps": overlaps,
        "cycle": cycle,
    }
