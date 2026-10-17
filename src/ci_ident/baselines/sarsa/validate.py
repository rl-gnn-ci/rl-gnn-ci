"""Checks of a CI selection on the original DDG, independent of both stages.

Used by :func:`ci_ident.baselines.sarsa.identify_sarsa` as a guard (an
invalid selection is never returned) and by the tests.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, List, Sequence

import networkx as nx

from ci_ident.costs import EPS, cycles

Node = Hashable


def io_counts(g: nx.DiGraph, S) -> tuple:
    S = set(S)
    ins = {p for v in S for p in g.predecessors(v) if p not in S}
    outs = {v for v in S if g.out_degree(v) == 0 or any(s not in S for s in g.successors(v))}
    return len(ins), len(outs)


def is_convex(g: nx.DiGraph, S) -> bool:
    S = set(S)
    frontier = {s for v in S for s in g.successors(v) if s not in S}
    seen = set(frontier)
    while frontier:
        u = frontier.pop()
        for s in g.successors(u):
            if s in S:
                return False
            if s not in seen:
                seen.add(s)
                frontier.add(s)
    return True


def critical_path(g: nx.DiGraph, S) -> float:
    sub = g.subgraph(S)
    best: Dict[Node, float] = {}
    for v in nx.topological_sort(sub):
        best[v] = float(g.nodes[v]["hw_latency"]) + max((best[p] for p in sub.predecessors(v)), default=0.0)
    return max(best.values()) if best else 0.0


def saving(g: nx.DiGraph, S) -> int:
    """Whole-cycle saving ``|S| - cycles(cp(S))`` of one instance."""
    return len(S) - cycles(critical_path(g, S))


def contraction_is_acyclic(g: nx.DiGraph, cis: Sequence[Sequence[Node]]) -> bool:
    rep: Dict[Node, Any] = {}
    for i, S in enumerate(cis):
        for v in S:
            rep[v] = ("ci", i)
    h = nx.DiGraph()
    h.add_nodes_from(rep.get(v, v) for v in g.nodes())
    for u, v in g.edges():
        a, b = rep.get(u, u), rep.get(v, v)
        if a != b:
            h.add_edge(a, b)
    return nx.is_directed_acyclic_graph(h)


def check_selection(g: nx.DiGraph, cis: Sequence[Sequence[Node]], max_in: int, max_out: int,
                    connected: bool = True, area_budget: float = None,
                    signature: Callable[[Sequence[Node]], Any] = None) -> Dict[str, Any]:
    """Validate a selection; returns ``{"ok": bool, "errors": [...], "area_used": float|None}``."""
    errors: List[str] = []
    owner: Dict[Node, int] = {}
    for i, S in enumerate(cis):
        if len(set(S)) != len(S) or not S:
            errors.append(f"instance {i}: empty or repeated nodes")
        for v in S:
            if v in owner:
                errors.append(f"node {v!r} in instances {owner[v]} and {i}")
            owner[v] = i
        if not all(g.nodes[v]["ci_eligible"] for v in S):
            errors.append(f"instance {i}: inadmissible node")
        n_in, n_out = io_counts(g, S)
        if n_in > max_in or n_out > max_out:
            errors.append(f"instance {i}: I/O ({n_in},{n_out}) exceeds ({max_in},{max_out})")
        if not is_convex(g, S):
            errors.append(f"instance {i}: not convex")
        if connected and S and not nx.is_weakly_connected(g.subgraph(S)):
            errors.append(f"instance {i}: not connected")
        if saving(g, S) <= 0:
            errors.append(f"instance {i}: no saving")
    if not errors and not contraction_is_acyclic(g, cis):
        errors.append("contracting the selection creates a cycle")
    used = None
    if signature is not None:
        area: Dict[Any, float] = {}
        for S in cis:
            area.setdefault(signature(S), float(sum(g.nodes[v]["area"] for v in S)))
        used = float(sum(area.values()))
        if area_budget is not None and used > area_budget + EPS:
            errors.append(f"area {used} exceeds budget {area_budget}")
    return {"ok": not errors, "errors": errors, "area_used": used}
