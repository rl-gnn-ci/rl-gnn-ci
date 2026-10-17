"""Independent checks of a set of CIs on the original DDG (no environment code)."""
import math

import networkx as nx


def io_counts(g, ci):
    ci = set(ci)
    ins = {p for v in ci for p in g.predecessors(v) if p not in ci}
    outs = {v for v in ci if g.out_degree(v) == 0 or any(s not in ci for s in g.successors(v))}
    return len(ins), len(outs)


def is_convex(g, ci):
    ci = set(ci)
    # a path leaving ci and re-entering it
    frontier = {s for v in ci for s in g.successors(v) if s not in ci}
    seen = set(frontier)
    while frontier:
        u = frontier.pop()
        for s in g.successors(u):
            if s in ci:
                return False
            if s not in seen:
                seen.add(s)
                frontier.add(s)
    return True


def contraction_is_dag(g, cis):
    rep = {}
    for i, ci in enumerate(cis):
        for v in ci:
            assert v not in rep, f"node {v} in two CIs"
            rep[v] = ("ci", i)
    h = nx.DiGraph()
    h.add_nodes_from(rep.get(v, v) for v in g.nodes())
    for u, v in g.edges():
        a, b = rep.get(u, u), rep.get(v, v)
        if a != b:
            h.add_edge(a, b)
    return nx.is_directed_acyclic_graph(h)


def critical_path(g, ci):
    ci = set(ci)
    sub = g.subgraph(ci)
    best = {}
    for v in nx.topological_sort(sub):
        best[v] = g.nodes[v]["hw_latency"] + max((best[p] for p in sub.predecessors(v)), default=0.0)
    return max(best.values()) if best else 0.0


def saving(g, ci):
    # whole cycles, with the tolerance of the specification (costs.cycles)
    return len(ci) - math.ceil(critical_path(g, ci) - 1e-9)


def check_solution(g, cis, max_in, max_out, area_budget, signature=None, connected=False):
    """Assert validity; return (speedup, area_used)."""
    for ci in cis:
        if connected:
            assert nx.is_weakly_connected(g.subgraph(ci)), "disconnected CI"
        assert all(g.nodes[v]["ci_eligible"] for v in ci), "inadmissible node in a CI"
        n_in, n_out = io_counts(g, ci)
        assert n_in <= max_in and n_out <= max_out, (n_in, n_out, max_in, max_out)
        assert is_convex(g, ci), "non-convex CI"
    assert contraction_is_dag(g, cis), "committed CIs are mutually dependent"
    saved = sum(saving(g, ci) for ci in cis)
    n = g.number_of_nodes()
    speedup = n / (n - saved)
    if signature is not None:
        area = {}
        for ci in cis:
            area.setdefault(signature(ci), sum(g.nodes[v]["area"] for v in ci))
        used = sum(area.values())
        assert used <= area_budget + 2e-9, (used, area_budget)
    else:
        used = None
    return speedup, used
