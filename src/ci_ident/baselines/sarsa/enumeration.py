"""Exact enumeration of the CI design space (first stage of the SARSA baseline).

:func:`enumerate_candidates` lists every node set ``S`` of a DDG that is

* legal -- every node has ``ci_eligible`` set (the only legality source),
* convex -- every path between two nodes of ``S`` lies in ``S``,
* I/O-bounded -- ``|In(S)| <= max_in`` and ``|Out(S)| <= max_out``, where
  ``In(S)`` are the distinct nodes outside ``S`` with a successor in ``S`` and
  ``Out(S)`` the nodes of ``S`` with a successor outside ``S`` or with no
  successor (DDG sinks are outputs),
* connected (weakly), unless ``connected=False``,

i.e. the design space of the TACO paper (*Problem Formulation*).  The search runs in the C++
extension ``_xiao_enum``; its header comment gives the algorithm: depth-first
growth over the NodeFilter frontier of Xiao et al. (2014) with sibling
deletion, roots in topological order with each processed root excluded from
the later ones (every candidate is generated exactly once), and pruning by
permanent I/O only.  Every state is expanded whether or not its own I/O
counts are within the bounds, so the enumeration is exhaustive; it stops
early only at the deadline.

Nodes are indexed in a deterministic topological order (ties broken by the
canonical DOT order), so the output order is a function of the graph and the
bounds alone.  Parallel edges are merged (they change neither convexity nor
the distinct-node I/O counts).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Hashable, List

import networkx as nx

from ci_ident.baselines.sarsa import _ext
from ci_ident.costs import EPS

Node = Hashable


def topo_order(g: nx.DiGraph) -> List[Node]:
    """Topological order of ``g`` with ties broken by canonical (DOT) position."""
    pos = {n: i for i, n in enumerate(g.nodes())}
    return list(nx.lexicographical_topological_sort(g, key=pos.__getitem__))


class GraphIndex:
    """Dense, topologically ordered view of a DDG shared by both stages."""

    def __init__(self, g: nx.DiGraph):
        self.order: List[Node] = topo_order(g)
        self.idx = {n: i for i, n in enumerate(self.order)}
        self.edges = sorted({(self.idx[u], self.idx[v]) for u, v in g.edges()})
        self.eligible = [1 if g.nodes[n]["ci_eligible"] else 0 for n in self.order]
        self.hw = [float(g.nodes[n]["hw_latency"]) for n in self.order]
        self.area = [float(g.nodes[n]["area"]) for n in self.order]

    @property
    def num_nodes(self) -> int:
        return len(self.order)


def enumerate_candidates(g: nx.DiGraph, max_in: int, max_out: int, *, connected: bool = True,
                         seconds: float = math.inf, store_all: bool = False,
                         max_candidates: int = 0, index: GraphIndex = None) -> Dict[str, Any]:
    """Run the exact enumeration.

    ``seconds`` is the time limit from the call (``math.inf``: none; ``<= 0``:
    already expired, nothing is enumerated).  By default only candidates that
    can be CI instances are stored (``|S| >= 2`` and a positive whole-cycle
    saving ``|S| - ceil(cp)``); ``store_all=True`` keeps every candidate.
    ``max_candidates > 0`` stops the search once that many candidates are
    stored (``capped``); 0 means no bound.

    Returns a dict with ``index`` (:class:`GraphIndex`), ``nodes``/``offsets``
    (candidate ``k`` is ``nodes[offsets[k]:offsets[k+1]]``, topological
    indices, ascending), ``cp`` (longest hardware-latency path per stored
    candidate), ``valid`` (number of candidates found, stored or not),
    ``stored``, ``explored`` (search states), ``timed_out``, ``capped`` and
    ``elapsed_s``.
    """
    if int(max_in) < 0 or int(max_out) < 0:
        raise ValueError("I/O bounds must be non-negative")
    if int(max_candidates) < 0:
        raise ValueError("max_candidates must be >= 0")
    mod = _ext.load("_xiao_enum")
    gi = index if index is not None else GraphIndex(g)
    core = mod.Enumerator(gi.num_nodes, gi.edges, gi.eligible, gi.hw)
    secs = math.inf if seconds is None else float(seconds)
    res = core.run(int(max_in), int(max_out), bool(connected), secs, bool(store_all), float(EPS),
                   int(max_candidates))
    res = dict(res)
    res["index"] = gi
    return res


def candidate_sets(res: Dict[str, Any]) -> List[List[Node]]:
    """Node-name lists of the stored candidates of an :func:`enumerate_candidates` result."""
    order = res["index"].order
    nodes, off = res["nodes"], res["offsets"]
    return [[order[i] for i in nodes[off[k]:off[k + 1]]] for k in range(len(off) - 1)]
