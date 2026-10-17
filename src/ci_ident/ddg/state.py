"""Static DDG structure and the mutable identification state built on it.

:class:`StaticDDG` holds everything about a graph that never changes during an
episode (topology, opcode, latency, area, topological order).  It is built
once per DOT file and shared read-only.  :class:`DDGState` holds the
per-episode state: the status of every node, the input/output sets of the
candidate under construction, the selection-closure flags used for convexity,
and the committed CIs.  A fresh :class:`DDGState` is created for every
episode, so no mutable container is ever shared between episodes or between
environments.

Validity of a candidate ``C`` (all enforced here, on the original DDG):

* admissibility -- only nodes in the pool (CI-admissible, not yet used) join;
* I/O bounds -- ``|In(C)| <= max_in`` and ``|Out(C)| <= max_out`` with
  ``In(C)`` the distinct producers outside ``C`` of values consumed in ``C``
  and ``Out(C)`` the nodes of ``C`` whose value is used outside ``C`` or that
  are sinks of the block (their value is live-out);
* convexity under contraction -- no node outside ``C`` lies on a path between
  two nodes of ``C``, where every committed CI is treated as one atomic
  super-node.  Contraction makes the convexity test also reject a candidate
  that would be both upstream and downstream of a committed CI, so the
  committed set stays acyclic as a whole (set-level acyclicity).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx

from ci_ident.costs import cycles
from ci_ident.ddg.status import (
    ALREADY_SELECTED_GROUP,
    FIXED_GROUP,
    INPUT_NODE_GROUP,
    INVALID_GROUP,
    NOT_SELECTED_GROUP,
    SELECTED_GROUP,
    NodeStatus,
)

_IO_FIXED = ALREADY_SELECTED_GROUP | INVALID_GROUP  # can never be absorbed into the candidate
_PERM_INPUT_SRC = INPUT_NODE_GROUP | ALREADY_SELECTED_GROUP | INVALID_GROUP


@dataclass(frozen=True)
class StaticDDG:
    """Immutable view of an annotated DDG (see :func:`ci_ident.ddg.graph.load_ddg`)."""

    graph: nx.DiGraph
    nodes: Tuple[str, ...]
    predecessors: Dict[str, Tuple[str, ...]]
    successors: Dict[str, Tuple[str, ...]]
    topo_order: Tuple[str, ...]
    latency: Dict[str, float]          # hardware latency per node
    area: Dict[str, float]
    op: Dict[str, str]
    eligible: Dict[str, bool]
    input_nodes: frozenset             # sources of the DDG
    output_nodes: frozenset            # sinks of the DDG
    total_area: float                  # A_ref (inadmissible nodes have area 0)
    sw_total: float                    # T_sw = sum of cycles(latency) = |V|

    @classmethod
    def from_graph(cls, g: nx.DiGraph) -> "StaticDDG":
        nodes = tuple(g.nodes())
        preds = {v: tuple(g.predecessors(v)) for v in nodes}
        succs = {v: tuple(g.successors(v)) for v in nodes}
        topo = tuple(nx.topological_sort(g))
        lat = {v: float(g.nodes[v]["hw_latency"]) for v in nodes}
        area = {v: float(g.nodes[v]["area"]) for v in nodes}
        op = {v: str(g.nodes[v]["op"]) for v in nodes}
        elig = {v: bool(g.nodes[v]["ci_eligible"]) for v in nodes}
        ins = frozenset(v for v in nodes if g.in_degree(v) == 0)
        outs = frozenset(v for v in nodes if g.out_degree(v) == 0)
        return cls(
            graph=g, nodes=nodes, predecessors=preds, successors=succs,
            topo_order=topo, latency=lat, area=area, op=op, eligible=elig,
            input_nodes=ins, output_nodes=outs,
            total_area=float(sum(area.values())),
            sw_total=float(sum(cycles(x) for x in lat.values())),
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def critical_path(self, ci_nodes) -> float:
        """Longest latency-weighted path inside the subgraph induced by ``ci_nodes``."""
        if not ci_nodes:
            return 0.0
        S = set(ci_nodes)
        L: Dict[str, float] = {}
        best_all = 0.0
        lat, preds = self.latency, self.predecessors
        for v in self.topo_order:
            if v not in S:
                continue
            best = lat[v]
            for p in preds[v]:
                pv = L.get(p)
                if pv is not None and lat[v] + pv > best:
                    best = lat[v] + pv
            L[v] = best
            if best > best_all:
                best_all = best
        return best_all


class DDGState:
    """Mutable identification state of one episode on a :class:`StaticDDG`."""

    def __init__(self, static: StaticDDG, max_in: int, max_out: int):
        self.static = static
        self.max_in = int(max_in)
        self.max_out = int(max_out)
        nodes = static.nodes

        self.pattern: Dict[str, NodeStatus] = {}
        self.nodes_pool: Set[str] = set()
        for n in nodes:
            if n in static.input_nodes:
                self.pattern[n] = NodeStatus.INPUT_NODE
            elif not static.eligible[n]:
                self.pattern[n] = NodeStatus.INVALID
            else:
                self.pattern[n] = NodeStatus.NOT_SELECTED
                self.nodes_pool.add(n)

        # I/O sets of the candidate under construction
        self.input_set: Set[str] = set()
        self.output_set: Set[str] = set()
        self.perm_input_set: Set[str] = set()
        self.perm_output_set: Set[str] = set()

        # selection-closure flags: node has a selected ancestor / descendant
        self._sel_anc: Dict[str, bool] = {n: False for n in nodes}
        self._sel_desc: Dict[str, bool] = {n: False for n in nodes}
        self._selected: List[str] = []

        # committed CIs, contracted to atomic groups for convexity
        self._committed_group: Dict[str, int] = {}
        self._group_members: Dict[int, List[str]] = {}

    # candidate lifecycle ---------------------------------------------------
    def reset_candidate(self) -> None:
        """Clear the candidate's I/O sets and closure flags; free nodes return
        to NOT_SELECTED, committed nodes to ALREADY_SELECTED."""
        self.input_set.clear()
        self.perm_input_set.clear()
        self.output_set.clear()
        self.perm_output_set.clear()
        for n in self.static.nodes:
            self._sel_anc[n] = False
            self._sel_desc[n] = False
        self._selected.clear()
        st = self.static
        for n in st.nodes:
            if n in st.input_nodes:
                base = NodeStatus.INPUT_NODE
            elif not st.eligible[n]:
                base = NodeStatus.INVALID
            elif self.pattern[n] in ALREADY_SELECTED_GROUP:
                base = NodeStatus.ALREADY_SELECTED
            else:
                base = NodeStatus.NOT_SELECTED
            self.pattern[n] = base

    def register_committed_group(self, nodes) -> None:
        """Record a committed CI as one atomic group (contraction convexity)."""
        members = [n for n in nodes if n in self._sel_anc]
        if not members:
            return
        gid = len(self._group_members)
        self._group_members[gid] = list(members)
        for n in members:
            self._committed_group[n] = gid

    # I/O accounting --------------------------------------------------------
    def update_io(self, node: str) -> dict:
        """Account for ``node`` having just been marked SELECTED; return an
        undo token for :meth:`rollback_io`."""
        patt = self.pattern
        preds = self.static.predecessors[node]
        succs = self.static.successors[node]
        delta = {
            "node": node, "pattern_prev": patt[node], "pat_changes": [],
            "in_add": set(), "in_rem": set(), "out_add": set(), "out_rem": set(),
            "pin_add": set(), "pin_rem": set(), "pout_add": set(), "pout_rem": set(),
        }
        pat_changes = delta["pat_changes"]

        def _set(u, status):
            prev = patt.get(u)
            if prev is not status:
                pat_changes.append((u, prev))
                patt[u] = status

        in_set, pin_set = self.input_set, self.perm_input_set
        out_set, pout_set = self.output_set, self.perm_output_set

        for s in succs:  # node is an output if any consumer is outside the candidate
            if patt[s] not in SELECTED_GROUP:
                if node not in out_set:
                    out_set.add(node)
                    delta["out_add"].add(node)
                break
        for s in succs:  # ... and a permanent output if a consumer can never join
            if patt[s] in _IO_FIXED:
                if node not in pout_set:
                    pout_set.add(node)
                    delta["pout_add"].add(node)
                break
        if node in self.static.output_nodes:  # block sinks are live-out
            if node not in out_set:
                out_set.add(node)
                delta["out_add"].add(node)
            if node not in pout_set:
                pout_set.add(node)
                delta["pout_add"].add(node)

        released_outputs = []
        for p in preds:
            pst = patt[p]
            if pst not in SELECTED_GROUP and p not in in_set:
                in_set.add(p)
                delta["in_add"].add(p)
            if pst in _PERM_INPUT_SRC and p not in pin_set:
                pin_set.add(p)
                delta["pin_add"].add(p)
            if p in out_set and p not in pout_set:
                if all(patt[s2] in SELECTED_GROUP for s2 in self.static.successors[p]):
                    out_set.remove(p)
                    delta["out_rem"].add(p)
                    released_outputs.append(p)
        if node in in_set:
            in_set.remove(node)
            delta["in_rem"].add(node)

        # status codes consistent with the sets
        for s in in_set:
            if s in pin_set:
                continue
            if patt.get(s) in NOT_SELECTED_GROUP:
                _set(s, NodeStatus.NOT_SELECTED_INPUT)
        for s in pin_set:
            if s in self.static.input_nodes:
                _set(s, NodeStatus.INPUT_NODE_PERM_INPUT)
            elif self.static.eligible[s]:
                _set(s, NodeStatus.ALREADY_SELECTED_PERM_INPUT)
            else:
                _set(s, NodeStatus.INVALID_PERM_INPUT)
        if node in out_set:
            _set(node, NodeStatus.SELECTED)
        for s in released_outputs:
            _set(s, NodeStatus.SELECTED)
        for s in out_set:
            if s not in pout_set:
                _set(s, NodeStatus.SELECTED_OUTPUT)
        for s in pout_set:
            _set(s, NodeStatus.SELECTED_PERM_OUTPUT)
        return delta

    def rollback_io(self, delta: dict) -> None:
        self.input_set.difference_update(delta["in_add"])
        self.input_set.update(delta["in_rem"])
        self.output_set.difference_update(delta["out_add"])
        self.output_set.update(delta["out_rem"])
        self.perm_input_set.difference_update(delta["pin_add"])
        self.perm_input_set.update(delta["pin_rem"])
        self.perm_output_set.difference_update(delta["pout_add"])
        self.perm_output_set.update(delta["pout_rem"])
        for u, prev in reversed(delta["pat_changes"]):
            self.pattern[u] = prev
        self.pattern[delta["node"]] = delta["pattern_prev"]

    def perm_io_ok(self) -> bool:
        return len(self.perm_input_set) <= self.max_in and len(self.perm_output_set) <= self.max_out

    def io_ok(self) -> bool:
        return len(self.input_set) <= self.max_in and len(self.output_set) <= self.max_out

    def estimate_io_after_add(self, node: str) -> Tuple[int, int, int, int]:
        """``(|In|, |Out|, |permIn|, |permOut|)`` if ``node`` were selected next."""
        patt = self.pattern
        in_set, pin_set = self.input_set, self.perm_input_set
        out_set, pout_set = self.output_set, self.perm_output_set
        n_in, n_out, n_pin, n_pout = len(in_set), len(out_set), len(pin_set), len(pout_set)
        succs = self.static.successors[node]

        if any(patt[s] not in SELECTED_GROUP for s in succs) and node not in out_set:
            n_out += 1
        if any(patt[s] in _IO_FIXED for s in succs) and node not in pout_set:
            n_pout += 1
        if node in self.static.output_nodes:
            if node not in out_set:
                n_out += 1
            if node not in pout_set:
                n_pout += 1
        for p in self.static.predecessors[node]:
            pst = patt[p]
            if pst not in SELECTED_GROUP and p not in in_set:
                n_in += 1
            if pst in _PERM_INPUT_SRC and p not in pin_set:
                n_pin += 1
            if p in out_set and p not in pout_set:
                if all(s2 == node or patt[s2] in SELECTED_GROUP
                       for s2 in self.static.successors[p]):
                    n_out -= 1
        if node in in_set:
            n_in -= 1
        return n_in, n_out, n_pin, n_pout

    # convexity under contraction -----------------------------------------
    def _expand(self, u: str):
        gid = self._committed_group.get(u)
        return self._group_members[gid] if gid is not None else ()

    def _propagate_from(self, s: str) -> bool:
        """Propagate closure flags from a newly selected ``s``; False if some
        fixed node now has both a selected ancestor and descendant."""
        patt = self.pattern
        q = deque()
        if not self._sel_anc[s]:
            self._sel_anc[s] = True
            q.append(s)
        while q:
            u = q.popleft()
            for m in self._expand(u):
                if not self._sel_anc[m]:
                    self._sel_anc[m] = True
                    if patt[m] in FIXED_GROUP and self._sel_desc[m]:
                        return False
                    q.append(m)
            for v in self.static.successors[u]:
                if not self._sel_anc[v]:
                    self._sel_anc[v] = True
                    if patt[v] in FIXED_GROUP and self._sel_desc[v]:
                        return False
                    q.append(v)
        q.clear()
        if not self._sel_desc[s]:
            self._sel_desc[s] = True
            q.append(s)
        while q:
            u = q.popleft()
            for m in self._expand(u):
                if not self._sel_desc[m]:
                    self._sel_desc[m] = True
                    if patt[m] in FIXED_GROUP and self._sel_anc[m]:
                        return False
                    q.append(m)
            for v in self.static.predecessors[u]:
                if not self._sel_desc[v]:
                    self._sel_desc[v] = True
                    if patt[v] in FIXED_GROUP and self._sel_anc[v]:
                        return False
                    q.append(v)
        return True

    def _recompute_flags(self) -> None:
        for n in self.static.nodes:
            self._sel_anc[n] = False
            self._sel_desc[n] = False
        for flags, adj in ((self._sel_anc, self.static.successors),
                           (self._sel_desc, self.static.predecessors)):
            q = deque()
            for s in self._selected:
                if not flags[s]:
                    flags[s] = True
                    q.append(s)
            while q:
                u = q.popleft()
                for m in self._expand(u):
                    if not flags[m]:
                        flags[m] = True
                        q.append(m)
                for v in adj[u]:
                    if not flags[v]:
                        flags[v] = True
                        q.append(v)

    def convexity_after_select(self, node: str) -> bool:
        """Incremental test after ``node`` has been tentatively selected."""
        if not self._selected or self._selected[-1] != node:
            self._selected.append(node)
        return self._propagate_from(node)

    def convexity_rollback(self, node: str) -> None:
        if self._selected and self._selected[-1] == node:
            self._selected.pop()
        self._recompute_flags()

    def convexity_ok(self) -> bool:
        """Global test: no non-candidate node between two candidate nodes."""
        patt = self.pattern
        for n in self.static.nodes:
            if patt[n] not in SELECTED_GROUP and self._sel_anc[n] and self._sel_desc[n]:
                return False
        return True

    def estimate_convexity_after_add(self, node: str) -> bool:
        """Exact, non-mutating: would selecting ``node`` keep the candidate
        convex under contraction?"""
        patt = self.pattern

        def fixed(u):
            return u != node and patt.get(u) not in SELECTED_GROUP

        for start, adj, other in ((self.static.successors, self.static.successors, self._sel_desc),
                                  (self.static.predecessors, self.static.predecessors, self._sel_anc)):
            first = start[node]
            if not first:
                continue
            q = deque(first)
            seen = set(first)
            while q:
                v = q.popleft()
                if fixed(v) and other.get(v, False):
                    return False
                for m in self._expand(v):
                    if fixed(m) and other.get(m, False):
                        return False
                    if m not in seen:
                        seen.add(m)
                        q.append(m)
                for w in adj[v]:
                    if w not in seen:
                        seen.add(w)
                        q.append(w)
        return True

    def convexity_reach_sets(self):
        """Candidate-independent part of :meth:`estimate_convexity_after_add`,
        computed once per decision state (O(V+E)) so that the per-candidate
        test is O(degree).  Exactness: see :meth:`convexity_ok_after_add`."""
        patt = self.pattern
        bad_d = {v for v, f in self._sel_desc.items() if f and patt.get(v) not in SELECTED_GROUP}
        bad_a = {u for u, f in self._sel_anc.items() if f and patt.get(u) not in SELECTED_GROUP}

        def closure(seeds, adj):
            seen = set(seeds)
            q = deque(seen)
            while q:
                x = q.popleft()
                for m in self._expand(x):
                    if m not in seen:
                        seen.add(m)
                        q.append(m)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
            return seen

        return (closure(bad_d, self.static.predecessors),
                closure(bad_a, self.static.successors), bad_d, bad_a)

    def convexity_ok_after_add(self, node: str, reach=None) -> bool:
        """:meth:`estimate_convexity_after_add`, answered from ``reach`` when given.

        A hit on ``reach`` is spurious only if the whole witness set is
        ``{node}``, which requires ``node`` to be in the corresponding bad set;
        that case defers to the exact test, so the answer is identical."""
        if reach is None:
            return self.estimate_convexity_after_add(node)
        r_d, r_a, bad_d, bad_a = reach
        for s in self.static.successors[node]:
            if s in r_d:
                return self.estimate_convexity_after_add(node) if node in bad_d else False
        for p in self.static.predecessors[node]:
            if p in r_a:
                return self.estimate_convexity_after_add(node) if node in bad_a else False
        return True
