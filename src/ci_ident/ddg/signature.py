"""Direction-aware 1-WL canonical signature of a CI (after the WL-signature algorithm of the TACO paper).

Two CIs with the same signature are treated as instances of one pattern: the
pattern's area is charged once and every instance contributes its cycle
saving.  The signature is computed on the subgraph induced by the CI, with
node colours initialised from (opcode, in-degree, out-degree, depth) inside
the CI and refined for five rounds over sorted predecessor and successor
colours.  The signature is the sorted multiset of final node colours together
with the sorted multiset of (producer colour, consumer colour) pairs of the
CI's edges, so it is invariant to the order of the nodes.  The same function
is used by the RL environment and the baselines.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Hashable, List, Sequence, Tuple

Signature = Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]

WL_ROUNDS = 5


def _h(parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


class SignatureIndex:
    """Index-space view of a DDG for repeated signature computations.

    ``pred``/``succ`` are adjacency lists over node indices (parallel edges
    kept) and ``op`` the opcode per index, all in canonical node order.
    """

    def __init__(self, nodes: Sequence[Hashable], pred: List[List[int]],
                 succ: List[List[int]], op: Sequence[str]):
        self.nodes = list(nodes)
        self.idx_of = {n: i for i, n in enumerate(self.nodes)}
        self.pred = pred
        self.succ = succ
        self.op = list(op)

    @classmethod
    def from_graph(cls, g) -> "SignatureIndex":
        nodes = list(g.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        pred: List[List[int]] = [[] for _ in nodes]
        succ: List[List[int]] = [[] for _ in nodes]
        for u, v in g.edges():
            succ[idx[u]].append(idx[v])
            pred[idx[v]].append(idx[u])
        op = [g.nodes[n].get("op", "") for n in nodes]
        return cls(nodes, pred, succ, op)

    def signature(self, ci_nodes: Sequence[Hashable]) -> Signature:
        """Canonical signature of the CI given by node names."""
        if not ci_nodes:
            return ((), ())
        S = [self.idx_of[n] for n in ci_nodes if n in self.idx_of]
        if not S:
            return ((), ())
        pred, succ, op = self.pred, self.succ, self.op
        inS = set(S)

        indeg = {u: sum(1 for p in pred[u] if p in inS) for u in S}
        outdeg = {u: sum(1 for v in succ[u] if v in inS) for u in S}

        # depth inside the CI (longest path from a CI source, in edges)
        work = dict(indeg)
        lvl = {u: 0 for u in S}
        q = deque(u for u in S if work[u] == 0)
        while q:
            u = q.popleft()
            for v in succ[u]:
                if v in inS:
                    if lvl[u] + 1 > lvl[v]:
                        lvl[v] = lvl[u] + 1
                    work[v] -= 1
                    if work[v] == 0:
                        q.append(v)

        color = {u: _h(("0", op[u], f"in{indeg[u]}", f"out{outdeg[u]}", f"lvl{lvl[u]}"))
                 for u in S}
        for _ in range(WL_ROUNDS):
            new = {}
            for u in S:
                pc = sorted(color[p] for p in pred[u] if p in inS)
                sc = sorted(color[v] for v in succ[u] if v in inS)
                new[u] = _h(("1", color[u], "[" + ",".join(pc) + "]", "[" + ",".join(sc) + "]"))
            color = new

        # Both parts are multisets over colours, so the signature does not
        # depend on the order in which the CI's nodes are given.
        colours = tuple(sorted(color[u] for u in S))
        edges = tuple(sorted((color[u], color[v]) for u in S for v in succ[u] if v in inS))
        return colours, edges
