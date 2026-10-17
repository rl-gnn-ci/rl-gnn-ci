"""Stage 1 of the ILP baseline: iterative ILP-IO extraction of CI candidates.

Model
-----
One MILP per extraction round, after Atasu et al.'s ILP-IO:

* ``x_i`` selects node ``i``; inadmissible nodes, contracted CIs and nodes
  whose area exceeds the per-CI cap are forced to 0;
* inputs  ``sum b_i + sum c_s <= max_in`` (``b``: unselected non-source with
  a selected successor, ``c``: source with a selected successor);
* outputs ``sum_{sinks} x_i + sum d_i <= max_out`` (``d``: selected node with
  an unselected successor; a selected DDG sink is an output);
* convexity by the ancestor/descendant recursion ``A_i + D_i <= x_i + 1``;
* connectivity (``connected=True``) by a single-root spanning tree with depth
  variables;
* timing: ``l_i`` arrival times on the transitive reduction, integer ``L``
  bounding every exit, objective ``max sum sw_i x_i - L - lambda * area``;
* per-CI area cap ``sum a_i x_i <= f * A_budget``.

After a round the chosen CI is contracted into one super-node and the next
round runs on the contracted graph, until no CI with positive gain remains.
Each pass of the (cap fraction ``f``) x (``lambda``) sweep starts from the
original graph; the lambda grid is derived from the graph
(:func:`lambda_grid_from_graph`).  Candidates are de-duplicated by node set across passes.

The cap is a fraction of the *global* budget ``A_budget = area_factor *
A_ref``: ``f`` in {1/3, 2/3, 1} by default.

Port counts after contraction
-----------------------------
A contracted super-node ``A`` that feeds a later candidate ``S`` enters
the input sum of the model above through one binary ``b_A``.  On the
original DDG, ``In(S)`` counts every distinct outside predecessor, so a
candidate that consumes two original outputs of ``A`` has two inputs from
``A``.  The enumerator counts inputs exactly with
respect to the original DDG, inside the model.  A *split* super-node
replaces ``b_A`` in the input sum by one binary ``w_{A,o}`` per original
output ``o`` of ``A`` that still has a successor in the graph, with
``w_{A,o} >= x_j`` and ``w_{A,o} <= sum_j x_j`` over those successors
``j``.  With every super-node split, the model's input count equals
``|In(S)|`` on the original DDG.  Outputs, convexity and connectivity of a
candidate are the same on the contracted and the original graph
(convexity on the contracted graph is stronger: it also forbids a cycle
through an earlier CI of the same pass, which those CIs need to be
co-schedulable).

Splitting is lazy.  A round is first solved with one binary per
super-node; if the candidate exceeds ``max_in`` on the original DDG and
consumes two or more original outputs of some super-nodes, those
super-nodes are split (for the rest of the pass) and the round is
re-solved.  The accepted candidate is an optimum of the fully split
(exact) model: the partially split model is a relaxation of it, and the
accepted optimum is feasible for it.

Every extracted candidate is also checked on the original DDG
(:func:`.validate.check_ci`).  A failing candidate is never emitted; it is
recorded in ``rejected`` and excluded by a no-good cut
(``sum_{i in C} x_i - sum_{i notin C} x_i <= |C| - 1``) before the round is
re-solved.

Rounding and tolerances
-----------------------
Critical paths are rounded to whole cycles with :func:`ci_ident.costs.cycles`
(``ceil`` up to :data:`ci_ident.costs.EPS`), as in the RL environment, so
both methods agree at exact boundaries (e.g. a path of latencies summing to
``1.0000000000000002`` takes one cycle; inside the MILP the integer ``L``
reaches the same value through the solver's feasibility tolerance).  Nodes
are excluded by the per-CI cap only if their area exceeds it by more than
:data:`ci_ident.costs.EPS`.  The cap row itself keeps the original
right-hand side ``f * A_budget``; the solver's feasibility tolerance admits
a sum equal to the cap up to float rounding.

Time handling
-------------
All solves share the caller's :class:`Deadline` (one monotonic clock for
the scenario).  The remaining enumeration time is split fairly over the
remaining (cap, lambda) passes; every solve gets ``min(pass deadline) -
now`` as its time limit and no solve is started after the deadline.  Every solve
appends a diagnostics record (status, wall time, primal, bound, gap, ...).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, FrozenSet, Hashable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from ci_ident.baselines.ilp import validate
from ci_ident.costs import EPS, cycles

logger = logging.getLogger(__name__)

HAVE_CPLEX = False
try:  # pragma: no cover - CPLEX is optional and not available in CI
    from docplex.mp.model import Model as CpxModel

    HAVE_CPLEX = True
except Exception:  # noqa: BLE001
    CpxModel = None

HAVE_SCIP = False
try:
    from pyscipopt import Model as ScipModel
    from pyscipopt import quicksum as scip_quicksum

    HAVE_SCIP = True
except Exception:  # noqa: BLE001
    ScipModel = None
    scip_quicksum = None

Node = Hashable


# ------------------------------------------------------------------ helpers


class Deadline:
    """An absolute point on the monotonic clock.  ``seconds=None`` or ``inf``
    means no limit; ``seconds <= 0`` is already expired."""

    def __init__(self, seconds: Optional[float], start: Optional[float] = None):
        self.start = time.monotonic() if start is None else float(start)
        if seconds is None or math.isinf(float(seconds)):
            self.at = math.inf
        else:
            self.at = self.start + max(0.0, float(seconds))

    @classmethod
    def absolute(cls, at: float) -> "Deadline":
        d = cls(None)
        d.at = float(at)
        return d

    def remaining(self) -> float:
        return self.at - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    @property
    def bounded(self) -> bool:
        return not math.isinf(self.at)


def available_solvers() -> List[str]:
    out = []
    if HAVE_CPLEX:
        out.append("cplex")
    if HAVE_SCIP:
        out.append("scip")
    return out


def resolve_solver(solver: str = "auto") -> str:
    """``"auto"`` prefers CPLEX (docplex) and falls back to SCIP (PySCIPOpt)."""
    s = (solver or "auto").lower()
    if s == "auto":
        if HAVE_CPLEX:
            return "cplex"
        if HAVE_SCIP:
            return "scip"
        raise RuntimeError("No MILP solver found: install docplex (CPLEX) or pyscipopt (SCIP).")
    if s == "cplex" and not HAVE_CPLEX:
        raise RuntimeError("CPLEX requested but docplex is not importable.")
    if s == "scip" and not HAVE_SCIP:
        raise RuntimeError("SCIP requested but pyscipopt is not importable.")
    if s not in ("cplex", "scip"):
        raise ValueError(f"unknown solver {solver!r}")
    return s


def _finite_or_none(v) -> Optional[float]:
    try:
        f = float(v)
    except Exception:  # noqa: BLE001
        return None
    return f if math.isfinite(f) and abs(f) < 1e19 else None


def simple_digraph(g: nx.DiGraph) -> nx.DiGraph:
    """Fresh simple :class:`networkx.DiGraph` with the cost attributes of
    ``g`` (node order and edge order preserved, parallel edges merged).
    ``g`` itself is not modified."""
    H = nx.DiGraph()
    for n, d in g.nodes(data=True):
        H.add_node(n, op=d.get("op", ""), hw_latency=float(d["hw_latency"]),
                   sw_latency=float(d["sw_latency"]), area=float(d["area"]),
                   ci_eligible=bool(d.get("ci_eligible", False)))
    H.add_edges_from((u, v) for u, v in g.edges())
    return H


def _topo_cp_latency(G: nx.DiGraph, nodes, latency_attr: str = "hw_latency") -> float:
    """Critical-path latency on the induced subgraph ``G[nodes]``."""
    S = G.subgraph(set(nodes)).copy()
    if len(S) == 0:
        return 0.0
    order = list(nx.topological_sort(S))
    dist = {u: float(S.nodes[u][latency_attr]) for u in S.nodes}
    for u in order:
        for v in S.successors(u):
            cand = dist[u] + float(S.nodes[v][latency_attr])
            if cand > dist[v]:
                dist[v] = cand
    return max(dist.values())


def _transitive_reduction_edges(G: nx.DiGraph) -> List[Tuple[Node, Node]]:
    try:
        tr = set(nx.transitive_reduction(G).edges())
        return [e for e in G.edges() if e in tr]  # deterministic (graph edge order)
    except Exception:  # noqa: BLE001  pragma: no cover
        logger.warning("transitive reduction failed; using all edges for precedence")
        return list(G.edges())


def lambda_grid_from_graph(G: nx.DiGraph, k: int = 3) -> List[float]:
    """Area-penalty grid: quantiles of the per-node saving density
    ``(sw - hw) / area``."""
    dens = []
    for _, dn in G.nodes(data=True):
        sw = float(dn.get("sw_latency", 0.0))
        hw = float(dn.get("hw_latency", 0.0))
        ar = float(dn.get("area", 0.0))
        if ar > 0 and sw > hw:
            dens.append((sw - hw) / ar)
    if not dens:
        return [0.0]
    d = np.array(dens)
    mu, p90, dmax = np.median(d), np.percentile(d, 90), np.max(d)
    base = [0.0, 0.25 * mu, 0.5 * mu, 1.0 * mu, 1.0 * p90, 2.0 * p90]
    lam_max = 1.2 * dmax
    grid = sorted({float(min(max(0.0, x), lam_max)) for x in base})
    if k and len(grid) > k:
        mid = np.geomspace(max(grid[1], 1e-12), grid[-2], num=max(0, k - 2))
        grid = [0.0] + [float(v) for v in mid] + [grid[-1]]
    return grid


# --------------------------------------------------------------- enumerator


class ILPEnumerator:
    """ILP-IO candidate extraction over the (cap fraction x lambda) sweep.

    ``g`` is the annotated original DDG; it is copied, never modified.
    ``area_max`` is the global budget ``area_factor * A_ref``; pass ``f``
    caps each candidate's area at ``f * area_max``.
    """

    def __init__(
        self,
        g: nx.DiGraph,
        max_in: int,
        max_out: int,
        area_max: Optional[float],
        area_cap_fracs: Sequence[float] = (1 / 3, 2 / 3, 1.0),
        *,
        connected: bool = True,
        solver: str = "auto",
        deadline: Optional[Deadline] = None,
        threads: int = 1,
        lambda_area: Optional[float] = None,
        lambda_grid: Optional[Sequence[float]] = None,
        seed: int = 0,
    ):
        self.g_orig = g
        self.G0 = simple_digraph(g)
        if not nx.is_directed_acyclic_graph(self.G0):
            raise ValueError("DDG must be a DAG")
        self.invalid0: Set[Node] = {n for n, d in self.G0.nodes(data=True) if not d["ci_eligible"]}
        self.max_in = int(max_in)
        self.max_out = int(max_out)
        self.area_max = None if area_max is None else float(area_max)
        self.area_cap_fracs = [float(f) for f in (area_cap_fracs or [1.0])]
        self.connected = bool(connected)
        self.solver = resolve_solver(solver)
        self.deadline = deadline if deadline is not None else Deadline(None)
        self.threads = max(1, int(threads or 1))
        self.lambda_area = lambda_area
        self.lambda_grid = list(lambda_grid) if lambda_grid else None
        self.seed = int(seed or 0)
        logger.info("ILP enumerator: solver=%s nodes=%d caps=%s", self.solver,
                    self.G0.number_of_nodes(), self.area_cap_fracs)

        self.diagnostics: List[Dict[str, Any]] = []
        self.rejected: List[Dict[str, Any]] = []
        self.truncated = False  # a deadline cut a pass short
        self.lazy_splits = 0  # super-nodes split for exact input counts
        self._ctx: Dict[str, Any] = {}

    # ------------------------------------------------------------ statics

    def _static(self, G: nx.DiGraph, agg_outputs: Dict[Node, List[Node]],
                split_active: Set[Node]) -> Dict[str, Any]:
        nodes_topo = list(nx.topological_sort(G))
        I2 = [n for n in nodes_topo if G.in_degree(n) == 0]
        I2s = set(I2)
        I1 = [n for n in nodes_topo if n not in I2s]
        I3 = [n for n in I1 if G.out_degree(n) == 0]
        I3s = set(I3)
        I4 = [n for n in I1 if n not in I3s]

        succ_all = {i: list(G.successors(i)) for i in G.nodes}
        pred_all = {i: list(G.predecessors(i)) for i in G.nodes}

        # undirected neighbours on I1 (insertion-ordered, deterministic)
        I1s = set(I1)
        neighbors: Dict[Node, Dict[Node, None]] = {i: {} for i in I1}
        for i in I1:
            for j in succ_all[i]:
                if j in I1s:
                    neighbors[i][j] = None
                    neighbors[j][i] = None
            for j in pred_all[i]:
                if j in I1s:
                    neighbors[i][j] = None
                    neighbors[j][i] = None

        U_end: Dict[Node, float] = {}
        for u in nodes_topo:
            hu = float(G.nodes[u]["hw_latency"])
            U_end[u] = hu if G.in_degree(u) == 0 else max(U_end[p] for p in pred_all[u]) + hu

        # split super-nodes get one input binary per original output
        split: Dict[Node, List[List[Node]]] = {}
        for a, outs in agg_outputs.items():
            if a not in G or a not in split_active:
                continue
            groups = []
            for o in outs:
                cons = [s for s in self.G0.successors(o) if s in G and s in I1s]
                if cons:
                    groups.append(cons)
            if len(groups) >= 2:
                split[a] = groups

        return dict(G=G, I1=I1, I2=I2, I3=I3, I4=I4, succ_all=succ_all, pred_all=pred_all,
                    neighbors=neighbors, U_end=U_end, prec_edges=_transitive_reduction_edges(G),
                    split=split)

    # ------------------------------------------------------------ diagnostics

    def _record(self, **values: Any) -> None:
        self.diagnostics.append({"phase": "enum", **self._ctx, **values})

    # ------------------------------------------------------------ SCIP

    def _solve_scip(self, S, lam, amax, invalid, nogoods, time_limit):
        m = ScipModel("ILP_IO_iter")
        m.hideOutput()
        try:
            m.setIntParam("display/verblevel", 0)
        except Exception:  # noqa: BLE001
            pass
        for p, v in (("parallel/maxnthreads", self.threads), ("lp/threads", self.threads)):
            try:
                m.setIntParam(p, v)
            except Exception:  # noqa: BLE001
                pass
        if time_limit is not None and math.isfinite(time_limit):
            m.setRealParam("limits/time", float(time_limit))
        for p, v in (("limits/gap", 0.0), ("limits/absgap", 0.0)):
            try:
                m.setRealParam(p, float(v))
            except Exception:  # noqa: BLE001
                pass
        if self.seed:
            m.setIntParam("randomization/randomseedshift", self.seed)

        G: nx.DiGraph = S["G"]
        I1_nodes = list(S["I1"])
        if not I1_nodes:
            return [], 0.0, None
        I2_nodes, I3_nodes, I4_nodes = list(S["I2"]), list(S["I3"]), list(S["I4"])
        succ_all, pred_all = S["succ_all"], S["pred_all"]
        neighbors, U_end, prec_edges = S["neighbors"], S["U_end"], S["prec_edges"]
        split = S["split"]

        idx_of = {n: k for k, n in enumerate(I1_nodes)}
        node_of = {k: n for n, k in idx_of.items()}
        nI1 = len(I1_nodes)
        I3 = [idx_of[n] for n in I3_nodes if n in idx_of]
        I4 = [idx_of[n] for n in I4_nodes if n in idx_of]

        sw = [0.0] * nI1
        hw = [0.0] * nI1
        ar = [0.0] * nI1
        Ue = [0.0] * nI1
        for n, k in idx_of.items():
            sw[k] = float(G.nodes[n]["sw_latency"])
            hw[k] = float(G.nodes[n]["hw_latency"])
            ar[k] = float(G.nodes[n]["area"])
            Ue[k] = float(U_end[n])
            if not all(math.isfinite(v) for v in (sw[k], hw[k], ar[k], Ue[k])):
                raise ValueError(f"non-finite cost attribute on node {n!r}")

        def qsum(exprs):
            exprs = list(exprs)
            return scip_quicksum(exprs) if exprs else 0.0

        succI1 = [[] for _ in range(nI1)]
        predI1 = [[] for _ in range(nI1)]
        has_ext_succ = [False] * nI1
        for n, k in idx_of.items():
            s_all = succ_all[n]
            s_in = [idx_of[s] for s in s_all if s in idx_of]
            succI1[k] = s_in
            if len(s_all) > 0 and len(s_in) < len(s_all):
                has_ext_succ[k] = True
            predI1[k] = [idx_of[p] for p in pred_all[n] if p in idx_of]

        x = {k: m.addVar(vtype="BINARY", name=f"x[{k}]") for k in range(nI1)}
        if amax is not None:
            for k in range(nI1):
                if ar[k] > float(amax) + EPS:
                    m.addCons(x[k] == 0)
        for n in I1_nodes:  # deterministic order (not set order)
            if n in invalid:
                m.addCons(x[idx_of[n]] == 0)

        # inputs (Eq. 1)
        b = {k: m.addVar(vtype="BINARY", name=f"b[{k}]") for k in range(nI1)}
        for k in range(nI1):
            succ_k = succI1[k]
            if not succ_k:
                m.addCons(b[k] == 0)
            elif len(succ_k) == 1:
                j = succ_k[0]
                m.addCons(b[k] <= 1 - x[k])
                m.addCons(b[k] <= x[j])
                m.addCons(b[k] >= x[j] - x[k])
            else:
                m.addCons(b[k] <= 1 - x[k])
                m.addCons(b[k] <= qsum(x[j] for j in succ_k))
                for j in succ_k:
                    m.addCons(b[k] >= x[j] - x[k])
        c = {}
        for t, s in enumerate(I2_nodes):
            c[s] = m.addVar(vtype="BINARY", name=f"c[{t}]")
            succ_s = [idx_of[j] for j in succ_all[s] if j in idx_of]
            if not succ_s:
                m.addCons(c[s] == 0)
            elif len(succ_s) == 1:
                m.addCons(c[s] == x[succ_s[0]])
            else:
                m.addCons(c[s] <= qsum(x[j] for j in succ_s))
                for j in succ_s:
                    m.addCons(c[s] >= x[j])
        # split super-nodes: one input per consumed ORIGINAL output
        split_k = {idx_of[a] for a in split if a in idx_of}
        w = []
        for a, groups in split.items():
            ka = idx_of[a]
            for t, cons in enumerate(groups):
                wv = m.addVar(vtype="BINARY", name=f"w[{ka},{t}]")
                cj = [idx_of[j] for j in cons]
                m.addCons(wv <= 1 - x[ka])
                m.addCons(wv <= qsum(x[j] for j in cj))
                for j in cj:
                    m.addCons(wv >= x[j] - x[ka])
                w.append(wv)
        m.addCons(qsum(b[k] for k in range(nI1) if k not in split_k) + qsum(c.values()) + qsum(w)
                  <= int(self.max_in))

        # outputs (Eq. 2)
        d = {k: m.addVar(vtype="BINARY", name=f"d[{k}]") for k in I4}
        for k in I4:
            succ_in = succI1[k]
            if not succ_in:
                m.addCons(d[k] == x[k])
            elif len(succ_in) == 1 and not has_ext_succ[k]:
                j = succ_in[0]
                m.addCons(d[k] <= x[k])
                m.addCons(d[k] <= 1 - x[j])
                m.addCons(d[k] >= x[k] - x[j])
            else:
                e = m.addVar(vtype="BINARY", name=f"e[{k}]")
                for j in succ_in:
                    m.addCons(e >= 1 - x[j])
                m.addCons(e <= qsum(1 - x[j] for j in succ_in))
                if has_ext_succ[k]:
                    m.addCons(e == 1)
                m.addCons(d[k] <= x[k])
                m.addCons(d[k] <= e)
                m.addCons(d[k] >= x[k] + e - 1)
        m.addCons(qsum(x[k] for k in I3) + qsum(d.values()) <= int(self.max_out))

        # convexity (Eqs. 3-5)
        Avar, Dvar = {}, {}
        for k in range(nI1):
            if predI1[k]:
                Avar[k] = m.addVar(vtype="BINARY", name=f"A[{k}]")
            if succI1[k]:
                Dvar[k] = m.addVar(vtype="BINARY", name=f"D[{k}]")
        for k in range(nI1):
            preds = predI1[k]
            if preds and k in Avar:
                for j in preds:
                    m.addCons(Avar[k] >= x[j])
                    if j in Avar:
                        m.addCons(Avar[k] >= Avar[j])
                m.addCons(Avar[k] <= qsum(x[j] for j in preds) + qsum(Avar[j] for j in preds if j in Avar))
            succs = succI1[k]
            if succs and k in Dvar:
                for j in succs:
                    m.addCons(Dvar[k] >= x[j])
                    if j in Dvar:
                        m.addCons(Dvar[k] >= Dvar[j])
                m.addCons(Dvar[k] <= qsum(x[j] for j in succs) + qsum(Dvar[j] for j in succs if j in Dvar))
            Ai = Avar[k] if k in Avar else 0
            Di = Dvar[k] if k in Dvar else 0
            m.addCons(Ai + Di <= x[k] + 1)

        # area cap
        if amax is not None:
            m.addCons(qsum(ar[k] * x[k] for k in range(nI1)) <= float(amax))

        # connectivity
        if self.connected and nI1 > 0:
            r = {k: m.addVar(vtype="BINARY", name=f"r[{k}]") for k in range(nI1)}
            u_conn = m.addVar(vtype="BINARY", name="u_conn")
            m.addCons(qsum(x[k] for k in range(nI1)) >= u_conn)
            m.addCons(qsum(x[k] for k in range(nI1)) <= nI1 * u_conn)
            m.addCons(qsum(r[k] for k in range(nI1)) == u_conn)
            for k in range(nI1):
                m.addCons(r[k] <= x[k])
            p = {}
            p_ctr = 0
            for ni in I1_nodes:
                i = idx_of[ni]
                for nj in neighbors.get(ni, ()):
                    if nj not in idx_of:
                        continue
                    j = idx_of[nj]
                    pv = m.addVar(vtype="BINARY", name=f"p[{p_ctr}]")
                    p_ctr += 1
                    p[(i, j)] = pv
                    m.addCons(pv <= x[i])
                    m.addCons(pv <= x[j])
            for ni in I1_nodes:
                i = idx_of[ni]
                neigh = [idx_of[nj] for nj in neighbors.get(ni, ()) if nj in idx_of]
                if neigh:
                    m.addCons(qsum(p[(i, j)] for j in neigh if (i, j) in p) == x[i] - r[i])
                else:
                    m.addCons(x[i] - r[i] == 0)
            m.addCons(qsum(p.values()) == qsum(x.values()) - u_conn)
            N = nI1
            h = {k: m.addVar(vtype="CONTINUOUS", lb=0.0, ub=float(N), name=f"h[{k}]") for k in range(N)}
            for k in range(N):
                m.addCons(h[k] <= N * x[k])
            for (i, j), pij in p.items():
                m.addCons(h[i] >= h[j] + 1 - N * (1 - pij))

        # timing (Eqs. 6-9)
        l = {k: m.addVar(vtype="CONTINUOUS", lb=0.0, name=f"l[{k}]") for k in range(nI1)}  # noqa: E741
        L_ub = cycles(max(Ue)) if nI1 else 0
        L = m.addVar(vtype="INTEGER", lb=0, ub=L_ub, name="L")
        for k in range(nI1):
            if not predI1[k]:
                m.addCons(l[k] == x[k] * hw[k])
            else:
                m.addCons(l[k] >= x[k] * hw[k])
            m.addCons(l[k] <= Ue[k] * x[k])
        for u, v in prec_edges:
            if u in idx_of and v in idx_of:
                ku, kv = idx_of[u], idx_of[v]
                m.addCons(l[kv] >= l[ku] + hw[kv] - Ue[kv] * (2 - x[ku] - x[kv]))
        for k in range(nI1):
            if not succI1[k]:
                m.addCons(L >= l[k])
        for k in I4:
            m.addCons(L >= l[k] - Ue[k] * (1 - d[k]))

        # no-good cuts (guard path, see module docstring)
        for C in nogoods:
            if all(n in idx_of for n in C):
                ks = {idx_of[n] for n in C}
                m.addCons(qsum(x[k] for k in ks) - qsum(x[k] for k in range(nI1) if k not in ks)
                          <= len(ks) - 1)

        profit_expr = qsum(sw[k] * x[k] for k in range(nI1)) - L
        area_expr = qsum(ar[k] * x[k] for k in range(nI1))
        m.setObjective(profit_expr - float(lam or 0.0) * area_expr, "maximize")

        t_start = time.monotonic()
        m.optimize()
        wall = time.monotonic() - t_start
        status = str(m.getStatus())
        nsols = m.getNSols()
        diag = dict(solver="scip", status=status, solver_wall_time_s=wall,
                    time_limit_s=_finite_or_none(time_limit), hit_time_limit=(status == "timelimit"),
                    primal_objective=_finite_or_none(m.getPrimalbound()) if nsols else None,
                    best_bound=_finite_or_none(m.getDualbound()),
                    relative_gap=_finite_or_none(m.getGap()) if nsols else None,
                    has_solution=bool(nsols > 0), split_supernodes=len(split),
                    nogood_cuts=len(nogoods))
        if nsols <= 0:
            return [], 0.0, diag
        sol = m.getBestSol()
        obj_val = _finite_or_none(m.getSolObjVal(sol))
        if obj_val is None or obj_val <= 0.0:
            return [], 0.0, diag
        chosen = [node_of[k] for k in range(nI1) if float(m.getSolVal(sol, x[k])) > 0.5]
        L_val = float(m.getSolVal(sol, L))
        obj_true = float(sum(sw[k] * float(m.getSolVal(sol, x[k])) for k in range(nI1)) - L_val)
        if not (math.isfinite(obj_true) and obj_true > 0.0):
            return [], 0.0, diag
        return chosen, obj_true, diag

    # ------------------------------------------------------------ CPLEX

    def _solve_cplex(self, S, lam, amax, invalid, nogoods, time_limit):  # pragma: no cover
        mdl = CpxModel(name="ILP_IO_iter")
        th = self.threads
        for setter in (
            lambda: setattr(mdl.parameters, "threads", th),
            lambda: setattr(mdl.context.cplex_parameters, "threads", th),
            lambda: setattr(mdl.parameters.mip, "display", 0),
            lambda: setattr(mdl.parameters.preprocessing, "presolve", 1),
            lambda: setattr(mdl.parameters.preprocessing, "aggregator", 3),
            lambda: setattr(mdl.parameters.preprocessing, "boundstrength", 1),
            lambda: setattr(mdl.parameters.mip.strategy, "search", 1),
            lambda: setattr(mdl.parameters.mip.limits, "cutpasses", 0),
            lambda: setattr(mdl.parameters.mip.tolerances, "mipgap", 0.0),
            lambda: setattr(mdl.parameters.emphasis, "mip", 1),
            lambda: setattr(mdl.parameters.mip.cuts, "mir", 2),
            lambda: setattr(mdl.parameters.mip.cuts, "covers", 2),
            lambda: setattr(mdl.parameters.mip.cuts, "flowcovers", 2),
            lambda: setattr(mdl.parameters.mip.cuts, "gomory", 2),
            lambda: setattr(mdl.parameters.mip.strategy, "rinsheur", 10),
        ):
            try:
                setter()
            except Exception:  # noqa: BLE001
                pass
        if time_limit is not None and math.isfinite(time_limit):
            mdl.parameters.timelimit = float(time_limit)
        if self.seed:
            mdl.parameters.randomseed = self.seed

        G = S["G"]
        I1, I2, I3, I4 = S["I1"], S["I2"], S["I3"], S["I4"]
        succ_all, pred_all = S["succ_all"], S["pred_all"]
        neighbors, U_end, prec_edges = S["neighbors"], S["U_end"], S["prec_edges"]
        split = S["split"]
        if not I1:
            return [], 0.0, None
        I1s = set(I1)

        x = {i: mdl.binary_var(name=f"x[{i}]") for i in I1}
        for i in I1:
            try:
                x[i].set_priority(10)
            except Exception:  # noqa: BLE001
                pass
        if amax is not None:
            for i in I1:
                if float(G.nodes[i]["area"]) > float(amax) + EPS:
                    mdl.add_constraint(x[i] == 0)
        for i in I1:  # deterministic order (not set order)
            if i in invalid:
                mdl.add_constraint(x[i] == 0)

        b = {i: mdl.binary_var(name=f"b[{i}]") for i in I1}
        for i in I1:
            succ_i = [j for j in succ_all[i] if j in I1s]
            if not succ_i:
                mdl.add_constraint(b[i] == 0)
            elif len(succ_i) == 1:
                j = succ_i[0]
                mdl.add_constraint(b[i] <= 1 - x[i])
                mdl.add_constraint(b[i] <= x[j])
                mdl.add_constraint(b[i] >= x[j] - x[i])
            else:
                mdl.add_constraint(b[i] <= 1 - x[i])
                mdl.add_constraint(b[i] <= mdl.sum(x[j] for j in succ_i))
                for j in succ_i:
                    mdl.add_constraint(b[i] >= x[j] - x[i])
        c = {i: mdl.binary_var(name=f"c[{i}]") for i in I2}
        for i in I2:
            succ_i = [j for j in succ_all[i] if j in I1s]
            if not succ_i:
                mdl.add_constraint(c[i] == 0)
            elif len(succ_i) == 1:
                mdl.add_constraint(c[i] == x[succ_i[0]])
            else:
                mdl.add_constraint(c[i] <= mdl.sum(x[j] for j in succ_i))
                for j in succ_i:
                    mdl.add_constraint(c[i] >= x[j])
        w = []
        for a, groups in split.items():
            for t, cons in enumerate(groups):
                wv = mdl.binary_var(name=f"w[{a},{t}]")
                mdl.add_constraint(wv <= 1 - x[a])
                mdl.add_constraint(wv <= mdl.sum(x[j] for j in cons))
                for j in cons:
                    mdl.add_constraint(wv >= x[j] - x[a])
                w.append(wv)
        mdl.add_constraint(mdl.sum(b[i] for i in I1 if i not in split) + mdl.sum(c.values())
                           + mdl.sum(w) <= self.max_in)

        d = {i: mdl.binary_var(name=f"d[{i}]") for i in I4}
        for i in I4:
            succ_any = list(succ_all[i])
            succ_in_I1 = [j for j in succ_any if j in I1s]
            has_ext = len(succ_any) > 0 and len(succ_in_I1) < len(succ_any)
            if not succ_in_I1:
                mdl.add_constraint(d[i] == x[i])
            elif len(succ_in_I1) == 1 and not has_ext:
                j = succ_in_I1[0]
                mdl.add_constraint(d[i] <= x[i])
                mdl.add_constraint(d[i] <= 1 - x[j])
                mdl.add_constraint(d[i] >= x[i] - x[j])
            else:
                e_i = mdl.binary_var(name=f"e[{i}]")
                for j in succ_in_I1:
                    mdl.add_constraint(e_i >= 1 - x[j])
                mdl.add_constraint(e_i <= mdl.sum(1 - x[j] for j in succ_in_I1))
                if has_ext:
                    mdl.add_constraint(e_i == 1)
                mdl.add_constraint(d[i] <= x[i])
                mdl.add_constraint(d[i] <= e_i)
                mdl.add_constraint(d[i] >= x[i] + e_i - 1)
        mdl.add_constraint(mdl.sum(x[i] for i in I3 if i in x) + mdl.sum(d.values()) <= self.max_out)

        A = {i: mdl.binary_var(name=f"A[{i}]") for i in I1 if any(p in I1s for p in pred_all[i])}
        Dv = {i: mdl.binary_var(name=f"D[{i}]") for i in I1 if any(s in I1s for s in succ_all[i])}
        for i in I1:
            preds = [p for p in pred_all[i] if p in I1s]
            if preds and i in A:
                for j in preds:
                    mdl.add_constraint(A[i] >= x[j])
                    if j in A:
                        mdl.add_constraint(A[i] >= A[j])
                mdl.add_constraint(A[i] <= mdl.sum(x[j] for j in preds) + mdl.sum(A[j] for j in preds if j in A))
            succs = [s for s in succ_all[i] if s in I1s]
            if succs and i in Dv:
                for j in succs:
                    mdl.add_constraint(Dv[i] >= x[j])
                    if j in Dv:
                        mdl.add_constraint(Dv[i] >= Dv[j])
                mdl.add_constraint(Dv[i] <= mdl.sum(x[j] for j in succs) + mdl.sum(Dv[j] for j in succs if j in Dv))
            mdl.add_constraint((A[i] if i in A else 0) + (Dv[i] if i in Dv else 0) <= x[i] + 1)

        if amax is not None:
            mdl.add_constraint(mdl.sum(float(G.nodes[i]["area"]) * x[i] for i in I1) <= float(amax))

        if self.connected and I1:
            r = {i: mdl.binary_var(name=f"r[{i}]") for i in I1}
            u_conn = mdl.binary_var(name="u_conn")
            mdl.add_constraint(mdl.sum(x[i] for i in I1) >= u_conn)
            mdl.add_constraint(mdl.sum(x[i] for i in I1) <= len(I1) * u_conn)
            mdl.add_constraint(mdl.sum(r[i] for i in I1) == u_conn)
            for i in I1:
                mdl.add_constraint(r[i] <= x[i])
            p = {}
            for i in I1:
                for j in neighbors[i]:
                    if j in I1s:
                        p[(i, j)] = mdl.binary_var(name=f"p[{i},{j}]")
                        mdl.add_constraint(p[(i, j)] <= x[i])
                        mdl.add_constraint(p[(i, j)] <= x[j])
            for i in I1:
                neigh_i = [j for j in neighbors[i] if j in I1s]
                if neigh_i:
                    mdl.add_constraint(mdl.sum(p[(i, j)] for j in neigh_i) == x[i] - r[i])
                else:
                    mdl.add_constraint(x[i] - r[i] == 0)
            mdl.add_constraint(mdl.sum(p.values()) == mdl.sum(x[i] for i in I1) - u_conn)
            N = len(I1)
            h = {i: mdl.continuous_var(lb=0.0, ub=float(N), name=f"h[{i}]") for i in I1}
            for i in I1:
                mdl.add_constraint(h[i] <= N * x[i])
            for (i, j), pij in p.items():
                mdl.add_constraint(h[i] >= h[j] + 1 - N * (1 - pij))

        l = {i: mdl.continuous_var(lb=0.0, name=f"l[{i}]") for i in I1}  # noqa: E741
        L = mdl.integer_var(lb=0, ub=cycles(max(U_end[i] for i in I1)), name="L")
        for i in I1:
            hi = float(G.nodes[i]["hw_latency"])
            if not any(p_ in I1s for p_ in pred_all[i]):
                mdl.add_constraint(l[i] == x[i] * hi)
            else:
                mdl.add_constraint(l[i] >= x[i] * hi)
            mdl.add_constraint(l[i] <= float(U_end[i]) * x[i])
        for u, v in prec_edges:
            if u in I1s and v in I1s:
                mdl.add_constraint(l[v] >= l[u] + float(G.nodes[v]["hw_latency"])
                                   - float(U_end[v]) * (2 - x[u] - x[v]))
        for i in I1:
            if not any(s in I1s for s in succ_all[i]):
                mdl.add_constraint(L >= l[i])
        for i in I4:
            mdl.add_constraint(L >= l[i] - float(U_end[i]) * (1 - d[i]))

        for C in nogoods:
            if all(n in x for n in C):
                mdl.add_constraint(mdl.sum(x[n] for n in C) - mdl.sum(x[n] for n in I1 if n not in C)
                                   <= len(C) - 1)

        profit_expr = mdl.sum(float(G.nodes[i]["sw_latency"]) * x[i] for i in I1) - L
        area_expr = mdl.sum(float(G.nodes[i]["area"]) * x[i] for i in I1)
        mdl.maximize(profit_expr - float(lam or 0.0) * area_expr)

        t_start = time.monotonic()
        sol = mdl.solve(log_output=False)
        wall = time.monotonic() - t_start
        details = getattr(mdl, "solve_details", None)
        status = str(getattr(details, "status", "unknown"))
        diag = dict(solver="cplex", status=status, solver_wall_time_s=wall,
                    time_limit_s=_finite_or_none(time_limit),
                    hit_time_limit=("time limit" in status.lower()),
                    primal_objective=(_finite_or_none(sol.get_objective_value()) if sol else None),
                    best_bound=_finite_or_none(getattr(details, "best_bound", None)),
                    relative_gap=_finite_or_none(getattr(details, "mip_relative_gap", None)),
                    has_solution=bool(sol is not None), split_supernodes=len(split),
                    nogood_cuts=len(nogoods))
        if not sol or sol.get_objective_value() <= 0:
            return [], 0.0, diag
        chosen = [i for i in I1 if x[i].solution_value > 0.5]
        L_val = float(sol.get_value(L))
        obj_true = float(sum(float(G.nodes[i]["sw_latency"]) * sol.get_value(x[i]) for i in I1) - L_val)
        return chosen, obj_true, diag

    # ------------------------------------------------------------ contraction

    @staticmethod
    def _contract(G: nx.DiGraph, comp: List[Node], name: str) -> None:
        """Contract the CI ``comp`` into the super-node ``name`` (inadmissible,
        hw latency = whole cycles of the critical path, sw latency and area =
        sums)."""
        C = set(comp)
        members = [n for n in G.nodes if n in C]  # deterministic order
        preds = list(dict.fromkeys(u for (u, v) in G.in_edges(members) if u not in C))
        succs = list(dict.fromkeys(v for (u, v) in G.out_edges(members) if v not in C))
        hw_cp = cycles(_topo_cp_latency(G, C, "hw_latency"))
        G.add_node(name, op="AGG", hw_latency=float(hw_cp),
                   sw_latency=float(sum(float(G.nodes[n]["sw_latency"]) for n in C)),
                   area=float(sum(float(G.nodes[n]["area"]) for n in C)), ci_eligible=False)
        G.add_edges_from((p, name) for p in preds)
        G.add_edges_from((name, s) for s in succs)
        G.remove_nodes_from(C)
        if not nx.is_directed_acyclic_graph(G):  # pragma: no cover - convex CIs keep a DAG
            raise RuntimeError("contraction of a CI produced a cycle (non-convex candidate?)")

    # ------------------------------------------------------------ one pass

    def _one_pass(self, frac: float, lam: float, amax: Optional[float],
                  deadline: Deadline) -> List[List[Node]]:
        G = self.G0.copy()
        invalid = set(self.invalid0)
        agg_outputs: Dict[Node, List[Node]] = {}
        nogoods: List[FrozenSet[Node]] = []
        split_active: Set[Node] = set()
        found: List[List[Node]] = []
        round_index = 0
        while True:
            rem = deadline.remaining()
            if rem <= 0.0:
                self.truncated = True
                break
            S = self._static(G, agg_outputs, split_active)
            if not S["I1"]:
                break
            self._ctx = {"cap_fraction": float(frac), "lambda_area": float(lam),
                         "contraction_round": int(round_index), "remaining_graph_nodes": int(len(G))}
            solve = self._solve_cplex if self.solver == "cplex" else self._solve_scip
            chosen, obj_true, diag = solve(S, lam, amax, invalid, nogoods,
                                           rem if deadline.bounded else None)
            if diag is not None:
                self._record(candidate=sorted(chosen, key=str), objective_true=obj_true, **diag)
                if diag["hit_time_limit"]:
                    self.truncated = True
            if not chosen or obj_true <= 0.0:
                break
            # lazy split: a super-node of which the candidate consumes two or
            # more ORIGINAL outputs while the input bound on the original DDG
            # is exceeded is split (one input binary per output); re-solve
            if validate.io_counts(self.G0, chosen)[0] > self.max_in:
                cs = set(chosen)
                new_split = [a for a, outs in agg_outputs.items()
                             if a in G and a not in split_active
                             and sum(1 for o in outs if any(s_ in cs for s_ in self.G0.successors(o))) >= 2]
                if new_split:
                    split_active.update(new_split)
                    self.lazy_splits += len(new_split)
                    continue
            reasons = validate.check_ci(self.g_orig, chosen, self.max_in, self.max_out, self.connected)
            if reasons:
                key = frozenset(chosen)
                self.rejected.append({"nodes": sorted(chosen, key=str), "reasons": reasons,
                                      "cap_fraction": float(frac), "lambda_area": float(lam),
                                      "contraction_round": round_index})
                logger.warning("ILP candidate %s failed validation on the original DDG: %s",
                               sorted(chosen, key=str), reasons)
                if key in nogoods:  # pragma: no cover - solver ignored the cut
                    raise RuntimeError("no-good cut did not exclude an invalid candidate")
                nogoods.append(key)
                continue  # re-solve the same round with the cut
            comp = sorted(chosen, key=str)
            found.append(comp)
            outs = validate.io_sets(self.G0, comp)[1]
            name = f"__ci_{len(found)}__"
            while name in G or name in self.G0:
                name += "_"
            self._contract(G, comp, name)
            invalid.add(name)
            agg_outputs[name] = [n for n in self.G0.nodes if n in outs]
            round_index += 1
        return found

    # ------------------------------------------------------------ public

    def resolve_lambda_grid(self) -> List[float]:
        if self.lambda_grid:
            grid = list(self.lambda_grid)
        elif self.lambda_area is not None:
            grid = [float(self.lambda_area)]
        else:
            grid = lambda_grid_from_graph(self.G0)
        grid = sorted({float(x) for x in grid if x is not None and x >= 0.0})
        return grid or [0.0]

    def run(self) -> Dict[str, Any]:
        """Run every (cap, lambda) pass.  Returns ``templates`` (node lists,
        de-duplicated across passes, in order of discovery), ``potentials``
        (cycle savings) and ``areas``, all computed on the original DDG."""
        have_area = self.area_max is not None
        caps = self.area_cap_fracs if have_area else [1.0]
        lam_grid = self.resolve_lambda_grid()
        combos = [(f, lam) for f in caps for lam in lam_grid]
        seen: Set[FrozenSet[Node]] = set()
        T: List[List[Node]] = []
        for k, (frac, lam) in enumerate(combos):
            rem = self.deadline.remaining()
            if rem <= 0.0:
                self.truncated = True
                break
            if self.deadline.bounded:
                pass_deadline = Deadline.absolute(time.monotonic() + rem / (len(combos) - k))
            else:
                pass_deadline = Deadline(None)
            amax = float(frac) * self.area_max if have_area else None
            found = self._one_pass(frac, lam, amax, pass_deadline)
            for nodes in found:
                key = frozenset(nodes)
                if key not in seen:
                    seen.add(key)
                    T.append(nodes)
        return {
            "templates": T,
            "potentials": [float(validate.saving(self.g_orig, t)) for t in T],
            "areas": [validate.ci_area(self.g_orig, t) for t in T],
            "lambda_grid": lam_grid,
            "cap_fractions": list(caps),
            "diagnostics": list(self.diagnostics),
            "rejected": list(self.rejected),
            "truncated": bool(self.truncated),
            "lazy_splits": int(self.lazy_splits),
        }
