"""Stage 2 of the ILP baseline: conflict-free selection of extracted CIs.

The enumerated instances are grouped into patterns by the direction-aware
1-WL signature of :mod:`ci_ident.ddg.signature` (the one the RL environment
uses).  A 0-1 MILP then picks instances ``y_i`` and pattern switches ``z_k``:

    max  sum_i gain_i y_i - eps * sum_k area_k z_k
    s.t. y_i <= z_k (i in pattern k),  z_k <= sum_{i in k} y_i,
         sum_k area_k z_k <= A_budget        (area paid once per pattern),
         node exclusivity: sum_{i contains v} y_i <= 1 for every node v,
         y_i + y_j <= 1 for every mutually dependent pair,
         sum_{i in C} y_i <= |C| - 1 for every lazily found cycle C,
         sum_{k in K} z_k <= |K| - 1 for every lazily found over-budget K.

``eps = 1 / (10 * sum gains + 1)`` makes the area term a pure tie-break.

Area tolerance: the budget row has right-hand side ``A_budget``; the
solver's feasibility tolerance admits a sum equal to the budget up to float
rounding.  Every
comparison made outside the solver uses :data:`ci_ident.costs.EPS`, as the
RL environment does, and a solver answer whose pattern area exceeds
``A_budget + EPS`` is excluded by a lazy cut
``sum_{k in K} z_k <= |K| - 1`` on its set of patterns ``K`` and re-solved.

Schedulability of the selection
-------------------------------
Node exclusivity alone admits instances that are mutually dependent on the
original DDG (``i`` reaches ``j`` and ``j`` reaches ``i`` through nodes
outside both); contracting such a selection gives a cycle, and no schedule
can execute it.  The model therefore forbids every such pair up front, and
after each solve the set-level check of
:func:`.validate.find_contracted_cycle` (contract every selected CI to a
super-node in the original DDG and test for a DAG) is run.  A longer cycle
``C`` (three or more CIs) is excluded with the lazy constraint above and
the MILP is rebuilt and re-solved, until the selection is acyclic.  If the
deadline expires inside that loop with a cyclic incumbent, the lowest-gain
instance on each remaining cycle is dropped (status reported as a
time-out), so a selection that cannot be scheduled is never returned.

Patterns are grouped by :meth:`ci_ident.ddg.signature.SignatureIndex.signature`,
which does not depend on the order in which a CI's nodes are listed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, FrozenSet, Hashable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ci_ident.baselines.ilp import validate
from ci_ident.baselines.ilp.enumerator import (
    HAVE_CPLEX,
    HAVE_SCIP,
    CpxModel,
    Deadline,
    ScipModel,
    _finite_or_none,
    resolve_solver,
    scip_quicksum,
)
from ci_ident.costs import EPS
from ci_ident.ddg.signature import SignatureIndex

logger = logging.getLogger(__name__)

Node = Hashable

__all__ = ["ILPSelector", "HAVE_CPLEX", "HAVE_SCIP"]


class ILPSelector:
    """Instance-level packing MILP with area charged once per pattern.

    ``g`` is the annotated original DDG (read only); ``templates``,
    ``potentials`` and ``areas`` come from :class:`.enumerator.ILPEnumerator`.
    """

    def __init__(
        self,
        g: nx.DiGraph,
        templates: Sequence[Sequence[Node]],
        potentials: Sequence[float],
        areas: Sequence[float],
        area_max: float,
        *,
        nk: Optional[int] = None,
        solver: str = "auto",
        deadline: Optional[Deadline] = None,
        class_sort: str = "total",
        seed: int = 0,
    ):
        self.g = g
        self.templates = [list(t) for t in templates]
        self.potentials = [float(p) for p in potentials]
        self.areas = [float(a) for a in areas]
        if not (len(self.templates) == len(self.potentials) == len(self.areas)):
            raise ValueError("templates, potentials and areas differ in length")
        self.area_max = float(max(0.0, area_max))
        self.nk = nk
        self.solver = resolve_solver(solver)
        self.deadline = deadline if deadline is not None else Deadline(None)
        self.class_sort = class_sort
        self.seed = int(seed or 0)
        self.diagnostics: List[Dict[str, Any]] = []
        self.timed_out = False
        self.repaired = False
        self.cycle_cuts: List[FrozenSet[int]] = []
        self.area_cuts: List[FrozenSet[int]] = []
        self.pair_conflicts: List[Tuple[int, int]] = []
        logger.info("ILP selector: solver=%s candidates=%d", self.solver, len(self.templates))

    # ------------------------------------------------------------ grouping

    def group(self) -> List[Dict[str, Any]]:
        """Patterns (classes) of the kept instances; instances with fewer than
        two nodes or a non-positive gain are not kept."""
        kept = [i for i, T in enumerate(self.templates)
                if len(T) >= 2 and all(u in self.g for u in T) and self.potentials[i] > 0.0]
        sig = SignatureIndex.from_graph(self.g)
        code2idxs: Dict[Any, List[int]] = {}
        for i in kept:
            code2idxs.setdefault(sig.signature(self.templates[i]), []).append(i)
        classes: List[Dict[str, Any]] = []
        for code, idxs in code2idxs.items():
            rep = idxs[0]
            area_rep = self.areas[rep]
            for j in idxs[1:]:
                if abs(self.areas[j] - area_rep) > 1e-6:
                    raise AssertionError(f"area mismatch inside a pattern: {area_rep} vs {self.areas[j]}")
            pot_sum = float(sum(self.potentials[j] for j in idxs))
            if pot_sum <= 0.0:
                continue
            classes.append({"members": idxs, "rep_idx": rep, "area": area_rep, "potential": pot_sum,
                            "best_gain": float(max(self.potentials[j] for j in idxs)), "signature": code})
        if self.class_sort == "best":
            classes.sort(key=lambda c: (-c["best_gain"], c["area"]))
        else:
            classes.sort(key=lambda c: (-c["potential"], c["area"]))
        return classes

    # ------------------------------------------------------------ helpers

    def _greedy_start(self, classes, conflicts: Dict[int, Set[int]], stop_at_first: bool) -> Tuple[List[int], List[int]]:
        """Greedy warm start: patterns in order of gain density, skipping
        instances that conflict with a chosen one."""
        order = sorted(range(len(classes)),
                       key=lambda k: (classes[k]["potential"] / max(classes[k]["area"], 1e-12),
                                      classes[k]["potential"]), reverse=True)
        used: Set[Node] = set()
        chosen: List[int] = []
        ks: List[int] = []
        area_used = 0.0
        nk_left = int(self.nk) if self.nk is not None else len(classes)
        for k in order:
            if nk_left == 0:
                break
            a = classes[k]["area"]
            if area_used + a > self.area_max + EPS:
                continue
            picked = False
            for i in sorted(classes[k]["members"], key=lambda t: self.potentials[t], reverse=True):
                if any(u in used for u in self.templates[i]):
                    continue
                if conflicts.get(i) and any(j in conflicts[i] for j in chosen):
                    continue
                if any(C <= set(chosen) | {i} for C in self.cycle_cuts):
                    continue
                chosen.append(i)
                used.update(self.templates[i])
                picked = True
                if stop_at_first:
                    break
            if picked:
                ks.append(k)
                area_used += a
                nk_left -= 1
        return ks, chosen

    # ------------------------------------------------------------ MILPs

    def _solve_scip(self, classes, inst_idxs, node_to_insts, time_limit):
        m = ScipModel("InstanceClassSelection")
        m.hideOutput()
        try:
            m.setIntParam("display/verblevel", 0)
        except Exception:  # noqa: BLE001
            pass
        if time_limit is not None:
            m.setParam("limits/time", float(time_limit))
        if self.seed:
            m.setIntParam("randomization/randomseedshift", self.seed)
        y = {i: m.addVar(vtype="BINARY", name=f"y[{i}]") for i in inst_idxs}
        z = {k: m.addVar(vtype="BINARY", name=f"z[{k}]") for k in range(len(classes))}
        for Iu in node_to_insts.values():
            if len(Iu) > 1:
                m.addCons(scip_quicksum(y[i] for i in Iu) <= 1)
        for k, cls in enumerate(classes):
            Ik = [i for i in cls["members"] if i in y]
            if not Ik:
                m.addCons(z[k] == 0)
                continue
            for i in Ik:
                m.addCons(y[i] <= z[k])
            m.addCons(z[k] <= scip_quicksum(y[i] for i in Ik))
        m.addCons(scip_quicksum(classes[k]["area"] * z[k] for k in range(len(classes))) <= self.area_max)
        if self.nk is not None:
            m.addCons(scip_quicksum(z[k] for k in range(len(classes))) <= int(self.nk))
        # schedulability: mutually dependent pairs and lazily found cycles
        for i, j in self.pair_conflicts:
            m.addCons(y[i] + y[j] <= 1)
        for C in self.cycle_cuts:
            m.addCons(scip_quicksum(y[i] for i in C) <= len(C) - 1)
        for K in self.area_cuts:
            m.addCons(scip_quicksum(z[k] for k in K) <= len(K) - 1)

        gains_expr = scip_quicksum(self.potentials[i] * y[i] for i in inst_idxs)
        area_expr = scip_quicksum(classes[k]["area"] * z[k] for k in range(len(classes)))
        max_gain = sum(max(0.0, p) for p in self.potentials)
        eps = 1.0 / (10.0 * max_gain + 1.0)
        m.setObjective(gains_expr - eps * area_expr, "maximize")

        try:  # greedy warm start (SCIP: all fitting members)
            ks, chosen = self._greedy_start(classes, self._conflict_map, stop_at_first=False)
            sol = m.createSol()
            for i in chosen:
                m.setSolVal(sol, y[i], 1.0)
            for k in ks:
                m.setSolVal(sol, z[k], 1.0)
            m.addSol(sol, free=True)
        except Exception:  # noqa: BLE001
            pass

        t_start = time.monotonic()
        m.optimize()
        wall = time.monotonic() - t_start
        status = str(m.getStatus())
        nsols = m.getNSols()
        diag = dict(phase="select", solver="scip", status=status, solver_wall_time_s=wall,
                    time_limit_s=_finite_or_none(time_limit), hit_time_limit=(status == "timelimit"),
                    primal_objective=_finite_or_none(m.getPrimalbound()) if nsols else None,
                    best_bound=_finite_or_none(m.getDualbound()),
                    relative_gap=_finite_or_none(m.getGap()) if nsols else None,
                    has_solution=bool(nsols > 0), pair_conflicts=len(self.pair_conflicts),
                    cycle_cuts=len(self.cycle_cuts))
        if status not in ("optimal", "timelimit") or nsols <= 0:
            return None, diag
        sol = m.getBestSol()
        return [i for i in inst_idxs if m.getSolVal(sol, y[i]) > 0.5], diag

    def _solve_cplex(self, classes, inst_idxs, node_to_insts, time_limit):  # pragma: no cover
        mdl = CpxModel(name="InstanceClassSelection")
        for setter in (lambda: setattr(mdl.parameters.mip, "display", 0),
                       lambda: setattr(mdl.parameters, "threads", 1)):
            try:
                setter()
            except Exception:  # noqa: BLE001
                pass
        if time_limit is not None:
            mdl.parameters.timelimit = float(time_limit)
        if self.seed:
            mdl.parameters.randomseed = self.seed
        y = {i: mdl.binary_var(name=f"y[{i}]") for i in inst_idxs}
        z = {k: mdl.binary_var(name=f"z[{k}]") for k in range(len(classes))}
        for Iu in node_to_insts.values():
            if len(Iu) > 1:
                mdl.add_constraint(mdl.sum(y[i] for i in Iu) <= 1)
        for k, cls in enumerate(classes):
            Ik = [i for i in cls["members"] if i in y]
            if not Ik:
                mdl.add_constraint(z[k] == 0)
                continue
            for i in Ik:
                mdl.add_constraint(y[i] <= z[k])
            mdl.add_constraint(z[k] <= mdl.sum(y[i] for i in Ik))
        mdl.add_constraint(mdl.sum(classes[k]["area"] * z[k] for k in range(len(classes))) <= self.area_max)
        if self.nk is not None:
            mdl.add_constraint(mdl.sum(z[k] for k in range(len(classes))) <= int(self.nk))
        for i, j in self.pair_conflicts:
            mdl.add_constraint(y[i] + y[j] <= 1)
        for C in self.cycle_cuts:
            mdl.add_constraint(mdl.sum(y[i] for i in C) <= len(C) - 1)
        for K in self.area_cuts:
            mdl.add_constraint(mdl.sum(z[k] for k in K) <= len(K) - 1)
        max_gain = sum(max(0.0, p) for p in self.potentials)
        eps = 1.0 / (10.0 * max_gain + 1.0)
        mdl.maximize(mdl.sum(self.potentials[i] * y[i] for i in inst_idxs)
                     - eps * mdl.sum(classes[k]["area"] * z[k] for k in range(len(classes))))
        try:  # greedy warm start (CPLEX: one member per pattern)
            ks, chosen = self._greedy_start(classes, self._conflict_map, stop_at_first=True)
            ws = mdl.new_solution()
            for i in chosen:
                ws.add_var_value(y[i], 1)
            for k in ks:
                ws.add_var_value(z[k], 1)
            mdl.add_mip_start(ws)
        except Exception:  # noqa: BLE001
            pass
        t_start = time.monotonic()
        sol = mdl.solve(log_output=False)
        wall = time.monotonic() - t_start
        details = getattr(mdl, "solve_details", None)
        status = str(getattr(details, "status", "unknown"))
        diag = dict(phase="select", solver="cplex", status=status, solver_wall_time_s=wall,
                    time_limit_s=_finite_or_none(time_limit),
                    hit_time_limit=("time limit" in status.lower()),
                    primal_objective=(_finite_or_none(sol.get_objective_value()) if sol else None),
                    best_bound=_finite_or_none(getattr(details, "best_bound", None)),
                    relative_gap=_finite_or_none(getattr(details, "mip_relative_gap", None)),
                    has_solution=bool(sol is not None), pair_conflicts=len(self.pair_conflicts),
                    cycle_cuts=len(self.cycle_cuts))
        if sol is None:
            return None, diag
        return [i for i in inst_idxs if y[i].solution_value > 0.5], diag

    # ------------------------------------------------------------ repair

    def _area_of(self, chosen: Sequence[int]) -> Tuple[float, FrozenSet[int]]:
        """Area (once per pattern) and pattern set of a selection."""
        ks = frozenset(self._class_of[i] for i in chosen)
        return float(sum(self._classes[k]["area"] for k in ks)), ks

    def _repair(self, chosen: List[int]) -> List[int]:
        """Deadline fallback only: drop the lowest-gain instance of each
        remaining contracted cycle until the selection is acyclic, then the
        lowest-gain instances until the area fits the budget."""
        chosen = list(chosen)
        while True:
            cyc = validate.find_contracted_cycle(self.g, [self.templates[i] for i in chosen])
            if cyc is None:
                break
            on_cycle = [chosen[t] for t in cyc]
            drop = min(on_cycle, key=lambda i: (self.potentials[i], -i))
            chosen.remove(drop)
            self.repaired = True
        while chosen and self._area_of(chosen)[0] > self.area_max + EPS:
            chosen.remove(min(chosen, key=lambda i: (self.potentials[i], -i)))
            self.repaired = True
        return chosen

    # ------------------------------------------------------------ public

    def run(self) -> Dict[str, Any]:
        classes = self.group()
        inst_idxs = sorted({i for c in classes for i in c["members"]})
        self._conflict_map: Dict[int, Set[int]] = {}
        result = {"classes": classes, "chosen_instances": [], "chosen_classes": [],
                  "solved": False}
        if not classes or self.area_max <= 0 or not inst_idxs:
            result["solved"] = True  # nothing to select: trivially optimal
            return result

        node_to_insts: Dict[Node, List[int]] = {}
        for i in inst_idxs:
            for u in self.templates[i]:
                node_to_insts.setdefault(u, []).append(i)
        pairs = validate.mutually_dependent_pairs(self.g, [self.templates[i] for i in inst_idxs])
        self.pair_conflicts = [(inst_idxs[a], inst_idxs[b]) for a, b in pairs]
        for i, j in self.pair_conflicts:
            self._conflict_map.setdefault(i, set()).add(j)
            self._conflict_map.setdefault(j, set()).add(i)

        self._classes = classes
        self._class_of = {i: k for k, c in enumerate(classes) for i in c["members"]}
        chosen: Optional[List[int]] = None
        while True:
            rem = self.deadline.remaining()
            if rem <= 0.0:
                self.timed_out = True
                break
            tl = rem if self.deadline.bounded else None
            solve = self._solve_cplex if self.solver == "cplex" else self._solve_scip
            sel, diag = solve(classes, inst_idxs, node_to_insts, tl)
            self.diagnostics.append(diag)
            if diag["hit_time_limit"]:
                self.timed_out = True
            if sel is None:
                break
            chosen = sel
            area, ks = self._area_of(sel)
            if area > self.area_max + EPS:
                logger.info("selection area %.12g exceeds the budget %.12g; adding a lazy cut",
                            area, self.area_max)
                self.area_cuts.append(ks)
                if self.timed_out:
                    break
                continue
            cyc = validate.find_contracted_cycle(self.g, [self.templates[i] for i in sel])
            if cyc is None:
                result["solved"] = not self.timed_out
                break
            C = frozenset(sel[t] for t in cyc)
            if len(C) < 3:  # pragma: no cover - pairs are excluded up front
                raise RuntimeError(f"selection contains a mutually dependent pair {sorted(C)}")
            logger.info("selection has a contracted cycle through %d CIs; adding a lazy cut", len(C))
            self.cycle_cuts.append(C)
            if self.timed_out:
                break
        if chosen is not None:
            chosen = self._repair(chosen)
        else:
            chosen = []
        sel_set = set(chosen)
        result["chosen_instances"] = chosen
        result["chosen_classes"] = [k for k, c in enumerate(classes) if any(i in sel_set for i in c["members"])]
        return result
