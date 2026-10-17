"""Entry point of the ILP baseline: :func:`identify_ilp`."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from typing import Any, Dict, Hashable, List, Sequence

import networkx as nx

from ci_ident.baselines.ilp import validate
from ci_ident.baselines.ilp.enumerator import Deadline, ILPEnumerator, resolve_solver
from ci_ident.baselines.ilp.selector import ILPSelector
from ci_ident.costs import EPS
from ci_ident.ddg.graph import area_reference

logger = logging.getLogger(__name__)

Node = Hashable


def identify_ilp(
    g: nx.DiGraph,
    max_in: int,
    max_out: int,
    area_factor: float,
    *,
    time_budget_s: float = 3600.0,
    solver: str = "auto",
    area_cap_fracs: Sequence[float] = (1 / 3, 2 / 3, 1.0),
    connected: bool = True,
    seed: int = 0,
    enum_share: float = 0.7,
    sel_min_s: float = 1.0,
    lambda_grid: Sequence[float] = None,
) -> Dict[str, Any]:
    """ILP-based CI identification on one DDG.

    Parameters
    ----------
    g
        Annotated DDG from :func:`ci_ident.ddg.graph.load_ddg`; read only.
    max_in, max_out
        Port bounds ``|In(S)| <= max_in``, ``|Out(S)| <= max_out``.
    area_factor
        ``alpha``; the area budget is ``alpha * A_ref`` with ``A_ref`` the
        area of all CI-admissible nodes (:func:`area_reference`).
    time_budget_s
        Wall-clock budget of the whole scenario (enumeration + selection)
        on one monotonic deadline.  ``math.inf`` disables it; ``<= 0`` means
        already expired.
    solver
        ``"auto"`` (CPLEX via docplex if importable, else SCIP), ``"cplex"``
        or ``"scip"``.
    area_cap_fracs
        Per-CI area caps of the enumeration sweep, as fractions of the
        global budget ``alpha * A_ref``.
    connected
        Require weakly connected CIs.
    seed
        Solver random-seed shift (0 keeps the solver defaults).
    enum_share, sel_min_s
        The enumeration may use ``min(enum_share * T, T - sel_min_s)`` of the
        budget ``T``; selection gets the
        rest of the same deadline.
    lambda_grid
        Explicit area-penalty grid; by default derived from the graph.

    Returns
    -------
    dict with ``speedup`` (``T_sw / (T_sw - sum savings)``, ``T_sw = |V|``),
    ``area_used``, ``area_budget``, ``a_ref``, ``num_patterns``,
    ``num_instances``, ``instances`` (node-name lists), ``pattern_of_instance``
    (index into the selected patterns), ``mean_ci_size``, ``runtime_s``,
    ``status`` (``"ok"``: every solve finished within its limit;
    ``"timeout_partial"``: a time limit bound somewhere, the result is the
    best found; ``"no_solution"``: a time limit bound and no CI was
    selected), ``solver``, ``diagnostics`` (one dict per MILP solve) and
    ``rejected`` (enumerated candidates dropped by validation on the
    original DDG, with reasons).
    """
    t0 = time.monotonic()
    budget = math.inf if time_budget_s is None else float(time_budget_s)
    deadline = Deadline(budget, start=t0)
    if math.isinf(budget):
        enum_deadline = Deadline(None, start=t0)
    else:
        enum_s = max(0.0, min(budget * float(enum_share),
                              budget - (max(0.0, sel_min_s) if budget >= sel_min_s else 0.0)))
        enum_deadline = Deadline(enum_s, start=t0)

    solver_name = resolve_solver(solver)
    a_ref = float(area_reference(g))
    area_budget = float(area_factor) * a_ref
    T_sw = float(g.number_of_nodes())

    enum = ILPEnumerator(g, max_in, max_out, area_budget, area_cap_fracs, connected=connected,
                         solver=solver_name, deadline=enum_deadline, seed=seed,
                         lambda_grid=lambda_grid)
    er = enum.run()

    sel = ILPSelector(g, er["templates"], er["potentials"], er["areas"], area_budget,
                      solver=solver_name, deadline=deadline, seed=seed)
    sr = sel.run()

    classes = sr["classes"]
    chosen = sr["chosen_instances"]
    instances: List[List[Node]] = [list(er["templates"][i]) for i in chosen]

    # independent final check on the original DDG; never return an invalid set
    chk = validate.check_selection(g, instances, max_in, max_out, connected)
    if not chk["ok"]:  # pragma: no cover - guarded by construction
        raise RuntimeError(f"ILP baseline produced an invalid selection: {chk}")

    sel_classes = sr["chosen_classes"]
    pos = {k: t for t, k in enumerate(sel_classes)}
    cls_of = {i: k for k, c in enumerate(classes) for i in c["members"]}
    pattern_of_instance = [pos[cls_of[i]] for i in chosen]
    area_used = float(sum(classes[k]["area"] for k in sel_classes))
    if area_used > area_budget + EPS:  # pragma: no cover
        raise RuntimeError("selection exceeds the area budget")

    savings = [validate.saving(g, S) for S in instances]
    accel = T_sw - sum(savings)
    speedup = 1.0 if T_sw <= 0 else (math.inf if accel <= 0 else T_sw / accel)

    diagnostics = list(er["diagnostics"]) + list(sel.diagnostics)
    hit_limit = er["truncated"] or sel.timed_out
    if hit_limit and not instances:
        status = "no_solution"
    elif hit_limit or not sr["solved"]:
        status = "timeout_partial"
    else:
        status = "ok"

    reasons = Counter(r for rej in er["rejected"] for r in rej["reasons"])
    return {
        "speedup": float(speedup),
        "area_used": area_used,
        "area_budget": area_budget,
        "a_ref": a_ref,
        "num_patterns": len(sel_classes),
        "num_instances": len(instances),
        "instances": instances,
        "pattern_of_instance": pattern_of_instance,
        "mean_ci_size": (float(sum(len(S) for S in instances)) / len(instances)) if instances else 0.0,
        "runtime_s": time.monotonic() - t0,
        "status": status,
        "solver": solver_name,
        "diagnostics": diagnostics,
        "rejected": {"count": len(er["rejected"]), "reasons": dict(reasons),
                     "candidates": er["rejected"]},
        # extra bookkeeping
        "savings": savings,
        "num_candidates": len(er["templates"]),
        "num_patterns_total": len(classes),
        "num_instances_total": int(sum(len(c["members"]) for c in classes)),
        "pair_conflicts": len(sel.pair_conflicts),
        "cycle_cuts": len(sel.cycle_cuts),
        "area_cuts": len(sel.area_cuts),
        "selection_repaired": bool(sel.repaired),
        "lambda_grid": er["lambda_grid"],
        "cap_fractions": er["cap_fractions"],
        "enum_truncated": bool(er["truncated"]),
        "lazy_splits": int(er["lazy_splits"]),
    }
