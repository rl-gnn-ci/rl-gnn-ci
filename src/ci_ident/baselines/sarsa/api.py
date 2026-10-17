"""Entry point of the SARSA baseline: :func:`identify_sarsa`."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import numpy as np
import networkx as nx

from ci_ident.baselines.sarsa import _ext, validate
from ci_ident.baselines.sarsa.enumeration import GraphIndex, enumerate_candidates
from ci_ident.baselines.sarsa.instances import build_instances, priority_order, reorder
from ci_ident.baselines.sarsa.selection import run_core
from ci_ident.costs import EPS
from ci_ident.ddg.graph import area_reference
from ci_ident.ddg.signature import SignatureIndex

#: Minimum share of the budget kept for instance building and selection when
#: the budget is at least this many seconds (same split rule as the ILP
#: baseline).
SEL_MIN_S = 1.0


def identify_sarsa(
    g: nx.DiGraph,
    max_in: int,
    max_out: int,
    area_factor: float,
    *,
    time_budget_s: float = 3600.0,
    episodes: int = 10_000,
    alpha: float = 0.012,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    seed: int = 0,
    connected: bool = True,
    enum_share: float = 0.7,
    max_candidates: int = 5_000_000,
) -> Dict[str, Any]:
    """SARSA-based CI identification on one DDG (exact enumeration, then
    tabular SARSA selection; see :mod:`ci_ident.baselines.sarsa`).

    Parameters
    ----------
    g
        Annotated DDG from :func:`ci_ident.ddg.graph.load_ddg`; read only.
        Legality is taken from the node attribute ``ci_eligible`` alone.
    max_in, max_out
        Port bounds ``|In(S)| <= max_in``, ``|Out(S)| <= max_out``.
    area_factor
        ``alpha_area``; the budget is ``area_factor * A_ref`` with ``A_ref``
        from :func:`ci_ident.ddg.graph.area_reference`.
    time_budget_s
        Wall-clock budget of the whole scenario on one monotonic deadline.
        ``math.inf`` (or ``None``) disables it; ``<= 0`` is already expired.
        The enumeration may use ``min(enum_share * T, T - 1 s)`` (for
        ``T >= 1 s``; ``enum_share * T`` otherwise); instance building and
        selection use the rest of the same deadline.
    episodes, alpha, gamma, eps_start, eps_end
        SARSA training episodes, step size, discount, and the linear
        epsilon schedule ``max(eps_end, eps_start * (1 - ep / episodes))``.
    seed
        Seed of the selector's random number generator.
    connected
        Restrict candidates to weakly connected sets (the paper's design
        space).
    enum_share
        Share of the budget available to the enumeration.
    max_candidates
        Bound on the number of stored candidates (about 1 GB of memory at the
        default); 0 means no bound.

    Returns
    -------
    dict with ``speedup`` (``T_sw / (T_sw - sum of savings)``, ``T_sw = |V|``),
    ``area_used``, ``area_budget``, ``a_ref``, ``num_patterns``,
    ``num_instances``, ``instances`` (node-name lists in topological order, in
    selection order), ``pattern_of_instance`` (index into the selected
    patterns, numbered by first selection), ``mean_ci_size``, ``runtime_s``,
    ``status`` (``"ok"``: enumeration, instance building, all episodes and
    the rollout completed; ``"timeout_partial"``: the deadline or the
    candidate bound was reached and the result is the selection obtained;
    ``"no_solution"``: a limit was reached and nothing was selected), ``num_candidates`` (valid
    candidates enumerated) and ``diagnostics``.

    Raises :class:`ci_ident.baselines.sarsa.ExtensionUnavailable` if the C++
    extensions are not built.
    """
    t0 = time.monotonic()
    budget = math.inf if time_budget_s is None else float(time_budget_s)
    if math.isnan(budget):
        raise ValueError("time_budget_s must be a number")
    if int(episodes) < 0:
        raise ValueError("episodes must be >= 0")
    if not 0.0 <= float(enum_share) <= 1.0:
        raise ValueError("enum_share must be in [0, 1]")
    deadline_at = math.inf if math.isinf(budget) else t0 + max(0.0, budget)
    if math.isinf(budget):
        enum_s = math.inf
    else:
        reserve = SEL_MIN_S if budget >= SEL_MIN_S else 0.0
        enum_s = max(0.0, min(budget * float(enum_share), budget - reserve))
    _ext.load("_xiao_enum")
    _ext.load("_sarsa_core")

    a_ref = float(area_reference(g))
    area_budget = float(area_factor) * a_ref
    T_sw = float(g.number_of_nodes())
    gi = GraphIndex(g)

    # 1. exact enumeration
    er = enumerate_candidates(g, max_in, max_out, connected=connected, seconds=enum_s,
                              max_candidates=int(max_candidates), index=gi)

    # 2. instances and patterns
    t_b = time.monotonic()
    inst = build_instances(g, er, deadline_at=deadline_at)
    inst = reorder(inst, priority_order(inst))
    build_s = time.monotonic() - t_b

    # 3. SARSA selection
    n_inst = len(inst["gains"])
    sr: Dict[str, Any] = {"chosen": [], "episodes_done": 0, "training_cut": False, "rollout_cut": False,
                          "epsilon_last": float(eps_start), "dependency_rejections": 0, "steps": 0,
                          "states": 0, "q_entries": 0, "train_s": 0.0, "rollout_s": 0.0, "num_actions": 0}
    if n_inst:
        core = _ext.load("_sarsa_core").Sarsa(
            gi.num_nodes, gi.edges, inst["nodes"], inst["offsets"], inst["gains"].astype(np.float64),
            inst["pattern"], inst["pattern_area"], area_budget, EPS)
        remaining = math.inf if math.isinf(deadline_at) else deadline_at - time.monotonic()
        sr = run_core(core, episodes=episodes, alpha=alpha, gamma=gamma, eps_start=eps_start,
                      eps_end=eps_end, seed=seed, seconds=remaining)

    # 4. result on the original DDG
    order, off, nodes = gi.order, inst["offsets"], inst["nodes"]
    chosen: List[int] = [int(c) for c in sr["chosen"]]
    instances = [[order[i] for i in nodes[off[c]:off[c + 1]]] for c in chosen]
    pat_pos: Dict[int, int] = {}
    pattern_of_instance = []
    for c in chosen:
        p = int(inst["pattern"][c])
        pattern_of_instance.append(pat_pos.setdefault(p, len(pat_pos)))
    area_used = float(sum(float(inst["pattern_area"][p]) for p in pat_pos))

    sig = SignatureIndex.from_graph(g).signature
    chk = validate.check_selection(g, instances, max_in, max_out, connected, area_budget, sig)
    if not chk["ok"]:  # pragma: no cover - guarded by construction
        raise RuntimeError(f"SARSA baseline produced an invalid selection: {chk['errors'][:5]}")
    savings = [validate.saving(g, S) for S in instances]
    if savings != [int(inst["gains"][c]) for c in chosen]:  # pragma: no cover
        raise RuntimeError("instance savings disagree with the cost model")
    accel = T_sw - sum(savings)
    speedup = 1.0 if T_sw <= 0 else (math.inf if accel <= 0 else T_sw / accel)

    training_cut = bool(sr["training_cut"]) and int(sr["episodes_done"]) < int(episodes)
    hit_limit = bool(er["timed_out"]) or bool(er["capped"]) or bool(inst["truncated"]) or training_cut or bool(sr["rollout_cut"])
    if hit_limit and not instances:
        status = "no_solution"
    elif hit_limit:
        status = "timeout_partial"
    else:
        status = "ok"

    diagnostics = {
        "T_sw": T_sw,
        "savings": savings,
        "time_budget_s": budget,
        "enum_share": float(enum_share),
        "enumeration": {"valid": int(er["valid"]), "stored": int(er["stored"]),
                        "explored_states": int(er["explored"]), "timed_out": bool(er["timed_out"]),
                        "capped": bool(er["capped"]), "max_candidates": int(max_candidates),
                        "elapsed_s": float(er["elapsed_s"]), "time_limit_s": float(enum_s)},
        "instances": {"num_instances": n_inst, "num_patterns": int(len(inst["pattern_area"])),
                      "truncated": bool(inst["truncated"]), "candidates_processed": int(inst["num_seen"]),
                      "unaffordable_instances": n_inst - int(sr.get("num_actions", 0)) if n_inst else 0,
                      "elapsed_s": float(build_s)},
        "sarsa": {"episodes_requested": int(episodes), "episodes_done": int(sr["episodes_done"]),
                  "training_cut": training_cut, "rollout_cut": bool(sr["rollout_cut"]),
                  "epsilon_last": float(sr["epsilon_last"]),
                  "dependency_rejections": int(sr["dependency_rejections"]),
                  "steps": int(sr["steps"]), "states": int(sr["states"]), "q_entries": int(sr["q_entries"]),
                  "train_s": float(sr["train_s"]), "rollout_s": float(sr["rollout_s"]),
                  "alpha": float(alpha), "gamma": float(gamma), "eps_start": float(eps_start),
                  "eps_end": float(eps_end), "seed": int(seed)},
    }
    return {
        "speedup": float(speedup),
        "area_used": area_used,
        "area_budget": area_budget,
        "a_ref": a_ref,
        "num_patterns": len(pat_pos),
        "num_instances": len(instances),
        "instances": instances,
        "pattern_of_instance": pattern_of_instance,
        "mean_ci_size": (float(sum(len(S) for S in instances)) / len(instances)) if instances else 0.0,
        "runtime_s": time.monotonic() - t0,
        "status": status,
        "num_candidates": int(er["valid"]),
        "diagnostics": diagnostics,
    }
