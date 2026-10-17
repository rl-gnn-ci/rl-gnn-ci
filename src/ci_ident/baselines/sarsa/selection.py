"""Second stage of the SARSA baseline: tabular SARSA over CI instances.

:func:`select` runs the C++ extension ``_sarsa_core`` (its header comment
gives the MDP and the update) on a list of instances.  The dependency
relation is evaluated on the full DDG ``g``, inadmissible nodes included, and
over the whole selection: an instance is an action only if contracting it
together with all selected instances leaves the DDG acyclic.  Node overlap
and the area budget (area charged once per pattern) complete the action
filter.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Optional, Sequence

import numpy as np
import networkx as nx

from ci_ident.baselines.sarsa import _ext
from ci_ident.baselines.sarsa.enumeration import GraphIndex
from ci_ident.costs import EPS

Node = Hashable


def select(g: nx.DiGraph, instances: Sequence[Sequence[Node]], gains: Sequence[float],
           pattern: Sequence[int], pattern_area: Sequence[float], area_budget: float, *,
           episodes: int = 10_000, alpha: float = 0.012, gamma: float = 0.95,
           eps_start: float = 1.0, eps_end: float = 0.01, seed: int = 0,
           seconds: float = math.inf, index: Optional[GraphIndex] = None) -> Dict[str, Any]:
    """Run SARSA on the given instances (node-name lists, in priority order:
    greedy ties go to the earlier instance).

    Returns the core's result dict; ``chosen`` holds instance positions in
    selection order.
    """
    gi = index if index is not None else GraphIndex(g)
    flat, off = [], [0]
    for S in instances:
        idx = sorted(gi.idx[v] for v in S)
        flat.extend(idx)
        off.append(len(flat))
    core = _ext.load("_sarsa_core").Sarsa(
        gi.num_nodes, gi.edges, np.asarray(flat, dtype=np.uint32), np.asarray(off, dtype=np.uint64),
        np.asarray(gains, dtype=np.float64), np.asarray(pattern, dtype=np.int32),
        np.asarray(pattern_area, dtype=np.float64), float(area_budget), EPS)
    return run_core(core, episodes=episodes, alpha=alpha, gamma=gamma, eps_start=eps_start,
                    eps_end=eps_end, seed=seed, seconds=seconds)


def run_core(core, *, episodes: int, alpha: float, gamma: float, eps_start: float, eps_end: float,
             seed: int, seconds: float) -> Dict[str, Any]:
    secs = math.inf if seconds is None else float(seconds)
    res = dict(core.run(float(alpha), float(gamma), float(eps_start), float(eps_end), int(episodes),
                        int(seed) & 0xFFFFFFFFFFFFFFFF, secs))
    res["chosen"] = [int(c) for c in res["chosen"]]
    return res
