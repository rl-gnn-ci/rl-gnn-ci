"""SARSA baseline: exact convex enumeration, then tabular SARSA selection.

The baseline "Tabular RL on conflict graphs (SARSA)" of the TACO paper
(*Compared Methods*): candidates are enumerated exactly (Xiao et al., 2014)
and a tabular SARSA agent selects a conflict-free subset under the area
budget (Wang and Xiao, 2023).  Entry point: :func:`identify_sarsa`.

Stages
------
1. :mod:`~ci_ident.baselines.sarsa.enumeration` -- every legal (``ci_eligible``),
   convex, I/O-bounded and (by default) weakly connected node set of the DDG,
   by depth-first NodeFilter growth with sibling deletion and permanent-I/O
   pruning (C++ extension ``_xiao_enum``).  Every search state is expanded, so
   the candidate set is complete unless the deadline stops the search.
2. :mod:`~ci_ident.baselines.sarsa.instances` -- per candidate the whole-cycle
   saving ``|S| - cycles(cp(S))`` and the 1-WL pattern signature shared with
   the rest of the repository; instances without saving are dropped, area is
   charged once per pattern.
3. Selection (C++ extension ``_sarsa_core``) -- SARSA(0) over states = sets
   of selected instances, actions = instances that share no node with the
   selection, keep the contraction of the selection acyclic (dependencies
   taken over the full DDG, inadmissible nodes included) and fit the area
   budget, reward = instance saving; ``alpha = 0.012``, ``gamma = 0.95``,
   epsilon linear from 1.0 to 0.01, 10,000 episodes, then one greedy
   rollout.

Time: one monotonic deadline per scenario; the enumeration may use
``enum_share`` (0.7) of it.  Expiry stops the current stage and the result is
the selection obtained so far (``status="timeout_partial"`` or
``"no_solution"``).
Memory: the enumeration stores at most ``max_candidates`` candidates
(default 5,000,000) and stops there, with the same status.

The C++ extensions are built with ``python -m ci_ident.baselines.sarsa.build``.
This package imports without them; :func:`available` reports whether they
are present and :func:`identify_sarsa` raises :class:`ExtensionUnavailable`
if they are not.
"""

from __future__ import annotations

from ci_ident.baselines.sarsa._ext import ExtensionUnavailable, available
from ci_ident.baselines.sarsa.api import identify_sarsa

__all__ = ["identify_sarsa", "available", "ExtensionUnavailable"]
