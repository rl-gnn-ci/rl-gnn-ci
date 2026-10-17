"""Custom-instruction identification as a constrained MDP (TACO formulation).

One episode is one DDG.  The agent grows a candidate ``C`` one node at a time
and commits it with STOP; area is charged once per pattern (1-WL signature).

Actions: ``Discrete(max_nodes + 1)``; action ``i < |V|`` adds node ``i``
(canonical DOT order), action ``max_nodes`` is STOP; indices in
``[|V|, max_nodes)`` are padding and never admissible.

Growth predicate (FeasGrow): the node is free, CI-admissible, adjacent to ``C``
(any free node when ``C`` is empty), and ``C + v`` is convex and within the
port bound.  Commit predicate (FeasCommit): ``C`` is non-empty, within the port
bound, convex, and the area of a *new* pattern fits the remaining budget (a
repeated pattern costs no area).

Masks:

* ``mask="masked"``, ``mode="train"``: the frontier (free, admissible, not
  refused for this candidate, adjacent to ``C``).  The predicates are checked
  after the step.  A violation of the permanent port bound (ports that no
  further growth can remove), a convexity violation that no further growth can
  repair (a fixed node between two candidate nodes) or an area overflow is
  rolled back with a penalty and the node is refused for this candidate.  The
  regular port bound may be exceeded temporarily while growing and is enforced
  at commit; candidate-internal convexity debt (free nodes between candidate
  nodes) is likewise tolerated during growth and enforced at commit.
* ``mask="masked"``, ``mode="inference"``: the frontier filtered by the full
  growth predicate (port bound and exact convexity of ``C + v``).  Area is not
  in the mask; an overflow is rolled back as in training.
* ``mask="none"``: every node and STOP, in both modes; illegal actions are
  refused with a penalty.

STOP is in the mask iff ``C`` is non-empty (masked variants).

Reward (all variants; switches from :class:`ci_ident.env.config.EnvConfig`):

* commit: ``w_perf * [log S_{t+1} - log S_t]`` (+ ``(w_size/N) h(|C|)`` for
  Base/NoMask/AbsGain), where ``S`` is the block speedup of the committed set.
  Summed over an episode the commit terms equal ``w_perf * log S_final``.
* growth: ``w_ext * eta``, ``eta = max(G(C'),0) (1 - h(|C'|)) 1[FeasCommit(C')]``
  (``eta = 1`` for NoShape), ``G = rho = Delta/l_SW`` (``G = Delta`` for AbsGain).
* potential shaping (not NoShape) on every transition:
  ``(w_shp/N) [gamma Phi(s') - Phi(s)]`` with
  ``Phi = G(C) - lambda_io d_io(C) - lambda_conv d_conv(C)`` and ``Phi = 0``
  for an empty candidate and for terminal states, so the discounted shaping of
  an episode sums to zero (Ng, Harada and Russell, 1999).
* penalties (``/N``): illegal node action, infeasible STOP (the candidate is
  discarded), empty STOP.
* a discarded candidate's extension rewards are charged back at the discard
  (compounded by ``1/gamma`` per step since they were paid), so extension
  rewards are earned only by candidates that are committed.

Termination: the free pool is exhausted; or the candidate is empty and no free
node can start a feasible CI (its permanent ports exceed the bound or its area
exceeds the remaining budget; with the inference mask, no node is admissible).
Truncation: ``max_steps_factor * |V|`` decisions.  A feasible non-empty
candidate is committed (and paid) when the episode ends, otherwise discarded.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ci_ident.costs import EPS, cycles
from ci_ident.data.corpus import Corpus, load_corpus
from ci_ident.ddg.graph import load_ddg
from ci_ident.ddg.signature import SignatureIndex
from ci_ident.ddg.state import DDGState, StaticDDG
from ci_ident.ddg.status import NUM_STATUSES, SELECTED_GROUP, NodeStatus
from ci_ident.env.config import EnvConfig

STOP = object()                         # sentinel for the STOP action
N_GRAPH_FEATS = 12


class _StaticCache:
    """Process-wide read-only cache of parsed graphs, keyed by DOT path."""

    def __init__(self, max_items: int = 512):
        self.max_items = max_items
        self._d: Dict[str, Tuple[StaticDDG, SignatureIndex, dict]] = {}

    def get(self, dot_path: str):
        hit = self._d.get(dot_path)
        if hit is None:
            g = load_ddg(dot_path)
            static = StaticDDG.from_graph(g)
            hit = (static, SignatureIndex.from_graph(g),
                   {"pos": {n: i for i, n in enumerate(static.nodes)}})
            if len(self._d) >= self.max_items:
                self._d.pop(next(iter(self._d)))
            self._d[dot_path] = hit
        return hit


_CACHE = _StaticCache()


class CIEnv(gym.Env):
    """See the module docstring.

    ``env_config`` keys: every :class:`EnvConfig` field (or ``variant`` in
    place of ``mask``/``reward``) plus ``corpus`` (a corpus manifest,
    required), ``graphs`` (corpus indices visited in order, one per reset;
    default: all), ``seed`` and ``cache_size`` (parsed graphs kept per process).
    """

    metadata: Dict[str, Any] = {}

    def __init__(self, env_config: Optional[dict] = None):
        cfg = dict(env_config or {})
        corpus_path = cfg.pop("corpus", None)
        if corpus_path is None:
            raise KeyError("env_config['corpus'] (a corpus manifest) is required")
        graphs = cfg.pop("graphs", None)
        seed = int(cfg.pop("seed", 0))
        cache_size = cfg.pop("cache_size", None)
        if cache_size is not None:
            _CACHE.max_items = max(1, int(cache_size))
        self.config = EnvConfig.from_dict(cfg)
        self.corpus: Corpus = load_corpus(str(corpus_path))
        self.graph_indices: List[int] = (list(range(len(self.corpus))) if graphs is None
                                         else [int(i) for i in graphs])
        if not self.graph_indices:
            raise ValueError("no graphs to run on")
        self._rng = np.random.default_rng(seed)
        self._cursor = -1
        self.max_in, self.max_out = self.config.io
        self.max_nodes = int(self.config.max_nodes)
        self.observation_space = spaces.Dict({
            "node_status_vector": spaces.Box(0, NUM_STATUSES - 1, (self.max_nodes,), np.int64),
            "graph_feats": spaces.Box(0.0, 1.0, (N_GRAPH_FEATS,), np.float32),
            "action_mask": spaces.MultiBinary(self.max_nodes + 1),
            "graph_idx": spaces.Box(0, len(self.corpus) - 1, (1,), np.int64),
            "num_nodes": spaces.Box(0, self.max_nodes, (1,), np.int64),
        })
        self.action_space = spaces.Discrete(self.max_nodes + 1)
        self._load_graph(self.graph_indices[0])

    # ==================================================================
    # episode state
    # ==================================================================
    def _load_graph(self, index: int) -> None:
        dot = str(self.corpus.dot_path(index))
        static, sig, extra = _CACHE.get(dot)
        if static.num_nodes > self.max_nodes:
            raise ValueError(f"{dot}: {static.num_nodes} nodes > max_nodes={self.max_nodes}")
        self.graph_idx = int(index)
        self.static, self.sig_index, self._pos = static, sig, extra["pos"]
        self.ddg = DDGState(static, self.max_in, self.max_out)
        self.N = static.num_nodes
        self.max_area = float(self.config.area_factor) * static.total_area
        self.sw_total = static.sw_total
        self.max_steps = max(1, int(self.config.max_steps_factor * self.N))

        self.ci: List[str] = []
        self.ci_sw = 0.0
        self.ci_hw = 0.0
        self.ci_area = 0.0
        self.blocked: Set[str] = set()
        self.committed: List[List[str]] = []
        self.pattern_sigs: Set[tuple] = set()
        self.saved_total = 0.0
        self.area_used = 0.0
        self.kept: List[List[str]] = []
        self.kept_sigs: Set[tuple] = set()
        self.kept_saved = 0.0
        self.kept_area = 0.0
        self.speedup_paid = 1.0
        self.phi = 0.0
        self.ext_open = 0.0
        self.steps = 0
        self.violations = 0
        self.done = False
        self.stats = {"commits": 0, "r_commit": 0.0, "r_size": 0.0, "r_ext": 0.0,
                      "r_shape": 0.0, "r_penalty": 0.0}
        self._version = 0
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_ver = -1

    def _bump(self) -> None:
        self._version += 1

    # ==================================================================
    # Gym API
    # ==================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        idx = None if not options else options.get("graph_idx")
        if idx is None:
            self._cursor = (self._cursor + 1) % len(self.graph_indices)
            idx = self.graph_indices[self._cursor]
        self._load_graph(int(idx))
        return self._observation(), {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        action = int(action)
        cfg, W = self.config, self.config.weights
        N = max(1, self.N)
        if 0 <= action < self.N:
            node = self.static.nodes[action]
        elif action == self.max_nodes:
            node = STOP
        else:
            node = None
        self.steps += 1
        phi_before = self.phi
        self.ext_open /= cfg.gamma        # extension value of the open candidate, at this step
        rew = 0.0

        if node is STOP:
            if not self.ci:
                rew += self._penalty(W.w_empty_stop / N)
            elif self._commit_feasible():
                rew += self._pay_commit()
                self._commit()
            else:
                rew += self._penalty(W.w_bad_stop / N)
                rew += self._discard()
        elif node is None or not self._mask()[action] or not self._selectable(node):
            rew += self._penalty(W.w_invalid / N)            # illegal: state unchanged
        elif not self._grow(node):
            rew += self._penalty(W.w_invalid / N)            # rolled back
        else:
            eta = self._eta() if cfg.reward != "noshape" else 1.0
            r = W.w_ext * eta
            self.stats["r_ext"] += r
            self.ext_open += r
            rew += r

        # ---- end of episode ------------------------------------------------
        terminated = truncated = False
        reasons: List[str] = []
        if not self.ddg.nodes_pool:
            terminated = True
            reasons.append("pool_exhausted")
        elif not self.ci and not self._can_start():
            terminated = True
            reasons.append("no_feasible_start")
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            reasons.append("max_steps")
        if (terminated or truncated) and self.ci:
            if self._commit_feasible():
                rew += self._pay_commit()
                self._commit()
            else:
                rew += self._discard()

        # ---- potential shaping on the transition ----------------------------
        self.phi = 0.0 if (terminated or not self.ci) else self._potential()
        if cfg.shaping:
            f = (W.w_shp / N) * (cfg.gamma * self.phi - phi_before)
            self.stats["r_shape"] += f
            rew += f

        info: Dict[str, Any] = {"graph_idx": self.graph_idx}
        if terminated or truncated:
            self.done = True
            info.update(self.episode_summary())
            info["end_reasons"] = reasons
        return self._observation(), float(rew), terminated, truncated, info

    # ==================================================================
    # candidate lifecycle
    # ==================================================================
    def _penalty(self, r: float) -> float:
        self.violations += 1
        self.stats["r_penalty"] += r
        return r

    def _grow(self, node: str) -> bool:
        """Add ``node``; roll back (and refuse it for this candidate) if a
        predicate checked after the step fails."""
        ddg = self.ddg
        prev = ddg.pattern[node]
        ddg.pattern[node] = NodeStatus.SELECTED
        self.ci.append(node)
        delta = ddg.update_io(node)
        lat = float(cycles(self.static.latency[node]))
        self.ci_sw += lat
        self.ci_area += self.static.area[node]
        self.ci_hw = self.static.critical_path(self.ci)
        ok = ddg.convexity_after_select(node)       # no fixed node between candidate nodes
        ok = ok and ddg.perm_io_ok() and self._area_ok()
        if ok and self.config.mode == "inference" and self.config.mask == "masked":
            ok = ddg.io_ok() and ddg.convexity_ok()
        if not ok:
            self.ci.pop()
            self.ci_sw -= lat
            self.ci_area -= self.static.area[node]
            ddg.rollback_io(delta)
            ddg.convexity_rollback(node)
            ddg.pattern[node] = prev
            self.ci_hw = self.static.critical_path(self.ci)
            self.blocked.add(node)
        else:
            ddg.nodes_pool.discard(node)
        self._bump()
        return ok

    def _clear(self) -> None:
        self.ext_open = 0.0
        self.ci = []
        self.ci_sw = self.ci_hw = self.ci_area = 0.0
        self.blocked.clear()
        self.ddg.reset_candidate()
        self._bump()

    def _discard(self) -> float:
        """Return the candidate's nodes to the pool.  Returns the charge-back of
        the extension rewards paid for the discarded candidate (compounded by
        1/gamma per step, so their discounted value is exactly cancelled)."""
        charge = -self.ext_open
        self.stats["r_ext"] += charge
        for n in self.ci:
            self.ddg.nodes_pool.add(n)
            self.ddg.pattern[n] = NodeStatus.NOT_SELECTED
        self._clear()
        return charge

    def _commit(self) -> None:
        nodes = list(self.ci)
        saved = self._saving()
        sig = self.sig_index.signature(nodes)
        self.committed.append(nodes)
        self.saved_total += saved
        if sig not in self.pattern_sigs:
            self.pattern_sigs.add(sig)
            self.area_used += self.ci_area
        if len(nodes) > 1 and saved > 0:
            self.kept.append(nodes)
            self.kept_saved += saved
            if sig not in self.kept_sigs:
                self.kept_sigs.add(sig)
                self.kept_area += self.ci_area
        self.stats["commits"] += 1
        for n in nodes:
            self.ddg.pattern[n] = NodeStatus.ALREADY_SELECTED
        self.ddg.register_committed_group(nodes)
        self._clear()

    # ==================================================================
    # predicates
    # ==================================================================
    def _area_ok(self) -> bool:
        return self.ci_area + self.area_used <= self.max_area + EPS

    def _selectable(self, node: str) -> bool:
        """Legality and connectivity of FeasGrow: free, admissible, not refused
        for this candidate, and adjacent to the candidate (if non-empty)."""
        if node not in self.ddg.nodes_pool or node in self.blocked:
            return False
        if not self.ci:
            return True
        cset = set(self.ci)
        return any(u in cset for u in self.static.predecessors[node] + self.static.successors[node])

    def _commit_feasible(self) -> bool:
        """FeasCommit for the current candidate."""
        ddg = self.ddg
        if not self.ci or not (ddg.perm_io_ok() and ddg.io_ok() and ddg.convexity_ok()):
            return False
        sig = self.sig_index.signature(self.ci)
        extra = 0.0 if sig in self.pattern_sigs else self.ci_area
        return self.area_used + extra <= self.max_area + EPS and self._saving() >= 0

    def _frontier(self) -> List[str]:
        pool = self.ddg.nodes_pool - self.blocked
        if not self.ci:
            cand = pool
        else:
            preds, succs = self.static.predecessors, self.static.successors
            cand = {v for u in self.ci for v in preds[u] + succs[u] if v in pool}
        return sorted(cand, key=self._pos.__getitem__)

    def _mask(self) -> np.ndarray:
        if self._mask_ver == self._version and self._mask_cache is not None:
            return self._mask_cache
        mask = np.zeros(self.max_nodes + 1, dtype=np.int8)
        if self.config.mask == "none":
            mask[:self.N] = 1
            mask[self.max_nodes] = 1
        else:
            strict = self.config.mode == "inference"
            reach = self.ddg.convexity_reach_sets() if strict else None
            for n in self._frontier():
                if strict:
                    i_in, i_out, p_in, p_out = self.ddg.estimate_io_after_add(n)
                    if max(i_in, p_in) > self.max_in or max(i_out, p_out) > self.max_out:
                        continue
                    if not self.ddg.convexity_ok_after_add(n, reach):
                        continue
                mask[self._pos[n]] = 1
            mask[self.max_nodes] = 1 if self.ci else 0
        self._mask_cache, self._mask_ver = mask, self._version
        return mask

    def _can_start(self) -> bool:
        """Some free node can start a feasible CI (empty candidate)."""
        if self.config.mask == "masked" and self.config.mode == "inference":
            return bool(self._mask()[:-1].any())
        slack = self.max_area - self.area_used + EPS
        for n in self.ddg.nodes_pool - self.blocked:
            if self.static.area[n] > slack:
                continue
            _, _, p_in, p_out = self.ddg.estimate_io_after_add(n)
            if p_in <= self.max_in and p_out <= self.max_out:
                return True
        return False

    # ==================================================================
    # objective and reward terms
    # ==================================================================
    def _saving(self) -> float:
        return max(0.0, self.ci_sw - cycles(self.ci_hw))

    def _gain(self) -> float:
        """G(C): rho = Delta / l_SW (Base, NoMask) or Delta (AbsGain)."""
        d = self._saving()
        return d if self.config.gain == "absolute" else d / max(self.ci_sw, 1.0)

    def _target_ci_size(self) -> int:
        """k_target of h(k) = min(1, k/k_target): the smaller of the port
        capacity max_in + max_out + 1 and the number of median-area free nodes
        that fit the remaining budget, at least 2."""
        io_cap = self.max_in + self.max_out + 1
        slack = max(self.max_area - self.area_used, 0.0)
        areas = [self.static.area[n] for n in self.ddg.nodes_pool - self.blocked] or [1.0]
        area_cap = int(slack / max(float(np.median(areas)), 1e-6))
        return max(2, min(io_cap, area_cap))

    def _h(self, k: int) -> float:
        return min(1.0, k / max(1, self._target_ci_size()))

    def _eta(self) -> float:
        if not self._commit_feasible():
            return 0.0
        return max(self._gain(), 0.0) * (1.0 - self._h(len(self.ci)))

    def _trapped_fraction(self) -> float:
        """d_conv: fraction of free nodes lying between candidate nodes
        (convexity debt under contraction of the committed CIs)."""
        pool = self.ddg.nodes_pool
        if not pool:
            return 0.0
        anc, desc, patt = self.ddg._sel_anc, self.ddg._sel_desc, self.ddg.pattern
        k = sum(1 for n in pool if patt[n] not in SELECTED_GROUP and anc[n] and desc[n])
        return k / len(pool)

    def _potential(self) -> float:
        W = self.config.weights
        d_io = (max(len(self.ddg.input_set) - self.max_in, 0)
                + max(len(self.ddg.output_set) - self.max_out, 0))
        return self._gain() - W.lambda_io * d_io - W.lambda_conv * self._trapped_fraction()

    def speedup(self, include_candidate: bool = False) -> float:
        saved = self.saved_total + (self._saving() if include_candidate and self.ci else 0.0)
        rest = self.sw_total - saved
        return float("inf") if rest <= 0 else self.sw_total / rest

    def _pay_commit(self) -> float:
        """w_perf [log S_{t+1} - log S_t] (+ size term); call before _commit()."""
        W = self.config.weights

        def log_s(s: float) -> float:
            return math.log1p(max((1e12 if math.isinf(s) else s) - 1.0, 0.0))

        s_after = self.speedup(include_candidate=True)
        r = W.w_perf * (log_s(s_after) - log_s(self.speedup_paid))
        self.speedup_paid = s_after
        self.stats["r_commit"] += r
        if self.config.reward != "noshape":
            rs = (W.w_size / max(1, self.N)) * self._h(len(self.ci))
            self.stats["r_size"] += rs
            r += rs
        return r

    # ==================================================================
    # observation
    # ==================================================================
    def _graph_feats(self, mask: np.ndarray) -> np.ndarray:
        """12 progress features in [0, 1]: |V|/max_nodes, pool fraction, area
        utilisation (candidate included), candidate gain ratio rho(C),
        |In|, |permIn|, |Out|, |permOut| over their bounds, frontier size/|V|,
        admissible node actions/|V|, h(|C|), committed speedup as 1 - 1/S."""
        N = max(1, self.N)
        ddg = self.ddg
        area_util = (self.ci_area + self.area_used) / max(self.max_area, 1e-9)
        rho = self._saving() / max(self.ci_sw, 1.0)
        s = self.speedup()
        f = [
            min(1.0, N / self.max_nodes),
            len(ddg.nodes_pool) / N,
            min(1.0, area_util),
            rho,
            min(1.0, len(ddg.input_set) / self.max_in),
            min(1.0, len(ddg.perm_input_set) / self.max_in),
            min(1.0, len(ddg.output_set) / self.max_out),
            min(1.0, len(ddg.perm_output_set) / self.max_out),
            min(1.0, len(self._frontier()) / N),
            min(1.0, int(mask[:self.N].sum()) / N),
            self._h(len(self.ci)),
            0.0 if math.isinf(s) else 1.0 - 1.0 / s,
        ]
        return np.asarray(f, dtype=np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        status = np.zeros(self.max_nodes, dtype=np.int64)
        status[:self.N] = [self.ddg.pattern[n].value for n in self.static.nodes]
        mask = self._mask()
        return {
            "node_status_vector": status,
            "graph_feats": self._graph_feats(mask),
            "action_mask": mask.copy(),
            "graph_idx": np.asarray([self.graph_idx], dtype=np.int64),
            "num_nodes": np.asarray([self.N], dtype=np.int64),
        }

    # ==================================================================
    # results
    # ==================================================================
    def episode_summary(self) -> Dict[str, Any]:
        """Metrics of the committed CIs with more than one node and a
        positive saving (a one-node CI saves nothing)."""
        rest = self.sw_total - self.kept_saved
        n = len(self.kept)
        return {
            "speedup": float("inf") if rest <= 0 else self.sw_total / rest,
            "area_used": self.kept_area,
            "area_budget": self.max_area,
            "area_ref": self.static.total_area,
            "area_util": self.kept_area / max(self.max_area, 1e-9),
            "num_patterns": len(self.kept_sigs),
            "num_instances": n,
            "mean_ci_size": (sum(len(c) for c in self.kept) / n) if n else 0.0,
            "steps": self.steps,
            "violations": self.violations,
            "violation_rate": self.violations / max(1, self.steps),
            "commits": self.stats["commits"],
            **{k: v for k, v in self.stats.items() if k.startswith("r_")},
        }

    def committed_cis(self) -> List[List[str]]:
        return [list(c) for c in self.kept]
