"""Custom-instruction identification as a constrained sequential decision process.

One episode builds CIs on one DDG.  At every step the agent either adds a node
to the candidate CI under construction or chooses STOP, which commits the
candidate.  Every validity condition (admissibility, port bound, convexity
under contraction, global area budget) is enforced by the action mask, in
training exactly as at inference, so the policy is trained and evaluated on
one action set.

Action space: ``Discrete(max_nodes + 1)``; action ``i < |V|`` adds node ``i``
(canonical DOT order), action ``max_nodes`` is STOP; indices in
``[|V|, max_nodes)`` are padding and always masked.

Observation (``gymnasium.spaces.Dict``):

* ``node_status_vector`` -- status code per node (:class:`NodeStatus`), padded;
* ``node_dynamic_feats`` -- per node ``[depth_C(v), latency(v), height_C(v)] / S``
  where ``depth_C(v)`` (``height_C(v)``) is the longest latency-weighted path
  into (out of) ``v`` through the candidate ``C`` and ``S = max(1, ceil(cp(V)))``
  the block's own critical path, so the change ``v`` would make to the
  candidate's critical path is a function of the observation;
* ``graph_feats`` -- 16 scalars in [0, 1] (see :meth:`CIEnv._graph_feats`);
* ``action_mask`` -- the mask;
* ``graph_idx`` -- index of the graph in the corpus (the model's graph store
  serves the static graph tensor for this index);
* ``num_nodes`` -- ``|V|``, checked by the model against the stored graph.

Because every validity condition is in the mask, every non-empty candidate
reached through admissible actions is committable, and training and
evaluation use the same environment (evaluation only replaces sampling by the
policy's most probable action).  For callers that ignore the mask: a node
action outside the frontier leaves the state unchanged and counts towards the
no-progress limit; a growth that breaks the port bound or the area budget, or
places a fixed node between candidate nodes, is rolled back and counted as a
violation; a growth that leaves a free node between candidate nodes is kept,
and the candidate is then not committable; STOP on an empty or uncommittable
candidate discards it and counts as a violation.
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
from ci_ident.ddg.status import NUM_STATUSES, NodeStatus
from ci_ident.env.config import EnvConfig

STOP = object()                     # sentinel for the STOP action (never a node name)
N_GRAPH_FEATS = 16
N_NODE_DYN = 3


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
            sig = SignatureIndex.from_graph(g)
            pos = {n: i for i, n in enumerate(static.nodes)}
            lat_arr = [static.latency[n] for n in static.nodes]
            extra = {
                "pos": pos,
                "topo_pos": [0] * static.num_nodes,
                "lat_arr": lat_arr,
                "pred_idx": sig.pred,
                "succ_idx": sig.succ,
                "scale": max(1.0, float(cycles(static.critical_path(static.nodes)))),
            }
            for k, n in enumerate(static.topo_order):
                extra["topo_pos"][pos[n]] = k
            hit = (static, sig, extra)
            if len(self._d) >= self.max_items:
                self._d.pop(next(iter(self._d)))
            self._d[dot_path] = hit
        return hit


_CACHE = _StaticCache()


class CIEnv(gym.Env):
    """See module docstring.

    ``env_config`` keys: every :class:`EnvConfig` field plus

    * ``corpus`` -- path of a corpus manifest (:mod:`ci_ident.data.corpus`);
    * ``graphs`` -- optional list of corpus indices visited in order, one per
      reset (evaluation); without it every reset draws a graph from the whole
      corpus with the size sampler (training);
    * ``sampler`` -- ``{"size_min", "size_max", "exponent", "mode"}`` for the
      size sampler (:meth:`ci_ident.data.corpus.Corpus.sampler`);
    * ``seed`` -- base seed (RLlib's ``worker_index``/``vector_index`` are mixed in);
    * ``cache_size`` -- parsed graphs kept per process (default 512, about
      0.7 MB per 400-node graph).
    """

    metadata: Dict[str, Any] = {}

    def __init__(self, env_config: Optional[dict] = None):
        cfg = dict(env_config or {})
        worker = int(getattr(env_config, "worker_index", cfg.pop("worker_index", 0)) or 0)
        vector = int(getattr(env_config, "vector_index", cfg.pop("vector_index", 0)) or 0)
        corpus_path = cfg.pop("corpus", None)
        if corpus_path is None:
            raise KeyError("env_config['corpus'] (a corpus manifest) is required")
        graphs = cfg.pop("graphs", None)
        sampler_cfg = dict(cfg.pop("sampler", {}) or {})
        base_seed = int(cfg.pop("seed", 0))
        cache_size = cfg.pop("cache_size", None)       # parsed graphs kept per process
        if cache_size is not None:
            _CACHE.max_items = max(1, int(cache_size))
        self.config = EnvConfig.from_dict(cfg)

        self.corpus: Corpus = load_corpus(str(corpus_path))
        self.graph_indices: List[int] = (list(range(len(self.corpus))) if graphs is None
                                         else [int(i) for i in graphs])
        if not self.graph_indices:
            raise ValueError("no graphs to run on")
        self._rng = np.random.default_rng(
            (base_seed * 1_000_003 + 97 * worker + 17 * vector) & 0x7FFFFFFF)
        self._sampler = self.corpus.sampler(**sampler_cfg) if graphs is None else None
        self._cursor = -1

        self.max_in, self.max_out = self.config.io
        self.max_nodes = int(self.config.max_nodes)
        self.observation_space = spaces.Dict({
            # Integer status codes; the model embeds them.
            "node_status_vector": spaces.Box(0, NUM_STATUSES - 1, (self.max_nodes,), np.int64),
            "node_dynamic_feats": spaces.Box(0.0, 1.0, (self.max_nodes, N_NODE_DYN), np.float32),
            "graph_feats": spaces.Box(0.0, 1.0, (N_GRAPH_FEATS,), np.float32),
            "action_mask": spaces.MultiBinary(self.max_nodes + 1),
            "graph_idx": spaces.Box(0, len(self.corpus) - 1, (1,), np.int64),
            "num_nodes": spaces.Box(0, self.max_nodes, (1,), np.int64),
        })
        self.action_space = spaces.Discrete(self.max_nodes + 1)
        self._load_graph(self._next_graph_index())

    # ==================================================================
    # graph selection and episode state
    # ==================================================================
    def _next_graph_index(self) -> int:
        if self._sampler is not None:
            return self._sampler(self._rng)
        self._cursor = (self._cursor + 1) % len(self.graph_indices)
        return self.graph_indices[self._cursor]

    def _load_graph(self, index: int) -> None:
        dot = str(self.corpus.dot_path(index))
        static, sig, extra = _CACHE.get(dot)
        if static.num_nodes > self.max_nodes:
            raise ValueError(f"{dot}: {static.num_nodes} nodes > max_nodes={self.max_nodes}")
        self.graph_idx = int(index)
        self.static, self.sig_index, self._x = static, sig, extra
        self.ddg = DDGState(static, self.max_in, self.max_out)
        N = static.num_nodes
        self.max_area = float(self.config.area_factor) * static.total_area
        self.sw_total = static.sw_total
        self.np_limit = int(max(6, min(70, math.ceil(self.config.no_progress_base * math.log2(max(4, N))))))
        self.max_violations = int(self.config.violation_factor * max(1, N))

        self.ci: List[str] = []                 # candidate under construction
        self.committed: List[List[str]] = []
        self.blocked: Set[str] = set()          # nodes refused for this candidate

        self.ci_sw = 0.0
        self.ci_hw = 0.0
        self.ci_area = 0.0
        self.saved_total = 0.0                  # cycles saved by committed CIs
        self.area_used = 0.0                    # area charged (once per pattern)
        self.pattern_sigs: Set[tuple] = set()
        self.kept: List[List[str]] = []         # committed CIs with >1 node and a saving
        self.kept_sigs: Set[tuple] = set()
        self.kept_saved = 0.0
        self.kept_area = 0.0

        self.phi = 0.0
        self.speedup_paid = 1.0                 # speedup already paid by commit rewards
        self.steps = 0
        self.no_progress = 0
        self.stall = 0
        self.violations = 0
        self.stats = {"steps": 0, "commits": 0, "violations": 0, "r_commit": 0.0,
                      "r_shape": 0.0}
        self._version = 0
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_ver = -1
        self._frontier_cache: Optional[List[str]] = None
        self._frontier_ver = -1
        self._commit_cache: Optional[bool] = None
        self._commit_ver = -1

    def _bump(self) -> None:
        self._version += 1

    # ==================================================================
    # Gym API
    # ==================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        idx = None if not options else options.get("graph_idx")
        self._load_graph(self._next_graph_index() if idx is None else int(idx))
        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        N = self.static.num_nodes
        if 0 <= action < N:
            node = self.static.nodes[action]
        elif action == self.max_nodes:
            node = STOP
        else:
            node = None                                      # padding index
        self.steps += 1
        self.stats["steps"] += 1

        phi_before = self.phi
        attempted = growth_ok = False
        mask_after: Optional[np.ndarray] = None

        # ---- growth -------------------------------------------------
        if node not in (None, STOP) and node in self._selectable():
            attempted = True
            growth_ok = self._grow(node)
            self.phi = self._potential()
            mask_after = self._mask()

        stop = node == STOP
        commit_ok = self._commit_valid() if stop else False

        # ---- progress / stall accounting ------------------------------
        if not stop:
            if attempted and growth_ok:
                self.no_progress = 0 if (self.phi - phi_before) > self.config.no_progress_eps \
                    else self.no_progress + 1
            else:
                self.no_progress += 1
            if mask_after is None:
                mask_after = self._mask()
            if self.ci and not mask_after[:-1].any():
                self.stall += 1

        # ---- reward of the decision --------------------------------------
        rew = 0.0
        if stop:
            if self.ci and commit_ok:
                rew += self._commit_payment()
        elif attempted and growth_ok:
            rew += self._shaping(self.phi, phi_before)
        phi_pre_close = self.phi
        cand_before = bool(self.ci)

        if attempted and growth_ok:
            self.ddg.nodes_pool.discard(node)
            self._bump()

        # ---- STOP -----------------------------------------------------
        if stop:
            if not self.ci:
                self._discard()
                self.violations += 1
                self.stall = 0
            elif commit_ok:
                self._commit()
                self.stall = 0
            else:
                self._discard()
                self.violations += 1
                self.stall = 0

        # ---- episode limits -----------------------------------------------
        truncated = False
        reasons: List[str] = []
        if self.steps >= int(self.config.max_steps_factor * N):
            truncated = True
            reasons.append("max_steps")
        if not self._mask().any():
            truncated = True
            reasons.append("no_valid_nodes")
        if self.no_progress >= self.np_limit:
            truncated = True
            reasons.append("no_progress")
        if self.stall >= max(1, int(self.config.stall_patience)):
            truncated = True
            reasons.append("stalled")
        if self.violations >= self.max_violations:
            truncated = True
            reasons.append("violations")
        dead_end = reasons == ["no_valid_nodes"]   # absorbing: nothing can be earned
        if dead_end:
            truncated = False

        # ---- close out the candidate at the end of the pool / episode -------
        if not self.ddg.nodes_pool and self.ci:
            if self._commit_valid():
                rew += self._commit_payment()
                self._commit()
            else:
                self._discard()
                self.violations += 1
            self.stall = 0
        if (truncated or dead_end) and self.ci:
            if self._commit_valid():
                rew += self._commit_payment()
                self._commit()
            else:
                self._discard()
            self.stall = 0

        # ---- charge back the potential of a cleared candidate -------------
        if cand_before and not self.ci:
            rew += self._shaping(0.0, phi_pre_close)

        terminated = (not self.ddg.nodes_pool or dead_end) and not truncated
        self.stats["violations"] = self.violations
        info: Dict[str, Any] = {"graph_idx": self.graph_idx}
        if terminated or truncated:
            info.update(self.episode_summary())
            info["end_reasons"] = reasons or (["pool_exhausted"] if not self.ddg.nodes_pool else [])
        return self._observation(), float(rew), bool(terminated), bool(truncated), info

    # ==================================================================
    # candidate lifecycle
    # ==================================================================
    def _grow(self, node: str) -> bool:
        ddg = self.ddg
        prev = ddg.pattern[node]
        ddg.pattern[node] = NodeStatus.SELECTED
        self.ci.append(node)
        delta = ddg.update_io(node)
        lat = float(cycles(self.static.latency[node]))
        area = self.static.area[node]
        self.ci_sw += lat
        self.ci_hw = self.static.critical_path(self.ci)
        self.ci_area += area

        ok = (ddg.perm_io_ok() and ddg.io_ok() and self._area_ok()
              and ddg.convexity_after_select(node))
        if not ok:
            # Reached only by an action outside the mask.
            self.violations += 1
            self.blocked.add(node)
            self.ci.pop()
            self.ci_sw -= lat
            self.ci_area -= area
            ddg.rollback_io(delta)
            ddg.convexity_rollback(node)
            self.ci_hw = self.static.critical_path(self.ci)
            ddg.pattern[node] = prev
        self._bump()
        return ok

    def _clear_candidate(self) -> None:
        self.blocked.clear()
        self.ddg.reset_candidate()
        self.ci_sw = self.ci_hw = self.ci_area = 0.0
        self.phi = 0.0
        self._bump()

    def _discard(self) -> None:
        for n in self.ci:
            self.ddg.nodes_pool.add(n)
            self.ddg.pattern[n] = NodeStatus.NOT_SELECTED
        self.ci = []
        self._clear_candidate()

    def _commit(self) -> None:
        nodes = list(self.ci)
        saved = self._candidate_saving()
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
        self.ci = []
        self._clear_candidate()

    # ==================================================================
    # validity
    # ==================================================================
    def _area_ok(self) -> bool:
        return (self.ci_area + self.area_used) <= self.max_area + EPS

    def _commit_valid(self) -> bool:
        if self._commit_ver == self._version and self._commit_cache is not None:
            return self._commit_cache
        ddg = self.ddg
        ok = (ddg.perm_io_ok() and ddg.io_ok() and self._area_ok() and ddg.convexity_ok())
        self._commit_cache, self._commit_ver = bool(ok), self._version
        return self._commit_cache

    def _frontier(self) -> List[str]:
        """Pool nodes within ``radius`` hops (undirected DDG) of the candidate;
        the whole pool if the candidate is empty.  Sorted by node index."""
        if self._frontier_ver == self._version and self._frontier_cache is not None:
            return self._frontier_cache
        pool = self.ddg.nodes_pool - self.blocked
        pos = self._x["pos"]
        if not pool:
            out: List[str] = []
        elif not self.ci:
            out = sorted(pool, key=pos.__getitem__)
        else:
            S = set(self.ci)
            preds, succs = self.static.predecessors, self.static.successors
            cand: Set[str] = set()
            shell, seen = set(self.ci), set(self.ci)
            for _ in range(int(self.config.radius)):
                nxt: Set[str] = set()
                for u in shell:
                    for v in preds[u] + succs[u]:
                        if v in seen:
                            continue
                        nxt.add(v)
                        if v in pool and v not in S:
                            cand.add(v)
                if not nxt:
                    break
                seen |= nxt
                shell = nxt
            out = sorted(cand, key=pos.__getitem__)
        self._frontier_cache, self._frontier_ver = out, self._version
        return out

    def _selectable(self) -> Set[str]:
        return set(self._frontier())

    def _mask(self) -> np.ndarray:
        """Admissible actions: frontier nodes whose addition keeps the
        candidate within the port bound, convex under contraction and within
        the area budget; STOP iff the candidate is non-empty and committable,
        or no node action remains (a non-empty candidate is then closed)."""
        if self._mask_ver == self._version and self._mask_cache is not None:
            return self._mask_cache
        mask = np.zeros(self.max_nodes + 1, dtype=np.int8)
        ddg = self.ddg
        pos = self._x["pos"]
        reach = ddg.convexity_reach_sets()
        any_node = False
        for n in self._frontier():
            i_in, i_out, p_in, p_out = ddg.estimate_io_after_add(n)
            if p_in > self.max_in or p_out > self.max_out:
                continue
            if i_in > self.max_in or i_out > self.max_out:
                continue
            if not ddg.convexity_ok_after_add(n, reach):
                continue
            if self.ci_area + self.area_used + self.static.area[n] > self.max_area + EPS:
                continue
            mask[pos[n]] = 1
            any_node = True
        if self.ci:
            can_stop = self._commit_valid() or not any_node
        else:
            can_stop = False
        mask[self.max_nodes] = 1 if can_stop else 0
        self._mask_cache, self._mask_ver = mask, self._version
        return mask

    # ==================================================================
    # objective and reward
    # ==================================================================
    def _candidate_saving(self) -> float:
        return max(0.0, self.ci_sw - cycles(self.ci_hw))

    def speedup(self, include_candidate: bool = False) -> float:
        saved = self.saved_total + (self._candidate_saving() if include_candidate and self.ci else 0.0)
        rest = self.sw_total - saved
        return float("inf") if rest <= 0 else self.sw_total / rest

    def _potential(self) -> float:
        """Phi(C) = (ell_SW(C) - cp(C)) - area_price * A(C) / A_budget, with the
        critical path taken before rounding to whole cycles."""
        p = max(self.ci_sw - self.ci_hw, 0.0)
        price = self.config.reward.area_price
        if price:
            p -= price * self.ci_area / max(self.max_area, 1e-9)
        return p

    def _shaping(self, phi_new: float, phi_old: float) -> float:
        r = (self.config.reward.shape_coef / max(1, self.static.num_nodes)) * (phi_new - phi_old)
        self.stats["r_shape"] += r
        return r

    def _commit_payment(self) -> float:
        def log_s(s: float) -> float:
            return math.log1p(max((1e12 if math.isinf(s) else s) - 1.0, 0.0))

        s_after = self.speedup(include_candidate=True)
        r = self.config.reward.commit_coef * (log_s(s_after) - log_s(self.speedup_paid))
        self.speedup_paid = s_after
        self.stats["r_commit"] += r
        return r

    # ==================================================================
    # observation
    # ==================================================================
    def _target_ci_size(self) -> int:
        io_cap = self.max_in + self.max_out + 1
        slack = max(self.max_area - self.area_used, 0.0)
        areas = [self.static.area[n] for n in self.ddg.nodes_pool - self.blocked] or [1.0]
        area_cap = int(slack / max(float(np.median(areas)), 1e-6))
        return max(2, min(io_cap, area_cap))

    def _best_single_saving(self, legal: np.ndarray) -> int:
        """Largest increase of the candidate's saving obtainable by one admissible node."""
        if not self.ci or legal.size == 0:
            return 0
        sw0 = int(self.ci_sw)
        s0 = sw0 - cycles(self.ci_hw)
        best = 0
        for a in legal:
            n = self.static.nodes[int(a)]
            lat = cycles(self.static.latency[n])
            hw1 = self.static.critical_path(self.ci + [n])
            d = (sw0 + lat) - cycles(hw1) - s0
            if d > best:
                best = d
        return int(best)

    def _structure(self) -> Tuple[np.ndarray, float]:
        """Per-node [depth_C, latency, height_C] / S and the candidate's raw critical path."""
        x = self._x
        inv = 1.0 / x["scale"]
        dyn = np.zeros((self.max_nodes, N_NODE_DYN), dtype=np.float32)
        lat = x["lat_arr"]
        n_real = len(lat)
        dyn[:n_real, 1] = np.asarray(lat, dtype=np.float32) * inv
        if not self.ci:
            return dyn, 0.0
        pos, pred, succ, topo = x["pos"], x["pred_idx"], x["succ_idx"], x["topo_pos"]
        cidx = [pos[n] for n in self.ci]
        cset = set(cidx)
        order = sorted(cidx, key=topo.__getitem__)
        L: Dict[int, float] = {}
        hw = 0.0
        for v in order:
            best = lat[v]
            for p in pred[v]:
                if p in cset and p in L and lat[v] + L[p] > best:
                    best = lat[v] + L[p]
            L[v] = best
            hw = max(hw, best)
        H: Dict[int, float] = {}
        for v in reversed(order):
            best = lat[v]
            for s in succ[v]:
                if s in cset and s in H and lat[v] + H[s] > best:
                    best = lat[v] + H[s]
            H[v] = best
        dep: Dict[int, float] = {}
        hei: Dict[int, float] = {}
        for p in order:
            for v in succ[p]:
                if L[p] > dep.get(v, 0.0):
                    dep[v] = L[p]
        for s in order:
            for v in pred[s]:
                if H[s] > hei.get(v, 0.0):
                    hei[v] = H[s]
        for v, val in dep.items():
            dyn[v, 0] = val * inv
        for v, val in hei.items():
            dyn[v, 2] = val * inv
        return dyn, hw

    def _graph_feats(self, mask: np.ndarray, hw_c: float) -> np.ndarray:
        """16 progress features, all in [0, 1]:
        0 |V|/max_nodes, 1 pool fraction, 2 area utilisation (candidate counted),
        3 candidate saving ratio, 4-7 regular/permanent input and output use,
        8 frontier size/|V|, 9 admissible node actions/|V|, 10 no-progress/patience,
        11 candidate size/target size, 12 best one-node saving gain (squashed),
        13 area utilisation if the candidate were committed (pattern-deduplicated),
        14 cp(C)/S, 15 ceil(cp(C))/S."""
        N = max(1, self.static.num_nodes)
        ddg = self.ddg
        area_util = float(np.clip((self.ci_area + self.area_used) / max(self.max_area, 1.0), 0, 1))
        sw, hwc = self.ci_sw, cycles(self.ci_hw)
        ratio = max(sw - hwc, 0.0) / max(sw, 1.0)
        legal = np.flatnonzero(mask[:-1])
        s13 = 1.0 - 1.0 / (1.0 + float(max(0, self._best_single_saving(legal))))
        dedup = self.area_used
        if self.ci and self.sig_index.signature(self.ci) not in self.pattern_sigs:
            dedup += self.ci_area
        S = self._x["scale"]
        f = [
            min(1.0, N / self.max_nodes),
            len(ddg.nodes_pool) / N,
            area_util,
            ratio,
            min(1.0, len(ddg.input_set) / self.max_in),
            min(1.0, len(ddg.perm_input_set) / self.max_in),
            min(1.0, len(ddg.output_set) / self.max_out),
            min(1.0, len(ddg.perm_output_set) / self.max_out),
            min(1.0, len(self._frontier()) / N),
            min(1.0, int(mask[:-1].sum()) / N),
            min(1.0, self.no_progress / max(1, self.np_limit)),
            min(1.0, len(self.ci) / max(1, self._target_ci_size())),
            s13,
            float(np.clip(dedup / max(self.max_area, 1e-6), 0, 1)),
            float(np.clip(hw_c / S, 0, 1)),
            float(np.clip(cycles(hw_c) / S, 0, 1)),
        ]
        return np.asarray(f, dtype=np.float32)

    def _observation(self) -> Dict[str, np.ndarray]:
        N = self.static.num_nodes
        status = np.zeros(self.max_nodes, dtype=np.int64)
        status[:N] = [self.ddg.pattern[n].value for n in self.static.nodes]
        mask = self._mask()
        dyn, hw_c = self._structure()
        return {
            "node_status_vector": status,
            "node_dynamic_feats": dyn,
            "graph_feats": self._graph_feats(mask, hw_c),
            "action_mask": mask.copy(),
            "graph_idx": np.asarray([self.graph_idx], dtype=np.int64),
            "num_nodes": np.asarray([N], dtype=np.int64),
        }

    # ==================================================================
    # results
    # ==================================================================
    def episode_summary(self) -> Dict[str, Any]:
        """Metrics of the committed set.  Reported quantities count the CIs
        with more than one node and a positive saving (a one-node CI saves
        nothing, since every admissible operation takes one cycle)."""
        saved = self.kept_saved
        rest = self.sw_total - saved
        sp = float("inf") if rest <= 0 else self.sw_total / rest
        n_inst = len(self.kept)
        return {
            "speedup": sp,
            "area_used": self.kept_area,
            "area_budget": self.max_area,
            "area_ref": self.static.total_area,
            "area_util": self.kept_area / max(self.max_area, 1e-6),
            "num_patterns": len(self.kept_sigs),
            "num_instances": n_inst,
            "mean_ci_size": (sum(len(c) for c in self.kept) / n_inst) if n_inst else 0.0,
            "steps": self.steps,
            "violations": self.violations,
            "commits": self.stats["commits"],
            "return_commit": self.stats["r_commit"],
            "return_shape": self.stats["r_shape"],
        }

    def committed_cis(self) -> List[List[str]]:
        return [list(c) for c in self.kept]
