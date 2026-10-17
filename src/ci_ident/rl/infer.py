"""Greedy evaluation of an exported model (``ci-ident-eval``); no Ray needed.

At every step the policy takes its most probable admissible action.  The
environment is the training environment (same mask, same episode limits); only
the graph, the port bound and the area budget change.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ci_ident.env import CIEnv
from ci_ident.model import GraphStore, ModelConfig, PolicyNetwork

FORMAT = "ci_ident.model/1"


def load_model(path, corpus=None, *, device: str = "cpu") -> Tuple[PolicyNetwork, dict]:
    """Load an exported model (strictly: every parameter must match).  The
    graph store is attached for ``corpus`` if given."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    net = PolicyNetwork(ModelConfig.from_dict(payload["model"]),
                        GraphStore(corpus) if corpus is not None else None)
    net.backbone.load_state_dict(payload["state_dict"], strict=True)
    net.eval().to(device)
    return net, payload


def _to_batch(obs: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(v)[None]).to(device) for k, v in obs.items()}


@torch.no_grad()
def greedy_episode(net: PolicyNetwork, env: CIEnv, *, graph_idx: Optional[int] = None
                   ) -> Dict[str, Any]:
    device = next(net.parameters()).device
    opts = None if graph_idx is None else {"graph_idx": int(graph_idx)}
    obs, _ = env.reset(options=opts)
    t0 = time.perf_counter()
    info: Dict[str, Any] = {}
    while True:
        logits, _ = net(_to_batch(obs, device))
        action = int(torch.argmax(logits[0]).item())
        if not obs["action_mask"][action]:
            raise RuntimeError("greedy action outside the mask")
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    summary = env.episode_summary()
    summary["end_reasons"] = info.get("end_reasons", [])
    summary["runtime_s"] = time.perf_counter() - t0
    summary["cis"] = env.committed_cis()
    return summary


def evaluate(model_path, corpus, scenarios: Iterable[Tuple[int, Tuple[int, int], float]], *,
             radius: Optional[int] = None, device: str = "cpu") -> List[Dict[str, Any]]:
    """``scenarios``: ``(graph_idx, (max_in, max_out), area_factor)`` triples.
    The environment uses the model's training configuration except for the
    port bound and the area budget (and ``radius`` if given)."""
    net, payload = load_model(model_path, corpus, device=device)
    env_cfg = dict(payload["env"])
    if radius is not None:
        env_cfg["radius"] = int(radius)
    rows = []
    store = net.store
    for gi, io, af in scenarios:
        env = CIEnv({**env_cfg, "io": list(io), "area_factor": float(af),
                     "corpus": str(store.corpus.manifest_path), "graphs": [int(gi)]})
        r = greedy_episode(net, env, graph_idx=int(gi))
        r.update(graph_idx=int(gi), graph=str(store.corpus.dot_path(gi)), io=list(io),
                 area_factor=float(af), radius=int(env_cfg.get("radius", 1)))
        rows.append(r)
    return rows


def _pairs(s: str) -> List[Tuple[int, int]]:
    return [tuple(int(x) for x in p.split(",")) for p in s.split(";")]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ci-ident-eval", description="Greedy evaluation of an exported model.")
    ap.add_argument("model", help="exported model (.pt)")
    ap.add_argument("corpus", help="corpus manifest of the graphs to evaluate")
    ap.add_argument("--graphs", help="comma-separated corpus indices (default: all)")
    ap.add_argument("--io", default="4,2", help="port bounds, e.g. '2,1;4,2;6,3'")
    ap.add_argument("--area-factors", default="0.4", help="e.g. '0.2,0.4,1.0'")
    ap.add_argument("--radius", type=int, help="growth radius (default: the model's)")
    ap.add_argument("--device", default="cpu", help="torch device (default cpu)")
    ap.add_argument("--out", help="JSONL output (default: stdout)")
    a = ap.parse_args(argv)
    from ci_ident.data import Corpus
    n = len(Corpus.load(a.corpus))
    graphs = range(n) if not a.graphs else [int(x) for x in a.graphs.split(",")]
    scen = [(g, io, af) for g in graphs for io in _pairs(a.io)
            for af in (float(x) for x in a.area_factors.split(","))]
    rows = evaluate(a.model, a.corpus, scen, radius=a.radius, device=a.device)
    lines = [json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                         for k, v in r.items()}) for r in rows]
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
