"""Greedy inference (``ci-ident-eval``).

A trained network is applied in the environment's inference mode: at every
step the action with the highest score (PPO logit or DQN Q-value) among the
admissible actions is taken, until the episode ends.  The environment, its
variant and its reward weights are those the model was trained with; only the
graph, the port bound and the area factor change.

Model files (``.pt``) hold ``{"format": "ci_ident.taco_model/1", "variant":
name, "env": env dict, "model": ModelConfig dict, "provenance": dict,
"state_dict": ...}``; see
:func:`save_model`.  Parameters are loaded strictly.
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
from ci_ident.model import GraphStore, ModelConfig, TacoNetwork

FORMAT = "ci_ident.taco_model/1"


def save_model(path, net: TacoNetwork, env: dict, variant: str = "", **provenance) -> Path:
    payload = {"format": FORMAT, "variant": variant, "env": dict(env),
               "model": net.cfg.to_dict(), "provenance": provenance,
               "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}}
    torch.save(payload, path)
    return Path(path)


def load_model(path, corpus=None, *, device: str = "cpu") -> Tuple[TacoNetwork, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    net = TacoNetwork(ModelConfig.from_dict(payload["model"]),
                      GraphStore(corpus) if corpus is not None else None)
    net.load_state_dict(payload["state_dict"], strict=True)
    return net.eval().to(device), payload


@torch.no_grad()
def greedy_episode(net: TacoNetwork, env: CIEnv, *, graph_idx: Optional[int] = None
                   ) -> Dict[str, Any]:
    dev = net.stop_token.device
    obs, _ = env.reset(options=None if graph_idx is None else {"graph_idx": int(graph_idx)})
    t0 = time.perf_counter()
    info: Dict[str, Any] = {}
    while True:
        scores, _ = net({k: torch.as_tensor(np.asarray(v)[None]).to(dev) for k, v in obs.items()})
        a = int(torch.argmax(scores[0]).item())
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    out = env.episode_summary()
    out.update(end_reasons=info.get("end_reasons", []), runtime_s=time.perf_counter() - t0,
               cis=env.committed_cis())
    return out


def evaluate(model_path, corpus, scenarios: Iterable[Tuple[int, Tuple[int, int], float]], *,
             device: str = "cpu") -> List[Dict[str, Any]]:
    """``scenarios``: ``(graph_idx, (max_in, max_out), area_factor)`` triples."""
    net, payload = load_model(model_path, corpus, device=device)
    env_cfg = {**payload["env"], "mode": "inference"}
    rows = []
    for gi, io, af in scenarios:
        env = CIEnv({**env_cfg, "io": list(io), "area_factor": float(af),
                     "corpus": str(net.store.corpus.manifest_path), "graphs": [int(gi)]})
        r = greedy_episode(net, env, graph_idx=int(gi))
        r.update(graph_idx=int(gi), graph=str(net.store.corpus.dot_path(gi)), io=list(io),
                 area_factor=float(af), variant=payload.get("variant", ""))
        rows.append(r)
    return rows


def _pairs(s: str) -> List[Tuple[int, int]]:
    return [tuple(int(x) for x in p.split(",")) for p in s.split(";")]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ci-ident-eval", description="Greedy inference with a saved network.")
    ap.add_argument("model", help="model file (.pt)")
    ap.add_argument("corpus", help="corpus manifest of the graphs to evaluate")
    ap.add_argument("--graphs", help="comma-separated corpus indices (default: all)")
    ap.add_argument("--io", default="4,2", help="port bounds, e.g. '2,1;4,2;6,3'")
    ap.add_argument("--area-factors", default="0.4", help="e.g. '0.2,0.4,1.0'")
    ap.add_argument("--device", default="cpu", help="torch device (default cpu)")
    ap.add_argument("--out", help="JSONL output (default: stdout)")
    a = ap.parse_args(argv)
    from ci_ident.data import Corpus
    n = len(Corpus.load(a.corpus))
    graphs = range(n) if not a.graphs else [int(x) for x in a.graphs.split(",")]
    scen = [(g, io, af) for g in graphs for io in _pairs(a.io)
            for af in (float(x) for x in a.area_factors.split(","))]
    rows = evaluate(a.model, a.corpus, scen, device=a.device)
    lines = [json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                         for k, v in r.items()}) for r in rows]
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
