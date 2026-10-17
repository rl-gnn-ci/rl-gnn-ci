"""Scenario-grid evaluation (``ci-ident-bench``) and its summary
(``ci-ident-bench-summary``).

A grid file (``benchmarks/grid_240.yaml``) names a corpus, the benchmark DDGs,
the port bounds and the area factors.  Every (benchmark, port bound, area
factor) scenario is run by one method and written as one JSON line; a rerun
skips scenarios already in the output file.

Methods:

* ``rl``  -- greedy rollout of an exported PPO model (``--model``);
* ``ilp`` -- the ILP baseline (``--time-budget`` seconds per
  scenario, enumeration and selection together).

Every row records the speedup, the area used, the budget and ``A_ref``, the
number of patterns and instances, the runtime, and the identified CIs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import yaml

from ci_ident.data import Corpus

Scenario = Tuple[str, int, Tuple[int, int], float]  # name, corpus index, io, area factor


def load_grid(path) -> Tuple[Corpus, List[Scenario]]:
    path = Path(path).expanduser().resolve()
    spec = yaml.safe_load(path.read_text())
    unknown = set(spec) - {"corpus", "benchmarks", "io_pairs", "area_factors"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    corpus = Corpus.load(path.parent / spec["corpus"])
    scen = []
    for name, dot in spec["benchmarks"].items():
        gi = corpus.index_of(path.parent / dot)
        for io in spec["io_pairs"]:
            for af in spec["area_factors"]:
                scen.append((str(name), gi, (int(io[0]), int(io[1])), float(af)))
    return corpus, scen


def _key(row) -> Tuple:
    return (row["benchmark"], tuple(row["io"]), float(row["area_factor"]))


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def run_rl(model_path: str, corpus: Corpus, todo: Sequence[Scenario], *, device: str = "cpu",
           radius: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    from ci_ident.env import CIEnv
    from ci_ident.rl.infer import greedy_episode, load_model

    net, payload = load_model(model_path, corpus, device=device)
    env_cfg = dict(payload["env"])
    if radius is not None:
        env_cfg["radius"] = int(radius)
    for name, gi, io, af in todo:
        env = CIEnv({**env_cfg, "io": list(io), "area_factor": af,
                     "corpus": str(corpus.manifest_path), "graphs": [gi]})
        r = greedy_episode(net, env, graph_idx=gi)
        yield {"benchmark": name, "io": list(io), "area_factor": af, "method": "rl",
               "speedup": r["speedup"], "area_used": r["area_used"],
               "area_budget": r["area_budget"], "a_ref": r["area_ref"],
               "num_patterns": r["num_patterns"], "num_instances": r["num_instances"],
               "mean_ci_size": r["mean_ci_size"], "runtime_s": r["runtime_s"],
               "status": "ok", "radius": int(env_cfg.get("radius", 1)),
               "model": str(model_path), "instances": r["cis"]}


def run_ilp(corpus: Corpus, todo: Sequence[Scenario], *, time_budget_s: float,
            solver: str = "auto") -> Iterator[Dict[str, Any]]:
    from ci_ident.baselines.ilp import identify_ilp
    from ci_ident.ddg.graph import load_ddg

    for name, gi, io, af in todo:
        g = load_ddg(corpus.dot_path(gi))
        r = identify_ilp(g, io[0], io[1], af, time_budget_s=time_budget_s, solver=solver)
        yield {"benchmark": name, "io": list(io), "area_factor": af, "method": "ilp",
               "speedup": r["speedup"], "area_used": r["area_used"],
               "area_budget": r["area_budget"], "a_ref": r["a_ref"],
               "num_patterns": r["num_patterns"], "num_instances": r["num_instances"],
               "mean_ci_size": r["mean_ci_size"], "runtime_s": r["runtime_s"],
               "status": r["status"], "solver": r["solver"],
               "time_budget_s": float(time_budget_s), "instances": r["instances"]}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ci-ident-bench", description="Scenario-grid evaluation.")
    ap.add_argument("method", choices=("rl", "ilp"), help="method to evaluate")
    ap.add_argument("out", help="JSONL results (appended; finished scenarios are skipped)")
    ap.add_argument("--grid", default="benchmarks/grid_240.yaml", help="scenario grid (YAML)")
    ap.add_argument("--model", help="exported model (.pt); rl method")
    ap.add_argument("--radius", type=int, help="rl: growth radius (default: the model's)")
    ap.add_argument("--device", default="cpu", help="torch device (default cpu)")
    ap.add_argument("--time-budget", type=float, default=3600.0, help="ilp: seconds per scenario")
    ap.add_argument("--solver", default="auto", help="ilp: auto | scip | cplex")
    ap.add_argument("--only", help="comma-separated benchmark names")
    a = ap.parse_args(argv)

    corpus, scen = load_grid(a.grid)
    if a.only:
        keep = set(a.only.split(","))
        scen = [s for s in scen if s[0] in keep]
    out = Path(a.out)
    done = set()
    if out.exists():
        old = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        ident = {(r["method"], r.get("model")) for r in old}
        here = (a.method, str(a.model) if a.method == "rl" else None)
        if ident - {here}:
            ap.error(f"{out} holds results of {sorted(map(str, ident))}; "
                     f"use one output file per method and model")
        done = {_key(r) for r in old}
    todo = [s for s in scen if (s[0], s[2], s[3]) not in done]
    print(f"{len(scen)} scenarios, {len(todo)} to run", flush=True)
    if a.method == "rl":
        if not a.model:
            ap.error("--model is required for the rl method")
        rows = run_rl(a.model, corpus, todo, device=a.device, radius=a.radius)
    else:
        rows = run_ilp(corpus, todo, time_budget_s=a.time_budget, solver=a.solver)
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for i, row in enumerate(rows, 1):
            f.write(json.dumps({k: _clean(v) for k, v in row.items()}) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)} {time.time() - t0:.0f}s] {row['benchmark']} io={row['io']} "
                  f"af={row['area_factor']} speedup={row['speedup']:.4f} ({row['status']})", flush=True)


def summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Geometric-mean speedup per method (per model for RL), overall, per port
    bound and per benchmark."""
    out: Dict[str, Any] = {}
    label = lambda r: r["method"] if r["method"] != "rl" else f"rl:{r.get('model')}"
    for m in sorted({label(r) for r in rows}):
        rs = [r for r in rows if label(r) == m and r["speedup"] is not None]
        gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None
        out[m] = {"scenarios": len(rs), "geomean_speedup": gm([r["speedup"] for r in rs]),
                  "per_io": {f"{io[0]},{io[1]}": gm([r["speedup"] for r in rs if r["io"] == list(io)])
                             for io in sorted({tuple(r["io"]) for r in rs})},
                  "per_benchmark": {b: gm([r["speedup"] for r in rs if r["benchmark"] == b])
                                    for b in sorted({r["benchmark"] for r in rs})},
                  "statuses": {s: sum(r["status"] == s for r in rs)
                               for s in sorted({r["status"] for r in rs})}}
    return out


def summary_main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ci-ident-bench-summary",
                                 description="Geometric-mean speedups of ci-ident-bench results.")
    ap.add_argument("results", nargs="+", help="JSONL files written by ci-ident-bench")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for p in a.results for l in Path(p).read_text().splitlines() if l.strip()]
    print(json.dumps(summary(rows), indent=1))


if __name__ == "__main__":
    main()
