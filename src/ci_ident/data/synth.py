"""Synthetic DDG generator (training corpus).

Algorithm (topological fill, in-degree at most 2): nodes ``0 .. N-1`` are a
topological order by construction.  Every node ``i >= 1`` is a source with
probability ``source_fraction``; otherwise it receives ``k`` distinct
predecessors (``k = 2`` with probability ``p_two_edges``, else 1) drawn
uniformly from ``{0 .. i-1}``.  Sources are labelled ``load`` (a block input);
an internal node gets an inadmissible operation with probability
``invalid_node_frac`` and otherwise an admissible one drawn from the class mix
arithmetic/logic/shift/comparison = 0.45/0.25/0.20/0.10.

The graph of size ``N`` and replica ``k`` is seeded from
``sha256(f"{seed}:{N}:{k}")``, so a graph does not depend on which other graphs
are generated, and generation is byte-reproducible.

The defaults are those of the corpus configuration of the TACO paper (sizes 10 to
2000 in steps of two, 200 graphs per size, the operation mix above).  Uniform
predecessor choice gives a depth that grows logarithmically with ``N``; use
:func:`ci_ident.data.stats.corpus_stats` (``ci-ident-stats``) to report the
structural statistics of the corpus you train on.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ci_ident.costs import INVALID_OPERATIONS, VALID_OPERATIONS

SOURCE_LABEL = "load"


@dataclass(frozen=True)
class SynthConfig:
    sizes: Tuple[int, ...] = tuple(range(10, 2001, 2))
    per_size: int = 200
    seed: int = 42
    source_fraction: float = 0.18
    p_two_edges: float = 0.55
    invalid_node_frac: float = 0.024
    op_classes: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "arithmetic": ("add", "sub", "mul"),
        "logic": ("and", "or", "xor"),
        "shift": ("shl", "lshr", "ashr"),
        "comparison": ("icmp",),
    })
    op_class_weights: Dict[str, float] = field(default_factory=lambda: {
        "arithmetic": 0.45, "logic": 0.25, "shift": 0.20, "comparison": 0.10,
    })
    invalid_ops: Tuple[str, ...] = ("br", "ret", "call", "phi", "store", "switch")

    def __post_init__(self):
        if not self.sizes or min(self.sizes) < 2:
            raise ValueError("sizes must be >= 2")
        for name in ("source_fraction", "p_two_edges", "invalid_node_frac"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for ops in self.op_classes.values():
            bad = [o for o in ops if o not in VALID_OPERATIONS]
            if bad:
                raise ValueError(f"inadmissible op(s) in op_classes: {bad}")
        bad = [o for o in self.invalid_ops if o not in INVALID_OPERATIONS]
        if bad:
            raise ValueError(f"invalid_ops must be inadmissible operations, got {bad}")

    def op_distribution(self) -> Tuple[List[str], np.ndarray]:
        ops, p = [], []
        for cls, members in self.op_classes.items():
            w = float(self.op_class_weights.get(cls, 0.0))
            for op in members:
                ops.append(op)
                p.append(w / len(members))
        p = np.asarray(p, dtype=np.float64)
        if p.sum() <= 0:
            raise ValueError("operation class weights sum to zero")
        return ops, p / p.sum()


def graph_seed(seed: int, n: int, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{n}:{k}".encode()).digest()[:4], "little")


def generate_graph(n: int, rng: np.random.Generator, cfg: SynthConfig) -> nx.DiGraph:
    ops, p = cfg.op_distribution()
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(1, n):
        if rng.random() < cfg.source_fraction:
            continue
        k = min(2 if rng.random() < cfg.p_two_edges else 1, i)
        for u in rng.choice(i, size=k, replace=False):
            g.add_edge(int(u), i)
    use_invalid = cfg.invalid_node_frac > 0.0 and len(cfg.invalid_ops) > 0
    ops_arr, inv_arr = np.asarray(ops), np.asarray(cfg.invalid_ops)
    for v in g.nodes():
        if g.in_degree(v) == 0:
            g.nodes[v]["label"] = SOURCE_LABEL
        elif use_invalid and rng.random() < cfg.invalid_node_frac:
            g.nodes[v]["label"] = str(rng.choice(inv_arr))
        else:
            g.nodes[v]["label"] = str(rng.choice(ops_arr, p=p))
    return g


def write_dot(g: nx.DiGraph, path: Path, name: str) -> None:
    lines = [f'digraph "{name}" {{']
    lines += [f'  "{v}" [label="{g.nodes[v]["label"]}"];' for v in sorted(g.nodes())]
    lines += [f'  "{u}" -> "{v}";' for u, v in sorted(g.edges())]
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_size(args) -> List[str]:
    n, cfg, dot_dir, overwrite = args
    out = []
    for k in range(cfg.per_size):
        p = Path(dot_dir) / f"dot_file_{n}_{k}.dot"
        if overwrite or not p.exists():
            write_dot(generate_graph(n, np.random.default_rng(graph_seed(cfg.seed, n, k)), cfg),
                      p, p.stem)
        out.append(str(p))
    return out


def generate_corpus(out_dir, cfg: SynthConfig = SynthConfig(), *, overwrite: bool = False,
                    manifest: Optional[str] = "manifest.json", jobs: int = 1,
                    verbose: bool = True) -> List[Path]:
    """Write ``out_dir/dot/dot_file_{N}_{k}.dot`` for every size and replica,
    and (unless ``manifest`` is None) a corpus manifest ``out_dir/<manifest>``.
    The manifest lists graphs by size, then replica."""
    out_dir = Path(out_dir)
    dot_dir = out_dir / "dot"
    dot_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(n, cfg, str(dot_dir), overwrite) for n in cfg.sizes]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            groups = list(ex.map(_write_size, tasks, chunksize=4))
    else:
        groups = [_write_size(t) for t in tasks]
    paths = [Path(p) for grp in groups for p in grp]
    if verbose:
        print(f"  {len(paths)} graphs in {dot_dir}", flush=True)
    if manifest:
        from ci_ident.data.build import build_manifest
        build_manifest(paths, out_dir / manifest, name=out_dir.name, jobs=jobs)
    return paths


def _parse_sizes(s: str) -> Tuple[int, ...]:
    a, b, step = (int(x) for x in s.split(":"))
    return tuple(range(a, b + 1, step))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ci-ident-synth", description="Synthetic DDG generator (training corpus).")
    ap.add_argument("out_dir", help="output directory")
    ap.add_argument("--sizes", default="10:2000:2", help="first:last:step (default 10:2000:2)")
    ap.add_argument("--per-size", type=int, default=200, help="graphs per size (default 200)")
    ap.add_argument("--seed", type=int, default=42, help="generator seed (default 42)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--overwrite", action="store_true", help="replace existing files")
    a = ap.parse_args(argv)
    cfg = SynthConfig(sizes=_parse_sizes(a.sizes), per_size=a.per_size, seed=a.seed)
    generate_corpus(a.out_dir, cfg, overwrite=a.overwrite, jobs=a.jobs)


if __name__ == "__main__":
    main()
