"""Structural statistics of DDGs and corpora (``ci-ident-stats``)."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Sequence

import networkx as nx
import numpy as np

from ci_ident.data.corpus import Corpus
from ci_ident.ddg.graph import load_ddg

FIELDS = ("num_nodes", "num_edges", "depth", "max_in_degree", "max_out_degree",
          "q90_out_degree", "frac_sources", "frac_admissible", "edge_density")


def graph_stats(dot_path) -> Dict[str, float]:
    g = load_ddg(dot_path)
    n = g.number_of_nodes()
    d = nx.DiGraph(g)  # depth on the simple graph
    indeg = np.asarray([g.in_degree(v) for v in g.nodes()])
    outdeg = np.asarray([g.out_degree(v) for v in g.nodes()])
    return {
        "num_nodes": n,
        "num_edges": g.number_of_edges(),
        "depth": nx.dag_longest_path_length(d),
        "max_in_degree": int(indeg.max()),
        "max_out_degree": int(outdeg.max()),
        "q90_out_degree": float(np.quantile(outdeg, 0.9)),
        "frac_sources": float((indeg == 0).mean()),
        "frac_admissible": float(np.mean([g.nodes[v]["ci_eligible"] for v in g.nodes()])),
        "edge_density": g.number_of_edges() / n,
    }


def corpus_stats(corpus: Corpus, *, jobs: int = 1, sample: Optional[int] = None,
                 seed: int = 0) -> Dict[str, Dict[str, float]]:
    idx = np.arange(len(corpus))
    if sample is not None and sample < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, sample, replace=False))
    paths = [str(corpus.dot_path(i)) for i in idx]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            rows = list(ex.map(graph_stats, paths, chunksize=16))
    else:
        rows = [graph_stats(p) for p in paths]
    out = {"graphs": {"count": len(rows)}}
    for f in FIELDS:
        v = np.asarray([r[f] for r in rows], dtype=np.float64)
        out[f] = {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
                  "max": float(v.max())}
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ci-ident-stats", description="Structural statistics of DDGs and corpora.")
    ap.add_argument("manifest", help="corpus manifest (JSON)")
    ap.add_argument("--sample", type=int, help="statistics of a random sample of graphs")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes (default 1)")
    a = ap.parse_args(argv)
    print(json.dumps(corpus_stats(Corpus.load(a.manifest), jobs=a.jobs, sample=a.sample), indent=1))


if __name__ == "__main__":
    main()
