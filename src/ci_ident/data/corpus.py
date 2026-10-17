"""Graph corpora: a manifest that maps a graph index to a DOT file.

A corpus is a JSON manifest::

    {
      "name": "synth-train",
      "root": ".",                       # relative to the manifest's directory
      "graphs": [
        {"path": "dot/dot_file_10_0.dot", "num_nodes": 10, "num_edges": 9,
         "node_order": "3f2a..."},       # digest of the DOT node order
        ...
      ]
    }

The index of a graph is its position in ``graphs``.  The environment and the
model's graph store (:mod:`ci_ident.model.graph_store`) both address graphs
through the same manifest, and the store verifies the node-order digest of
every graph it serves, so node ``i`` of the environment is node ``i`` of the
GNN input by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class GraphEntry:
    path: Path
    num_nodes: int
    num_edges: int
    node_order: str


class Corpus:
    def __init__(self, name: str, entries: Sequence[GraphEntry], manifest_path: Optional[Path] = None):
        if not entries:
            raise ValueError(f"corpus {name!r} is empty")
        self.name = name
        self.entries: Tuple[GraphEntry, ...] = tuple(entries)
        self.manifest_path = manifest_path
        self._by_path: Dict[str, int] = {str(e.path): i for i, e in enumerate(self.entries)}

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, manifest: PathLike) -> "Corpus":
        manifest = Path(manifest).expanduser().resolve()
        data = json.loads(manifest.read_text())
        root = (manifest.parent / data.get("root", ".")).resolve()
        entries = [
            GraphEntry(path=(root / g["path"]).resolve(), num_nodes=int(g["num_nodes"]),
                       num_edges=int(g["num_edges"]), node_order=str(g["node_order"]))
            for g in data["graphs"]
        ]
        return cls(data.get("name", manifest.stem), entries, manifest)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def dot_path(self, index: int) -> Path:
        return self.entries[int(index)].path

    def index_of(self, dot_path: PathLike) -> int:
        key = str(Path(dot_path).expanduser().resolve())
        if key not in self._by_path:
            raise KeyError(f"{dot_path} is not in corpus {self.name!r}")
        return self._by_path[key]

    @property
    def sizes(self) -> np.ndarray:
        return np.asarray([e.num_nodes for e in self.entries], dtype=np.int64)

    # ------------------------------------------------------------------
    def sampler(self, *, size_min: int = 1, size_max: int = 10**9, exponent: float = 1.0,
                mode: str = "per_size") -> "SizeSampler":
        return SizeSampler(self, size_min=size_min, size_max=size_max,
                           exponent=exponent, mode=mode)


class SizeSampler:
    """Draws training graphs log-uniformly in their node count.

    ``mode="per_size"``: draw a node count ``n`` from the distinct sizes in
    ``[size_min, size_max]`` with probability proportional to ``n**-exponent``,
    then a graph uniformly among the graphs of that size.  On a corpus with the
    same number of graphs at every size (the synthetic grid) this is the
    log-uniform size distribution.

    ``mode="per_graph"``: draw a graph with probability proportional to
    ``num_nodes**-exponent`` (sizes weighted by how often they occur).
    """

    def __init__(self, corpus: Corpus, *, size_min: int, size_max: int,
                 exponent: float, mode: str):
        if mode not in ("per_size", "per_graph"):
            raise ValueError(f"unknown sampling mode {mode!r}")
        if not 0.0 <= float(exponent) <= 4.0:
            raise ValueError("exponent must be in [0, 4]")
        sizes = corpus.sizes
        cand = np.flatnonzero((sizes >= size_min) & (sizes <= size_max))
        if cand.size == 0:
            raise ValueError(f"no graph in corpus {corpus.name!r} with size in "
                             f"[{size_min}, {size_max}]")
        self.mode = mode
        self.exponent = float(exponent)
        if mode == "per_size":
            uniq = np.unique(sizes[cand])
            w = uniq.astype(np.float64) ** (-self.exponent)
            self._sizes = uniq
            self._p = w / w.sum()
            self._groups = {int(n): cand[sizes[cand] == n] for n in uniq}
        else:
            w = sizes[cand].astype(np.float64) ** (-self.exponent)
            self._cand = cand
            self._p = w / w.sum()

    def __call__(self, rng: np.random.Generator) -> int:
        if self.mode == "per_size":
            n = int(rng.choice(self._sizes, p=self._p))
            grp = self._groups[n]
            return int(grp[int(rng.integers(0, grp.size))])
        return int(self._cand[int(rng.choice(self._cand.size, p=self._p))])


@lru_cache(maxsize=16)
def load_corpus(manifest: str) -> Corpus:
    """Process-wide cached :meth:`Corpus.load`."""
    return Corpus.load(manifest)
