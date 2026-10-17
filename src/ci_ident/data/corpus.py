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


@lru_cache(maxsize=16)
def load_corpus(manifest: str) -> Corpus:
    """Process-wide cached :meth:`Corpus.load`."""
    return Corpus.load(manifest)
