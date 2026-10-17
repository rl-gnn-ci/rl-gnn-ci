"""Static graph tensors for the GNN, addressed by corpus index.

The environment observes ``graph_idx`` (an index into a corpus manifest) and
the model reads the DDG's adjacency and static node features from this store.
Both are built from the same DOT file through :func:`ci_ident.ddg.load_ddg`,
and the store checks the node-order digest recorded in the manifest and the
node count, so node ``i`` of the GNN input is node ``i`` of the action space
by construction.

Node ``i`` of a DGL graph is node ``i`` of the canonical order
(:func:`ci_ident.ddg.graph.node_order`).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

import dgl
import torch

from ci_ident.data.corpus import Corpus, load_corpus
from ci_ident.ddg.graph import edge_index, load_ddg, node_order, node_order_digest, static_node_features


def build_dgl_graph(dot_path, *, expected_digest: Optional[str] = None,
                    expected_nodes: Optional[int] = None) -> dgl.DGLGraph:
    """DGL graph of one DDG in canonical node order, with ``ndata['feat']`` =
    ``[hw_latency, area]`` (``[0, 0]`` for inadmissible nodes)."""
    g = load_ddg(dot_path)
    order = node_order(g)
    if expected_nodes is not None and len(order) != int(expected_nodes):
        raise ValueError(f"{dot_path}: {len(order)} nodes, manifest says {expected_nodes}")
    if expected_digest is not None and node_order_digest(order) != expected_digest:
        raise ValueError(f"{dot_path}: node order differs from the corpus manifest")
    src, dst = edge_index(g)
    dg = dgl.graph((torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)),
                   num_nodes=len(order), idtype=torch.int64)
    dg.ndata["feat"] = torch.tensor(static_node_features(g), dtype=torch.float32).reshape(-1, 2)
    return dg


class GraphStore:
    """LRU cache of DGL graphs of one corpus."""

    def __init__(self, corpus: "Corpus | str", cache_size: int = 1024):
        self.corpus = load_corpus(str(corpus)) if not isinstance(corpus, Corpus) else corpus
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[int, dgl.DGLGraph]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.corpus)

    def num_nodes(self, idx: int) -> int:
        return self.corpus.entries[int(idx)].num_nodes

    def graph(self, idx: int) -> dgl.DGLGraph:
        idx = int(idx)
        g = self._cache.get(idx)
        if g is not None:
            self._cache.move_to_end(idx)
            return g
        if not 0 <= idx < len(self.corpus):
            raise IndexError(f"graph index {idx} outside corpus {self.corpus.name!r} "
                             f"of {len(self.corpus)} graphs")
        e = self.corpus.entries[idx]
        g = build_dgl_graph(e.path, expected_digest=e.node_order, expected_nodes=e.num_nodes)
        self._cache[idx] = g
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return g

    def batch(self, indices: Sequence[int]) -> dgl.DGLGraph:
        return dgl.batch([self.graph(i) for i in indices])
