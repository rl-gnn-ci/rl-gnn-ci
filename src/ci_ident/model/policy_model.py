"""The network applied to batches of environment observations.

:class:`PolicyNetwork` has no dependency on Ray; training wraps it in an RLlib
model (:mod:`ci_ident.rl.model`) and evaluation calls it directly
(:mod:`ci_ident.rl.infer`)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ci_ident.model.backbone import GINBackbone, ModelConfig
from ci_ident.model.graph_store import GraphStore


class PolicyNetwork(nn.Module):
    """Backbone plus graph store: maps a batch of environment observations to
    masked logits and values."""

    def __init__(self, cfg: ModelConfig, store: Optional[GraphStore], *,
                 max_nodes_per_chunk: int = 200_000):
        super().__init__()
        self.backbone = GINBackbone(cfg)
        self.store = store
        self.max_nodes_per_chunk = int(max_nodes_per_chunk)

    def _chunks(self, sizes: Sequence[int]) -> List[slice]:
        out, start, acc = [], 0, 0
        for i, n in enumerate(sizes):
            if i > start and acc + n > self.max_nodes_per_chunk:
                out.append(slice(start, i))
                start, acc = i, 0
            acc += n
        out.append(slice(start, len(sizes)))
        return out

    def forward(self, obs: Dict[str, torch.Tensor]):
        if self.store is None:
            raise RuntimeError("PolicyNetwork has no graph store")
        dev = next(self.parameters()).device
        status = obs["node_status_vector"]
        B, N = status.shape[0], status.shape[1]
        idx = obs["graph_idx"].reshape(B).round().long().cpu().numpy()
        reported = obs["num_nodes"].reshape(B).round().long().cpu().numpy()
        sizes = np.asarray([self.store.num_nodes(i) for i in idx], dtype=np.int64)
        # Identity check per row: the observation's |V| must be the stored
        # graph's.  Rows with |V| = 0 are RLlib's all-zero dummy batch.
        bad = (reported != 0) & (reported != sizes)
        if bad.any():
            r = int(np.flatnonzero(bad)[0])
            raise RuntimeError(f"observation row {r}: graph {int(idx[r])} has "
                               f"{int(sizes[r])} nodes in the store, {int(reported[r])} in the env")
        if sizes.max(initial=0) > N:
            raise RuntimeError("graph larger than the observation's max_nodes")

        status = status.to(dev)
        node_dyn = obs["node_dynamic_feats"].to(dev, torch.float32)
        graph_feats = obs["graph_feats"].to(dev, torch.float32)
        action_mask = obs["action_mask"].to(dev)
        if graph_feats.numel() and (float(graph_feats.min()) < 0.0 or float(graph_feats.max()) > 1.0):
            raise ValueError("graph features must lie in [0, 1]")
        node_mask = torch.arange(N, device=dev).unsqueeze(0) < torch.as_tensor(sizes, device=dev).unsqueeze(1)

        logits, values = [], []
        for sl in self._chunks(sizes.tolist()):
            g = self.store.batch(idx[sl].tolist()).to(dev)
            lg, v = self.backbone(g, status[sl], node_dyn[sl], graph_feats[sl],
                                  node_mask[sl], action_mask[sl])
            logits.append(lg)
            values.append(v)
        return torch.cat(logits, 0), torch.cat(values, 0)
