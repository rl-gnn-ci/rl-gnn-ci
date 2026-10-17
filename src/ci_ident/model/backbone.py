"""GIN policy/value network.

Architecture:

* node input: an embedding of the node status (11 codes) and a projection of
  the node scalars ``[hw_latency, area]`` (static) and
  ``[depth_C, latency, height_C] / S`` (candidate-conditioned), concatenated;
* encoder: ``num_layers`` GIN layers (learnable epsilon; MLP
  ``Linear-LayerNorm-ReLU-Linear``), each followed by LayerNorm, a residual to
  the encoder input and LeakyReLU(0.2);
* graph embedding: mean over the graph's nodes, ``Linear-LeakyReLU-LayerNorm``,
  concatenated with a projection of the 16 graph features;
* policy: node logit ``q_head([h_v || g])``; STOP logit
  ``q_head(stop_proj(g))``; invalid actions are filled with ``-1e9``;
* value: ``value_out(g)``.

Every normalisation is per sample (LayerNorm), so ``pi(a|s)`` is a function of
``s`` alone, identical in RLlib's sampling (``eval``) and learning (``train``)
modes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GINConv

from ci_ident.ddg.status import NUM_STATUSES

MASK_FILL = -1e9


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 96
    out_feats: int = 192
    embedding_dim: int = 8
    num_layers: int = 4
    graph_feat_dim: int = 16
    node_static_dim: int = 2
    node_dynamic_dim: int = 3
    num_statuses: int = NUM_STATUSES

    def __post_init__(self):
        if self.hidden_size % 2:
            raise ValueError("hidden_size must be even")
        for k, v in asdict(self).items():
            if int(v) < 1:
                raise ValueError(f"{k} must be >= 1")

    @classmethod
    def from_dict(cls, d) -> "ModelConfig":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        if set(d) - known:
            raise ValueError(f"unknown model keys: {sorted(set(d) - known)}")
        return cls(**d)


class _Projection(nn.Module):
    """``[W_e e || W_f f]`` with each half of width ``hidden/2``."""

    def __init__(self, feat_dim: int, emb_dim: int, hidden: int, feat_name: str, emb_name: str):
        super().__init__()
        self._feat_name, self._emb_name = feat_name, emb_name
        setattr(self, feat_name, nn.Linear(feat_dim, hidden // 2))
        setattr(self, emb_name, nn.Linear(emb_dim, hidden // 2))

    def forward(self, emb: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        return torch.cat([getattr(self, self._emb_name)(emb),
                          getattr(self, self._feat_name)(feats)], dim=-1)


def _xavier(layer: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _small(layer: nn.Linear, scale: float = 1e-2) -> nn.Linear:
    with torch.no_grad():
        nn.init.uniform_(layer.weight, -scale, scale)
        layer.bias.zero_()
    return layer


class GINBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        H, F = cfg.hidden_size, cfg.out_feats
        self.activation = nn.LeakyReLU(0.2)

        self.embedding = nn.Sequential(
            nn.Embedding(cfg.num_statuses, cfg.embedding_dim),
            _xavier(nn.Linear(cfg.embedding_dim, H)),
            self.activation,
        )
        self.node_feat_integration = _Projection(
            cfg.node_static_dim + cfg.node_dynamic_dim, H, H,
            "node_feat_transform", "node_emb_transform")

        self.gnn_layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            mlp = nn.Sequential(_xavier(nn.Linear(H, H)), nn.LayerNorm(H), nn.ReLU(),
                                _xavier(nn.Linear(H, H)))
            self.gnn_layers.append(GINConv(apply_func=mlp, aggregator_type="sum", learn_eps=True))
        self.gnn_norm_layers = nn.ModuleList([nn.LayerNorm(H) for _ in range(cfg.num_layers)])

        self.state_ffn = nn.Sequential(_xavier(nn.Linear(H, H)), self.activation, nn.LayerNorm(H))
        self.graph_feature_integrator = _Projection(
            cfg.graph_feat_dim, H, H, "graph_feat_transform", "graph_emb_transform")

        self.stop_proj = _xavier(nn.Linear(H, 2 * H))
        self.q_head = nn.Sequential(_xavier(nn.Linear(2 * H, F)), self.activation,
                                    nn.LayerNorm(F), _small(nn.Linear(F, 1)))
        self.value_out = nn.Sequential(_xavier(nn.Linear(H, H)), self.activation,
                                       nn.Identity(), _small(nn.Linear(H, 1)))

    # ------------------------------------------------------------------
    def encode(self, g: dgl.DGLGraph, status: torch.Tensor, node_dynamic: torch.Tensor,
               graph_feats: torch.Tensor, node_mask: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Node embeddings ``[B, N, H]`` (zero at padding) and graph embeddings ``[B, H]``.

        ``g``: batch of the B graphs; ``status``: ``[B, N]`` integer codes;
        ``node_dynamic``: ``[B, N, node_dynamic_dim]``; ``graph_feats``:
        ``[B, graph_feat_dim]`` in [0, 1]; ``node_mask``: ``[B, N]`` bool, true
        on the first ``|V_b|`` positions of row b.
        """
        B, N = node_mask.shape
        n_valid = int(node_mask.sum())
        if g.num_nodes() != n_valid:
            raise RuntimeError(f"graph batch has {g.num_nodes()} nodes, node mask {n_valid}")
        codes = status[node_mask].long()
        if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= self.cfg.num_statuses):
            raise ValueError("node status code outside the status vocabulary")
        feats = torch.cat([g.ndata["feat"].to(node_dynamic.dtype), node_dynamic[node_mask]], dim=-1)

        x = self.node_feat_integration(self.embedding(codes), feats)
        x0 = x
        for conv, norm in zip(self.gnn_layers, self.gnn_norm_layers):
            x = self.activation(norm(conv(g, x)) + x0)

        ne = x.new_zeros(B, N, x.size(-1))
        ne[node_mask] = x
        lengths = node_mask.sum(dim=1).clamp(min=1).to(ne.dtype).unsqueeze(-1)
        graph_emb = self.state_ffn(ne.sum(dim=1) / lengths)
        graph_emb = self.graph_feature_integrator(graph_emb, graph_feats)
        return ne, graph_emb

    def heads(self, ne: torch.Tensor, graph_emb: torch.Tensor, node_mask: torch.Tensor,
              action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masked logits ``[B, N + 1]`` (STOP last) and values ``[B]``."""
        B, N, _ = ne.shape
        ge = graph_emb.unsqueeze(1).expand(-1, N, -1)[node_mask]
        node_logits = ne.new_full((B, N), MASK_FILL)
        node_logits[node_mask] = self.q_head(torch.cat([ne[node_mask], ge], dim=-1)).view(-1)
        stop_logit = self.q_head(self.stop_proj(graph_emb))
        logits = torch.cat([node_logits, stop_logit], dim=1)

        mask = action_mask > 0.5 if action_mask.dtype != torch.bool else action_mask
        empty = ~mask.any(dim=1)
        if bool(empty.any()):            # only RLlib's all-zero dummy rows
            mask = mask.clone()
            mask[empty, -1] = True
        logits = logits.masked_fill(~mask, MASK_FILL)
        value = self.value_out(graph_emb).squeeze(-1)
        return logits, value

    def forward(self, g, status, node_dynamic, graph_feats, node_mask, action_mask):
        ne, ge = self.encode(g, status, node_dynamic, graph_feats, node_mask)
        return self.heads(ne, ge, node_mask, action_mask)
