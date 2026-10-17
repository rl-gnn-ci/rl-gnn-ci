"""GNN encoder and the PPO / DQN heads of the TACO formulation.

Encoder (paper, "Graph Encoder"): node input ``x_v = [x_stat || x_dyn]`` with
``x_stat`` = opcode embedding, hardware latency and area, and ``x_dyn`` = an
embedding of the node-status code; ``L`` message-passing layers of one family
(GIN, GATv2 or GCN; GCN and GATv2 with self-loops) along the DDG edges
(messages follow data flow), each
followed by LayerNorm, a residual to the encoder input and ReLU; a graph
embedding ``g`` by mean, sum or attention pooling.  The global features
``f_t`` are projected and concatenated: ``g' = [g || ReLU(W_f f_t)]``.

Heads (paper, "RL Parameterizations"), on action features
``u(v) = [z_v || g']`` and ``u(STOP) = [e_STOP || g']`` with a learned vector
``e_STOP`` in the node slot, so one MLP scores node actions and STOP alike:

* PPO: logits ``l(A) = MLP_pi(u(A)) / tau``, value ``V = MLP_V(g')``;
* DQN (dueling): ``Q(A) = V(g') + A(u(A)) - mean_{A' feasible} A(u(A'))``,
  optionally with factorised-Gaussian noisy layers (NoisyNet) in both streams.

Inadmissible actions get ``-1e9``.  All normalisation is per sample
(LayerNorm), so the output for a state does not depend on the other states in
a batch or on train/eval mode (except NoisyNet noise, which is off in eval mode).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv, GINConv, GraphConv

from ci_ident.costs import OPCODES
from ci_ident.ddg.status import NUM_STATUSES
from ci_ident.model.graph_store import GraphStore

MASK_FILL = -1e9
CONVS = ("gin", "gatv2", "gcn")
POOLS = ("mean", "sum", "attention")
ALGOS = ("ppo", "dqn")


@dataclass(frozen=True)
class ModelConfig:
    algo: str = "dqn"
    conv: str = "gin"
    num_layers: int = 4
    hidden: int = 96
    heads: int = 1                   # GATv2 attention heads
    pooling: str = "mean"
    out_feats: int = 0               # head width; 0 = 2 * hidden
    status_dim: int = 8
    op_dim: int = 8
    graph_feat_dim: int = 12
    temperature: float = 1.0         # PPO policy temperature tau
    noisy: bool = False              # DQN: NoisyNet layers in the dueling streams
    sigma0: float = 0.5

    def __post_init__(self):
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}")
        if self.conv not in CONVS:
            raise ValueError(f"conv must be one of {CONVS}")
        if self.pooling not in POOLS:
            raise ValueError(f"pooling must be one of {POOLS}")
        if self.num_layers < 1 or self.hidden < 2 or self.heads < 1:
            raise ValueError("num_layers, hidden and heads must be positive")
        if self.conv == "gatv2" and self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.noisy and self.algo != "dqn":
            raise ValueError("noisy layers are a DQN option")

    @property
    def head_width(self) -> int:
        return self.out_feats or 2 * self.hidden

    @classmethod
    def from_dict(cls, d) -> "ModelConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown model keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy linear layer (Fortunato et al., 2018).
    Noise is drawn at every forward pass in training mode and is zero in
    evaluation mode."""

    def __init__(self, n_in: int, n_out: int, sigma0: float = 0.5):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        bound = 1.0 / math.sqrt(n_in)
        self.weight_mu = nn.Parameter(torch.empty(n_out, n_in).uniform_(-bound, bound))
        self.bias_mu = nn.Parameter(torch.empty(n_out).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(torch.full((n_out, n_in), sigma0 / math.sqrt(n_in)))
        self.bias_sigma = nn.Parameter(torch.full((n_out,), sigma0 / math.sqrt(n_in)))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.linear(x, self.weight_mu, self.bias_mu)
        e_in = self._f(torch.randn(self.n_in, device=x.device))
        e_out = self._f(torch.randn(self.n_out, device=x.device))
        w = self.weight_mu + self.weight_sigma * torch.outer(e_out, e_in)
        b = self.bias_mu + self.bias_sigma * e_out
        return F.linear(x, w, b)


def _mlp(n_in: int, width: int, noisy: bool = False, sigma0: float = 0.5) -> nn.Sequential:
    lin = (lambda a, b: NoisyLinear(a, b, sigma0)) if noisy else nn.Linear
    last = lin(width, 1)
    if not noisy:                                   # small initial scores
        nn.init.uniform_(last.weight, -1e-2, 1e-2)
        nn.init.zeros_(last.bias)
    return nn.Sequential(lin(n_in, width), nn.ReLU(), nn.LayerNorm(width), last)


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.hidden
        self.cfg = cfg
        self.status_emb = nn.Embedding(NUM_STATUSES, cfg.status_dim)
        self.op_emb = nn.Embedding(len(OPCODES), cfg.op_dim)
        self.inp = nn.Linear(cfg.status_dim + cfg.op_dim + 2, H)
        self.convs = nn.ModuleList()
        for _ in range(cfg.num_layers):
            if cfg.conv == "gin":
                mlp = nn.Sequential(nn.Linear(H, H), nn.LayerNorm(H), nn.ReLU(), nn.Linear(H, H))
                self.convs.append(GINConv(apply_func=mlp, aggregator_type="sum", learn_eps=True))
            elif cfg.conv == "gatv2":
                self.convs.append(GATv2Conv(H, H // cfg.heads, num_heads=cfg.heads,
                                            allow_zero_in_degree=True))
            else:
                self.convs.append(GraphConv(H, H, norm="both", allow_zero_in_degree=True))
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(cfg.num_layers)])
        self.gate = nn.Linear(H, 1) if cfg.pooling == "attention" else None
        self.glob = nn.Linear(cfg.graph_feat_dim, H)

    def forward(self, g: dgl.DGLGraph, status: torch.Tensor, graph_feats: torch.Tensor,
                node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Node embeddings ``[B, N, H]`` (zero at padding) and ``g' [B, 2H]``."""
        B, N = node_mask.shape
        if g.num_nodes() != int(node_mask.sum()):
            raise RuntimeError("graph batch and node mask disagree")
        codes = status[node_mask].long()
        if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= NUM_STATUSES):
            raise ValueError("node status code outside the status vocabulary")
        x = torch.cat([self.status_emb(codes), self.op_emb(g.ndata["op"]),
                       g.ndata["feat"].to(self.inp.weight.dtype)], dim=-1)
        x0 = F.relu(self.inp(x))
        h = x0
        if self.cfg.conv in ("gcn", "gatv2"):             # GCN and GAT include each node's own term
            g = dgl.add_self_loop(g)
        for conv, norm in zip(self.convs, self.norms):
            y = conv(g, h)
            if y.dim() == 3:                             # GATv2: concatenate the heads
                y = y.flatten(1)
            h = F.relu(norm(y) + x0)
        z = h.new_zeros(B, N, h.size(-1))
        z[node_mask] = h
        if self.cfg.pooling == "sum":
            pooled = z.sum(1)
        elif self.cfg.pooling == "mean":
            pooled = z.sum(1) / node_mask.sum(1, keepdim=True).clamp(min=1).to(z.dtype)
        else:
            s = self.gate(z).squeeze(-1).masked_fill(~node_mask, float("-inf"))
            w = torch.softmax(s, dim=1).nan_to_num(0.0).unsqueeze(-1)
            pooled = (w * z).sum(1)
        return z, torch.cat([pooled, F.relu(self.glob(graph_feats))], dim=-1)


class TacoNetwork(nn.Module):
    """Encoder plus the PPO or DQN head, applied to environment observations."""

    def __init__(self, cfg: ModelConfig, store: Optional[GraphStore] = None, *,
                 max_nodes_per_chunk: int = 200_000):
        super().__init__()
        self.cfg = cfg
        self.store = store
        self.max_nodes_per_chunk = int(max_nodes_per_chunk)
        H, Fw = cfg.hidden, cfg.head_width
        self.encoder = Encoder(cfg)
        self.stop_token = nn.Parameter(torch.zeros(H))
        if cfg.algo == "ppo":
            self.policy = _mlp(3 * H, Fw)
            self.value = _mlp(2 * H, Fw)
        else:
            self.advantage = _mlp(3 * H, Fw, cfg.noisy, cfg.sigma0)
            self.value = _mlp(2 * H, Fw, cfg.noisy, cfg.sigma0)

    def heads(self, z: torch.Tensor, gp: torch.Tensor, node_mask: torch.Tensor,
              action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores ``[B, N + 1]`` (logits for PPO, Q-values for DQN; STOP last)
        and state values ``[B]``."""
        B, N, H = z.shape
        node_in = torch.cat([z[node_mask], gp.unsqueeze(1).expand(-1, N, -1)[node_mask]], -1)
        stop_in = torch.cat([self.stop_token.expand(B, H), gp], -1)
        mlp = self.policy if self.cfg.algo == "ppo" else self.advantage
        scores = z.new_full((B, N + 1), 0.0)
        scores[:, :N][node_mask] = mlp(node_in).view(-1)
        scores[:, N] = mlp(stop_in).view(-1)
        mask = action_mask > 0.5 if action_mask.dtype != torch.bool else action_mask
        mask = mask.clone()
        mask[:, :N] &= node_mask
        empty = ~mask.any(1)
        mask[empty, N] = True                       # rows with no admissible action admit STOP
        value = self.value(gp).view(-1)
        if self.cfg.algo == "ppo":
            scores = scores / self.cfg.temperature
        else:                                       # dueling over the feasible actions
            mean_a = (scores * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
            scores = value.unsqueeze(1) + scores - mean_a
        return scores.masked_fill(~mask, MASK_FILL), value

    def _chunks(self, sizes: Sequence[int]) -> List[slice]:
        out, start, acc = [], 0, 0
        for i, n in enumerate(sizes):
            if i > start and acc + n > self.max_nodes_per_chunk:
                out.append(slice(start, i))
                start, acc = i, 0
            acc += n
        out.append(slice(start, len(sizes)))
        return out

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.store is None:
            raise RuntimeError("TacoNetwork has no graph store")
        dev = self.stop_token.device
        status = obs["node_status_vector"]
        B, N = status.shape
        idx = obs["graph_idx"].reshape(B).round().long().cpu().numpy()
        reported = obs["num_nodes"].reshape(B).round().long().cpu().numpy()
        sizes = np.asarray([self.store.num_nodes(i) for i in idx], dtype=np.int64)
        bad = (reported != 0) & (reported != sizes)
        if bad.any():
            r = int(np.flatnonzero(bad)[0])
            raise RuntimeError(f"observation row {r}: graph {int(idx[r])} has {int(sizes[r])} "
                               f"nodes in the store, {int(reported[r])} in the env")
        gf = obs["graph_feats"].to(dev, torch.float32)
        if gf.numel() and (float(gf.min()) < 0.0 or float(gf.max()) > 1.0):
            raise ValueError("graph features must lie in [0, 1]")
        status = status.to(dev)
        am = obs["action_mask"].to(dev)
        node_mask = torch.arange(N, device=dev).unsqueeze(0) < torch.as_tensor(sizes, device=dev).unsqueeze(1)
        scores, values = [], []
        for sl in self._chunks(sizes.tolist()):
            g = self.store.batch(idx[sl].tolist()).to(dev)
            z, gp = self.encoder(g, status[sl], gf[sl], node_mask[sl])
            s, v = self.heads(z, gp, node_mask[sl], am[sl])
            scores.append(s)
            values.append(v)
        return torch.cat(scores, 0), torch.cat(values, 0)
