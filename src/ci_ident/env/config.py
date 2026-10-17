"""Configuration of the custom-instruction identification environment.

Every field is read by the environment.  Unknown keys are rejected.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class RewardConfig:
    """Return of the formulation; all payments are in the cost model's own
    quantities.

    * commit:  ``commit_coef * [log S(committed + C) - log S(committed)]``
    * growth:  ``(shape_coef / |V|) * [Phi(C_{t+1}) - Phi(C_t)]`` with
      ``Phi(C) = saving_before_rounding(C) - area_price * A(C) / A_budget``
    * the accumulated potential is charged back whenever a candidate is
      committed, discarded or abandoned at the end of the episode, so the
      growth payments of every construction sum to zero and the episode
      return equals ``commit_coef * log(final speedup)``.

    The return is undiscounted (gamma = 1); the learner must also use 1.
    """

    commit_coef: float = 5.0
    shape_coef: float = 5.0
    area_price: float = 0.0

    def __post_init__(self):
        for name in ("commit_coef", "shape_coef", "area_price"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                raise ValueError(f"reward.{name} must be a finite number, got {v!r}")
            if v < 0:
                raise ValueError(f"reward.{name} must be >= 0, got {v!r}")
        if self.commit_coef <= 0:
            raise ValueError("reward.commit_coef must be > 0")


@dataclass(frozen=True)
class EnvConfig:
    """Environment configuration.

    ``io``            per-CI port bound ``(max_in, max_out)``
    ``area_factor``   global area budget as a fraction of ``A_ref``
    ``radius``        growth radius in hops of the undirected DDG; 1 admits
                      only neighbours of the candidate (connected CIs),
                      r >= 2 also admits nodes up
                      to r hops away (disjoint CIs)
    ``max_nodes``     size of the padded action space (STOP is action ``max_nodes``)
    """

    io: Tuple[int, int] = (4, 2)
    area_factor: float = 0.4
    radius: int = 1
    max_nodes: int = 2000
    reward: RewardConfig = field(default_factory=RewardConfig)
    # episode limits
    max_steps_factor: float = 10.0       # decision cap = factor * |V|
    no_progress_base: int = 12           # patience = clip(ceil(base * log2 |V|), 6, 70)
    no_progress_eps: float = 1e-6        # growth counts as progress if dPhi > eps
    stall_patience: int = 4              # consecutive states with a non-empty CI and no node action
    violation_factor: float = 3.0        # episode ends after factor * |V| refused actions

    def __post_init__(self):
        mi, mo = self.io
        if int(mi) < 1 or int(mo) < 1:
            raise ValueError(f"io must be positive, got {self.io}")
        object.__setattr__(self, "io", (int(mi), int(mo)))
        if not 0.0 < float(self.area_factor) <= 1.0:
            raise ValueError(f"area_factor must be in (0, 1], got {self.area_factor}")
        if int(self.radius) < 1:
            raise ValueError(f"radius must be >= 1, got {self.radius}")
        if int(self.max_nodes) < 2:
            raise ValueError("max_nodes must be >= 2")

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "EnvConfig":
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown environment keys: {sorted(unknown)}")
        if "reward" in d and not isinstance(d["reward"], RewardConfig):
            r = dict(d["reward"])
            rk = {f.name for f in dataclasses.fields(RewardConfig)}
            if set(r) - rk:
                raise ValueError(f"unknown reward keys: {sorted(set(r) - rk)}")
            d["reward"] = RewardConfig(**r)
        if "io" in d:
            d["io"] = tuple(d["io"])
        return cls(**d)
