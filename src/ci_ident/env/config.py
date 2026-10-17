"""Configuration of the environment (TACO formulation).

A configuration is a *variant* (constraint enforcement x reward signal, the two
ablation axes of the paper) plus the reward weights, the port bound, the area
factor and the MDP discount.  Unknown keys are rejected; every field is read.

Ablation axes (paper, Section "Ablation Study Design"):

=========  ============  =================================================
variant    mask          reward
=========  ============  =================================================
Base       masked        eta modulator, potential shaping, size term, G = rho
NoMask     none          as Base
NoShape    masked        eta = 1, no shaping, no size term
AbsGain    masked        as Base with G = Delta
=========  ============  =================================================
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

MASKS = ("masked", "none")
REWARDS = ("full", "noshape", "absgain")
MODES = ("train", "inference")

#: The four environment configurations of the paper.
VARIANTS = {
    "base": ("masked", "full"),
    "nomask": ("none", "full"),
    "noshape": ("masked", "noshape"),
    "absgain": ("masked", "absgain"),
}


@dataclass(frozen=True)
class RewardWeights:
    """Scalar weights of the reward (paper symbols in brackets).

    ``w_invalid``, ``w_bad_stop`` and ``w_empty_stop`` are signed (<= 0) and are
    divided by ``|V|``, as are ``w_size`` and ``w_shp``.
    """

    w_perf: float = 8.0          # [w_perf] commit-time log-speedup
    w_size: float = 1.0          # [w_size] commit-time size term h(|C|)
    w_ext: float = 0.3           # [w_ext] per-growth extension term eta
    w_shp: float = 0.1           # [w_shp] potential shaping
    lambda_io: float = 0.5       # [lambda_io] I/O-overage weight in Phi
    lambda_conv: float = 0.5     # [lambda_conv] convexity-debt weight in Phi
    w_invalid: float = -0.2      # illegal node action
    w_bad_stop: float = -0.5     # STOP on an infeasible candidate
    w_empty_stop: float = -1.0   # STOP on an empty candidate

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                raise ValueError(f"reward weight {f.name} must be a finite number, got {v!r}")
        for name in ("w_perf", "w_size", "w_ext", "w_shp", "lambda_io", "lambda_conv"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("w_invalid", "w_bad_stop", "w_empty_stop"):
            if getattr(self, name) > 0:
                raise ValueError(f"{name} is a penalty and must be <= 0")
        if self.w_perf <= 0:
            raise ValueError("w_perf must be > 0")


@dataclass(frozen=True)
class EnvConfig:
    """``mask``/``reward``: the variant (or set ``variant``); ``mode``: the
    training mask (frontier, post-hoc checks, temporary regular-I/O overflow)
    or the inference mask (every growth predicate); ``gamma``: the MDP
    discount, which the shaping uses and the learner must use too."""

    mask: str = "masked"
    reward: str = "full"
    mode: str = "train"
    io: Tuple[int, int] = (4, 2)
    area_factor: float = 0.4
    gamma: float = 0.99
    max_nodes: int = 2000
    max_steps_factor: float = 10.0
    weights: RewardWeights = field(default_factory=RewardWeights)

    def __post_init__(self):
        if self.mask not in MASKS:
            raise ValueError(f"mask must be one of {MASKS}, got {self.mask!r}")
        if self.reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {self.reward!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        mi, mo = self.io
        if int(mi) < 1 or int(mo) < 1:
            raise ValueError(f"io must be positive, got {self.io}")
        object.__setattr__(self, "io", (int(mi), int(mo)))
        if not 0.0 < float(self.area_factor) <= 1.0:
            raise ValueError(f"area_factor must be in (0, 1], got {self.area_factor}")
        if not 0.0 < float(self.gamma) < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if int(self.max_nodes) < 2:
            raise ValueError("max_nodes must be >= 2")
        if float(self.max_steps_factor) <= 0:
            raise ValueError("max_steps_factor must be > 0")

    @property
    def shaping(self) -> bool:
        return self.reward in ("full", "absgain")

    @property
    def gain(self) -> str:
        """``"ratio"`` (G = rho) or ``"absolute"`` (G = Delta)."""
        return "absolute" if self.reward == "absgain" else "ratio"

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "EnvConfig":
        d = dict(d or {})
        if "variant" in d:
            v = str(d.pop("variant")).lower()
            if v not in VARIANTS:
                raise ValueError(f"unknown variant {v!r}; one of {sorted(VARIANTS)}")
            if "mask" in d or "reward" in d:
                raise ValueError("give either 'variant' or 'mask'/'reward', not both")
            d["mask"], d["reward"] = VARIANTS[v]
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown environment keys: {sorted(unknown)}")
        if "weights" in d and not isinstance(d["weights"], RewardWeights):
            w = dict(d["weights"])
            wk = {f.name for f in dataclasses.fields(RewardWeights)}
            if set(w) - wk:
                raise ValueError(f"unknown reward weights: {sorted(set(w) - wk)}")
            d["weights"] = RewardWeights(**w)
        if "io" in d:
            d["io"] = tuple(d["io"])
        return cls(**d)
