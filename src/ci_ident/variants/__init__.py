"""The eight RL-GNN-CI variants of the paper ({Base, NoMask, NoShape,
AbsGain} x {PPO, DQN}), as the configuration files next to this module (``ci_ident/variants/*.yaml``).

A variant file has two sections: ``env`` (:class:`ci_ident.env.EnvConfig`
fields, with ``variant`` naming the environment configuration) and ``model``
(:class:`ci_ident.model.ModelConfig` fields).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

from ci_ident.env.config import EnvConfig
from ci_ident.model.network import ModelConfig

NAMES = tuple(f"{a}-{v}" for a in ("dqn", "ppo") for v in ("base", "nomask", "noshape", "absgain"))
CONFIG_DIR = Path(__file__).resolve().parent


def load_variant(name_or_path: str) -> Tuple[dict, ModelConfig]:
    """``(env_dict, model_config)`` of a variant (by name or file path).  The
    environment dictionary is validated and can be passed to
    :class:`ci_ident.env.CIEnv` together with ``corpus``, ``io``, ...."""
    p = Path(name_or_path)
    if not p.suffix:
        if name_or_path not in NAMES:
            raise KeyError(f"unknown variant {name_or_path!r}; one of {NAMES}")
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    spec = yaml.safe_load(p.read_text())
    if set(spec) - {"env", "model"}:
        raise ValueError(f"{p}: unknown sections {sorted(set(spec) - {'env', 'model'})}")
    env = dict(spec.get("env") or {})
    EnvConfig.from_dict(env)
    return env, ModelConfig.from_dict(spec.get("model"))


def all_variants() -> Dict[str, Tuple[dict, ModelConfig]]:
    return {n: load_variant(n) for n in NAMES}
