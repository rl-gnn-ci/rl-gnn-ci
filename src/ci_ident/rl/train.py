"""PPO training (``ci-ident-train``).

The configuration is one YAML/JSON file (``configs/ppo.yaml`` is the
reference configuration); every key is validated.
Each training run writes to ``<out>/``:

* ``config.json``     -- the resolved configuration;
* ``progress.jsonl``  -- one line of training metrics per iteration;
* ``rllib/iter_XXXXXX`` -- RLlib checkpoints (for resuming);
* ``model_iter_XXXXXX.pt`` -- exported network (state dict, model and
  environment configuration, provenance) for :mod:`ci_ident.rl.infer`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

import ci_ident
from ci_ident.env import CIEnv
from ci_ident.env.config import EnvConfig
from ci_ident.model import ModelConfig

ENV_NAME = "ci_env"

DEFAULTS: Dict[str, Any] = {
    "seed": 0,
    "iterations": 3000,
    "checkpoint_every": 25,
    "corpus": None,                       # training corpus manifest (required)
    "env": {                              # EnvConfig fields
        "io": [4, 2], "area_factor": 0.4, "radius": 1, "max_nodes": 2000,
        "reward": {"commit_coef": 5.0, "shape_coef": 5.0, "area_price": 0.0},
    },
    "sampler": {"size_min": 10, "size_max": 2000, "exponent": 1.0, "mode": "per_size"},
    "model": {},                          # ModelConfig fields (defaults: the reference network)
    "max_nodes_per_chunk": 200_000,
    "ppo": {
        "lr": 1e-4, "lambda_": 1.0,
        "clip_param": 0.3, "vf_clip_param": 20.0, "vf_loss_coeff": 1.0,
        "use_kl_loss": True, "kl_coeff": 0.2, "kl_target": 0.015,
        "grad_clip": 1.0,
        "train_batch_size": 4096, "sgd_minibatch_size": 512, "num_sgd_iter": 2,
    },
    "entropy": {"start": 0.01, "end": 0.0005, "decay_iters": 244},
    "rollout": {"num_env_runners": 16, "rollout_fragment_length": 8,
                "batch_mode": "truncate_episodes"},
    "resources": {"num_gpus": 0},
}


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise KeyError(f"unknown configuration key {path + k!r}")
        if isinstance(base[k], dict) and base[k] and isinstance(v, dict) and k not in ("model",):
            out[k] = _merge(base[k], v, f"{path}{k}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config(user: Optional[dict] = None) -> dict:
    cfg = _merge(DEFAULTS, user or {})
    if not cfg["corpus"]:
        raise KeyError("'corpus' (training corpus manifest) is required")
    cfg["corpus"] = str(Path(cfg["corpus"]).expanduser().resolve())
    EnvConfig.from_dict(cfg["env"])            # validate
    ModelConfig.from_dict(cfg["model"])        # validate
    if cfg["rollout"]["batch_mode"] not in ("truncate_episodes", "complete_episodes"):
        raise ValueError("rollout.batch_mode must be truncate_episodes or complete_episodes")
    return cfg


def env_config(cfg: dict) -> dict:
    return {**copy.deepcopy(cfg["env"]), "corpus": cfg["corpus"],
            "sampler": dict(cfg["sampler"]), "seed": int(cfg["seed"])}


def build_algorithm_config(cfg: dict):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    from ci_ident.rl.callbacks import make_callbacks
    from ci_ident.rl.model import MODEL_NAME

    register_env(ENV_NAME, lambda c: CIEnv(c))
    p, e, r = cfg["ppo"], cfg["entropy"], cfg["rollout"]
    return (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .framework("torch")
        .environment(ENV_NAME, env_config=env_config(cfg), clip_rewards=False,
                     normalize_actions=False, clip_actions=False)
        .env_runners(num_env_runners=int(r["num_env_runners"]), num_envs_per_env_runner=1,
                     rollout_fragment_length=int(r["rollout_fragment_length"]),
                     batch_mode=r["batch_mode"])
        .training(
            # The return is undiscounted and telescopes to commit_coef * log S.
            gamma=1.0, lambda_=float(p["lambda_"]), use_gae=True, use_critic=True,
            lr=float(p["lr"]), clip_param=float(p["clip_param"]),
            vf_clip_param=float(p["vf_clip_param"]), vf_loss_coeff=float(p["vf_loss_coeff"]),
            use_kl_loss=bool(p["use_kl_loss"]), kl_coeff=float(p["kl_coeff"]),
            kl_target=float(p["kl_target"]), entropy_coeff=float(e["start"]),
            grad_clip=float(p["grad_clip"]), grad_clip_by="global_norm",
            train_batch_size=int(p["train_batch_size"]),
            sgd_minibatch_size=int(p["sgd_minibatch_size"]), num_sgd_iter=int(p["num_sgd_iter"]),
            model={"custom_model": MODEL_NAME,
                   "custom_model_config": {"corpus": cfg["corpus"], "model": dict(cfg["model"]),
                                           "max_nodes_per_chunk": int(cfg["max_nodes_per_chunk"])}},
        )
        .callbacks(make_callbacks(float(e["start"]), float(e["end"]), int(e["decay_iters"])))
        .resources(num_gpus=cfg["resources"]["num_gpus"])
        .debugging(seed=int(cfg["seed"]))
    )


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_model(algo, cfg: dict, path, iteration: int) -> Path:
    """Write the network weights with everything needed to evaluate them."""
    policy = algo.get_policy()
    net = policy.model.net
    payload = {
        "format": "ci_ident.model/1",
        "state_dict": {k: v.detach().cpu() for k, v in net.backbone.state_dict().items()},
        "model": dict(cfg["model"]),
        "env": copy.deepcopy(cfg["env"]),
        "provenance": {
            "package_version": ci_ident.__version__,
            "iteration": int(iteration),
            "seed": int(cfg["seed"]),
            "training_corpus": cfg["corpus"],
            "training_corpus_sha256": _file_sha256(cfg["corpus"]),
            "config": cfg,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    }
    path = Path(path)
    torch.save(payload, path)
    return path


_PROGRESS_KEYS = ("episode_reward_mean", "episode_len_mean", "episodes_this_iter")


def _progress_row(result: dict) -> dict:
    row = {"iteration": result["training_iteration"],
           "timesteps": result.get("num_env_steps_sampled_lifetime",
                                   result.get("timesteps_total"))}
    src = result.get("env_runners", result)
    for k in _PROGRESS_KEYS:
        if k in src:
            row[k] = src[k]
    for k, v in (src.get("custom_metrics") or {}).items():
        if k.endswith("_mean"):
            row[k] = v
    stats = (result.get("info", {}).get("learner", {}).get("default_policy", {})
             .get("learner_stats", {}))
    for k in ("policy_loss", "vf_loss", "kl", "entropy", "entropy_coeff", "cur_kl_coeff",
              "vf_explained_var", "cur_lr"):
        if k in stats:
            row[k] = stats[k]
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}


def train(cfg: dict, out_dir, *, resume: Optional[str] = None) -> Path:
    import ray

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    if not ray.is_initialized():
        ray.init()
    algo = build_algorithm_config(cfg).build()
    if resume:
        algo.restore(resume)
        # restoring re-reads the policy's saved entropy coefficient; put the
        # schedule's value for the restored iteration back
        from ci_ident.rl.callbacks import entropy_at
        e = cfg["entropy"]
        value = entropy_at(int(algo.iteration), float(e["start"]), float(e["end"]), int(e["decay_iters"]))
        algo.env_runner.foreach_policy(lambda p, _pid: setattr(p, "entropy_coeff", value))
    last = None
    with open(out / "progress.jsonl", "a") as log:
        while algo.iteration < int(cfg["iterations"]):
            result = algo.train()
            it = int(result["training_iteration"])
            row = _progress_row(result)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if it % int(cfg["checkpoint_every"]) == 0 or it == int(cfg["iterations"]):
                algo.save(str(out / "rllib" / f"iter_{it:06d}"))
                last = export_model(algo, cfg, out / f"model_iter_{it:06d}.pt", it)
    algo.stop()
    return last


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ci-ident-train", description="PPO training.")
    ap.add_argument("config", help="YAML or JSON configuration (see configs/ppo.yaml)")
    ap.add_argument("out_dir", help="output directory")
    ap.add_argument("--corpus", help="override 'corpus'")
    ap.add_argument("--seed", type=int, help="override 'seed'")
    ap.add_argument("--iterations", type=int, help="override 'iterations'")
    ap.add_argument("--num-env-runners", type=int, help="override 'rollout.num_env_runners'")
    ap.add_argument("--num-gpus", type=float, help="override 'resources.num_gpus'")
    ap.add_argument("--resume", help="RLlib checkpoint directory to resume from")
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    if a.config.endswith((".yaml", ".yml")):
        import yaml
        user = yaml.safe_load(text) or {}
    else:
        user = json.loads(text)
    for key, val in (("corpus", a.corpus), ("seed", a.seed), ("iterations", a.iterations)):
        if val is not None:
            user[key] = val
    if a.num_env_runners is not None:
        user.setdefault("rollout", {})["num_env_runners"] = a.num_env_runners
    if a.num_gpus is not None:
        user.setdefault("resources", {})["num_gpus"] = a.num_gpus
    path = train(resolve_config(user), a.out_dir, resume=a.resume)
    print(f"exported {path}")


if __name__ == "__main__":
    main()
