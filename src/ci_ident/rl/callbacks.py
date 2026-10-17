"""Training callbacks: the entropy schedule and episode metrics."""

from __future__ import annotations

import math

from ray.rllib.algorithms.callbacks import DefaultCallbacks

#: metrics copied from the final ``info`` of every episode
EPISODE_METRICS = ("speedup", "area_util", "num_patterns", "num_instances", "mean_ci_size",
                   "steps", "violations", "return_commit", "return_shape")


def entropy_at(iteration: int, start: float, end: float, decay_iters: int) -> float:
    """``start * (end / start) ** (t / decay_iters)`` for ``t < decay_iters``,
    then ``end``; ``iteration`` counts completed training iterations."""
    if decay_iters <= 0 or iteration >= decay_iters:
        return float(end)
    return float(start * (end / start) ** (iteration / decay_iters))


def make_callbacks(entropy_start: float, entropy_end: float, entropy_decay_iters: int):
    if not (entropy_start > 0 and entropy_end > 0):
        raise ValueError("entropy coefficients must be > 0")

    class CICallbacks(DefaultCallbacks):
        """The entropy coefficient is a function of the training iteration
        only; it is recomputed from ``training_iteration``, also after
        restoring a checkpoint.  The KL coefficient is
        RLlib's own adaptive KL penalty (``use_kl_loss``)."""

        def _set_entropy(self, algorithm, iteration):
            value = entropy_at(iteration, entropy_start, entropy_end, entropy_decay_iters)
            algorithm.env_runner.foreach_policy(
                lambda p, _pid: setattr(p, "entropy_coeff", value))
            return value

        def on_algorithm_init(self, *, algorithm, **kwargs):
            self._set_entropy(algorithm, int(getattr(algorithm, "iteration", 0)))

        def on_train_result(self, *, algorithm, result, **kwargs):
            # the value just used is in the learner stats; set the next one
            value = self._set_entropy(algorithm, int(result["training_iteration"]))
            result.setdefault("custom_metrics", {})["entropy_coeff_next"] = value

        def on_episode_end(self, *, episode, **kwargs):
            info = episode.last_info_for() or {}
            for k in EPISODE_METRICS:
                v = info.get(k)
                if v is not None and math.isfinite(float(v)):
                    episode.custom_metrics[k] = float(v)

    return CICallbacks
