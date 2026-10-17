"""RLlib wrapper of :class:`ci_ident.model.PolicyNetwork`, registered as ``"ci_gin"``."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ci_ident.model import GraphStore, ModelConfig, PolicyNetwork

MODEL_NAME = "ci_gin"


class RLlibCIModel(TorchModelV2, nn.Module):
    """``custom_model_config = {"corpus": <manifest>, "model": {ModelConfig
    fields}, "max_nodes_per_chunk": int}``.  The corpus must be the one the
    environment draws from: the model reads the static graph of ``graph_idx``
    from it (and checks ``num_nodes`` against it on every row)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cmc = dict(model_config.get("custom_model_config") or {})
        unknown = set(cmc) - {"corpus", "model", "max_nodes_per_chunk"}
        if unknown:
            raise ValueError(f"unknown custom_model_config keys: {sorted(unknown)}")
        if "corpus" not in cmc:
            raise KeyError("custom_model_config['corpus'] is required")
        if num_outputs != action_space.n:
            raise ValueError(f"num_outputs {num_outputs} != action_space.n {action_space.n}")
        self.net = PolicyNetwork(ModelConfig.from_dict(cmc.get("model")), GraphStore(cmc["corpus"]),
                                 max_nodes_per_chunk=int(cmc.get("max_nodes_per_chunk", 200_000)))
        self._value: Optional[torch.Tensor] = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if not isinstance(obs, dict):
            obs = restore_original_dimensions(obs, self.obs_space, "torch")
        logits, self._value = self.net(obs)
        return logits, state

    def value_function(self):
        if self._value is None:
            raise RuntimeError("value_function() called before forward()")
        return self._value


ModelCatalog.register_custom_model(MODEL_NAME, RLlibCIModel)
