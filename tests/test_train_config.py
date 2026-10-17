import math

import pytest

from ci_ident.rl.train import DEFAULTS, env_config, resolve_config


def test_defaults_and_validation(synth_corpus):
    cfg = resolve_config({"corpus": synth_corpus})
    assert cfg["env"]["reward"]["area_price"] == 0.0 and cfg["env"]["radius"] == 1
    ec = env_config(cfg)
    assert ec["corpus"] == cfg["corpus"] and ec["sampler"]["size_max"] == 2000
    with pytest.raises(KeyError, match="unknown"):
        resolve_config({"corpus": synth_corpus, "ppo": {"gamma": 0.99}})
    with pytest.raises(KeyError, match="unknown"):
        resolve_config({"corpus": synth_corpus, "env": {"reward": {"size_bonus": 1.0}}})
    with pytest.raises(KeyError):
        resolve_config({})
    assert DEFAULTS["ppo"]["lambda_"] == 1.0


def test_entropy_schedule():
    pytest.importorskip("ray")
    from ci_ident.rl.callbacks import entropy_at

    assert entropy_at(0, 0.01, 0.0005, 244) == 0.01
    assert entropy_at(122, 0.01, 0.0005, 244) == pytest.approx(math.sqrt(0.01 * 0.0005))
    assert entropy_at(244, 0.01, 0.0005, 244) == 0.0005
    assert entropy_at(10_000, 0.01, 0.0005, 244) == 0.0005
