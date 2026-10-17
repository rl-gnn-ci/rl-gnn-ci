"""End-to-end: a few PPO iterations, export, greedy evaluation (slow)."""
import json

import pytest

pytestmark = pytest.mark.slow


def test_train_export_eval(tmp_path, synth_corpus, bench_manifest):
    pytest.importorskip("ray")
    from ci_ident.rl.infer import evaluate
    from ci_ident.rl.train import resolve_config, train

    cfg = resolve_config({"corpus": synth_corpus, "iterations": 2, "checkpoint_every": 2,
                          "ppo": {"train_batch_size": 128, "sgd_minibatch_size": 64},
                          "rollout": {"num_env_runners": 0},
                          "sampler": {"size_min": 10, "size_max": 60}})
    model = train(cfg, tmp_path / "run")
    rows = [json.loads(l) for l in (tmp_path / "run" / "progress.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [1, 2]
    for r in rows:   # the return is the commit payment; shaping telescopes to 0
        assert r["episode_reward_mean"] == pytest.approx(r["return_commit_mean"], abs=1e-6)
    # resuming continues the entropy schedule (iteration 3 trains with entropy_at(2))
    from ci_ident.rl.callbacks import entropy_at
    cfg3 = dict(cfg, iterations=3, checkpoint_every=3)
    train(cfg3, tmp_path / "run", resume=str(tmp_path / "run" / "rllib" / "iter_000002"))
    last = json.loads((tmp_path / "run" / "progress.jsonl").read_text().splitlines()[-1])
    assert last["iteration"] == 3
    assert last["entropy_coeff"] == pytest.approx(entropy_at(2, 0.01, 0.0005, 244))
    out = evaluate(model, bench_manifest, [(7, (4, 2), 0.4)])
    assert out[0]["speedup"] >= 1.0 and out[0]["area_used"] <= out[0]["area_budget"] + 1e-9
