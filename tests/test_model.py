import numpy as np
import pytest
import torch

from ci_ident.env import CIEnv
from ci_ident.model import GraphStore, TacoNetwork
from ci_ident.variants import NAMES, load_variant


def _obs_batch(manifest, picks, variant_env, steps=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gi in picks:
        env = CIEnv({**variant_env, "corpus": manifest, "graphs": [gi], "mode": "inference"})
        o, _ = env.reset()
        for _ in range(int(rng.integers(0, steps))):
            legal = np.flatnonzero(o["action_mask"][:-1])
            if not legal.size:
                break
            o, _, t, tr, _ = env.step(int(rng.choice(legal)))
            if t or tr:
                o, _ = env.reset()
                break
        rows.append(o)
    return {k: torch.as_tensor(np.stack([r[k] for r in rows])) for k in rows[0]}


@pytest.mark.parametrize("name", NAMES)
def test_variant_networks(name, synth_corpus):
    env_cfg, mcfg = load_variant(name)
    torch.manual_seed(0)
    net = TacoNetwork(mcfg, GraphStore(synth_corpus)).eval()
    obs = _obs_batch(synth_corpus, list(range(0, 36, 4)), env_cfg)
    B = obs["node_status_vector"].shape[0]
    with torch.no_grad():
        s, v = net(obs)
        one, _ = net({k: x[:1] for k, x in obs.items()})
    am = obs["action_mask"].bool()
    assert s.shape == (B, 2001) and v.shape == (B,)
    assert (s[~am] <= -1e8).all() and (s[am] > -1e8).all()
    assert am[torch.arange(B), s.argmax(1)].all()
    assert torch.allclose(s[:1][am[:1]], one[am[:1]], atol=1e-4), "batch-independent"
    if mcfg.algo == "dqn":
        # dueling: the advantages average to zero over the feasible actions
        adv = s - v.unsqueeze(1)
        mean = (adv * am).sum(1) / am.sum(1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    if not mcfg.noisy:
        net.train()
        with torch.no_grad():
            s_tr, _ = net(obs)
        assert torch.allclose(s_tr, s, atol=1e-5), "no train/eval dependence"


def test_noisy_layers_are_stochastic_only_in_training(synth_corpus):
    env_cfg, mcfg = load_variant("dqn-absgain")
    net = TacoNetwork(mcfg, GraphStore(synth_corpus))
    obs = _obs_batch(synth_corpus, [3, 9], env_cfg)
    with torch.no_grad():
        net.train()
        a, _ = net(obs)
        b, _ = net(obs)
        net.eval()
        c, _ = net(obs)
        d, _ = net(obs)
    assert not torch.equal(a, b) and torch.equal(c, d)


def test_greedy_roundtrip(tmp_path, bench_manifest):
    from ci_ident.rl.infer import evaluate, save_model

    env_cfg, mcfg = load_variant("dqn-noshape")
    torch.manual_seed(0)
    net = TacoNetwork(mcfg)
    p = save_model(tmp_path / "m.pt", net, env_cfg, variant="dqn-noshape")
    rows = evaluate(p, bench_manifest, [(4, (4, 2), 0.4), (7, (2, 1), 1.0)])
    for r in rows:
        assert r["speedup"] >= 1.0 and r["area_used"] <= r["area_budget"] + 1e-9
        assert r["variant"] == "dqn-noshape"
