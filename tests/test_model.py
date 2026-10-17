import numpy as np
import pytest
import torch

from ci_ident.env import CIEnv
from ci_ident.model import GraphStore, ModelConfig, PolicyNetwork


def _obs_batch(manifest, picks, steps=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gi in picks:
        env = CIEnv({"corpus": manifest, "graphs": [gi]})
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


@pytest.fixture(scope="module")
def net(synth_corpus):
    torch.manual_seed(0)
    return PolicyNetwork(ModelConfig(), GraphStore(synth_corpus)).eval()


def test_shapes_and_mask(net, synth_corpus):
    obs = _obs_batch(synth_corpus, list(range(0, 30, 3)))
    logits, value = net(obs)
    assert logits.shape == (10, 2001) and value.shape == (10,)
    am = obs["action_mask"].bool()
    assert (logits[~am] <= -1e8).all() and (logits[am] > -1e8).all()
    a = logits.argmax(1)
    assert am[torch.arange(10), a].all()


def test_policy_is_a_function_of_the_state(net, synth_corpus):
    """No batch statistics: a row's output does not depend on the other rows
    or on train/eval mode."""
    obs = _obs_batch(synth_corpus, list(range(0, 30, 3)))
    full, _ = net(obs)
    one, _ = net({k: v[:1] for k, v in obs.items()})
    am = obs["action_mask"][:1].bool()
    assert torch.allclose(full[:1][am], one[am], atol=1e-5)
    net.train()
    try:
        tr, _ = net(obs)
    finally:
        net.eval()
    assert torch.equal(tr, full)


def test_chunking_is_exact(synth_corpus):
    torch.manual_seed(1)
    a = PolicyNetwork(ModelConfig(), GraphStore(synth_corpus)).eval()
    b = PolicyNetwork(ModelConfig(), GraphStore(synth_corpus), max_nodes_per_chunk=50).eval()
    b.load_state_dict(a.state_dict())
    obs = _obs_batch(synth_corpus, list(range(0, 36, 2)))
    la, va = a(obs)
    lb, vb = b(obs)
    assert torch.allclose(la, lb, atol=1e-5) and torch.allclose(va, vb, atol=1e-5)


def test_identity_check(net, synth_corpus):
    obs = _obs_batch(synth_corpus, [4, 9])
    obs["num_nodes"] = obs["num_nodes"] + 1
    with pytest.raises(RuntimeError, match="nodes in the store"):
        net(obs)


def test_dummy_batch(net):
    """RLlib initialises the loss with an all-zero observation batch."""
    z = {"node_status_vector": torch.zeros(2, 2000), "node_dynamic_feats": torch.zeros(2, 2000, 3),
         "graph_feats": torch.zeros(2, 16), "action_mask": torch.zeros(2, 2001),
         "graph_idx": torch.zeros(2, 1), "num_nodes": torch.zeros(2, 1)}
    logits, _ = net(z)
    assert (logits.argmax(1) == 2000).all()


def test_export_roundtrip(tmp_path, net, synth_corpus):
    from ci_ident.rl.infer import FORMAT, greedy_episode, load_model

    p = tmp_path / "m.pt"
    torch.save({"format": FORMAT, "state_dict": net.backbone.state_dict(), "model": {},
                "env": {"io": [4, 2], "area_factor": 0.4}, "provenance": {}}, p)
    net2, payload = load_model(p, synth_corpus)
    env = CIEnv({"corpus": synth_corpus, "graphs": [7], **payload["env"]})
    r1 = greedy_episode(net, CIEnv({"corpus": synth_corpus, "graphs": [7], **payload["env"]}))
    r2 = greedy_episode(net2, env)
    assert r1["cis"] == r2["cis"] and r1["speedup"] == r2["speedup"]
    bad = net.backbone.state_dict()
    bad.pop(next(iter(bad)))
    torch.save({"format": FORMAT, "state_dict": bad, "model": {}, "env": {}}, p)
    with pytest.raises(RuntimeError):
        load_model(p, synth_corpus)
