"""Environment invariants, checked independently of the environment's own code."""
import math

import numpy as np
import pytest

from _check import check_solution
from ci_ident.data import Corpus
from ci_ident.ddg.graph import area_reference, load_ddg
from ci_ident.ddg.signature import SignatureIndex
from ci_ident.env import CIEnv

SETTINGS = [((2, 1), 0.2, 1), ((4, 2), 0.4, 1), ((6, 3), 1.0, 1), ((4, 2), 0.4, 2), ((12, 6), 1.0, 6)]


def _random_episode(env, rng, p_stop=0.2):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    ret, info = 0.0, {}
    while True:
        m = obs["action_mask"]
        legal = np.flatnonzero(m[:-1])
        if m[-1] and (legal.size == 0 or rng.random() < p_stop):
            a = env.max_nodes
        else:
            assert legal.size, "no admissible action in a non-terminal state"
            a = int(rng.choice(legal))
        obs, r, term, trunc, info = env.step(a)
        ret += r
        if env.ci:
            assert env._commit_valid(), "a candidate built from admissible actions must be committable"
        if term or trunc:
            return ret, info
        assert env.observation_space.contains(obs)


@pytest.mark.parametrize("io,af,radius", SETTINGS)
def test_random_policy_episodes_are_valid(synth_corpus, io, af, radius):
    corpus = Corpus.load(synth_corpus)
    rng = np.random.default_rng(hash((io, af, radius)) & 0xFFFF)
    for gi in range(0, len(corpus), 3):
        env = CIEnv({"corpus": synth_corpus, "graphs": [gi], "io": list(io),
                     "area_factor": af, "radius": radius})
        ret, info = _random_episode(env, rng)
        g = load_ddg(corpus.dot_path(gi))
        sig = SignatureIndex.from_graph(g).signature
        budget = af * area_reference(g)
        cis = env.committed_cis()
        sp, used = check_solution(g, cis, io[0], io[1], budget, signature=sig,
                                  connected=(radius == 1))
        s = env.episode_summary()
        assert s["speedup"] == pytest.approx(sp, rel=1e-12)
        assert s["area_used"] == pytest.approx(used, abs=1e-9)
        assert s["area_budget"] == pytest.approx(budget)
        # the return telescopes to commit_coef * log(speedup of all commits)
        assert ret == pytest.approx(5.0 * math.log(env.speedup()), abs=1e-6)
        assert s["return_shape"] == pytest.approx(0.0, abs=1e-9)
        assert info["end_reasons"]


def test_benchmarks_random_policy(bench_manifest):
    corpus = Corpus.load(bench_manifest)
    rng = np.random.default_rng(7)
    for gi in range(len(corpus)):
        env = CIEnv({"corpus": bench_manifest, "graphs": [gi], "io": [4, 2], "area_factor": 0.4})
        ret, _ = _random_episode(env, rng, p_stop=0.3)
        g = load_ddg(corpus.dot_path(gi))
        sp, _ = check_solution(g, env.committed_cis(), 4, 2, 0.4 * area_reference(g),
                               signature=SignatureIndex.from_graph(g).signature, connected=True)
        assert env.episode_summary()["speedup"] == pytest.approx(sp, rel=1e-12)


def test_stop_and_masked_actions(synth_corpus):
    env = CIEnv({"corpus": synth_corpus, "graphs": [5]})
    obs, _ = env.reset()
    assert not obs["action_mask"][env.max_nodes], "STOP must be masked on an empty candidate"
    n = env.static.num_nodes
    assert not obs["action_mask"][n:env.max_nodes].any(), "padding actions must be masked"
    status_before = obs["node_status_vector"].copy()
    # a masked action changes nothing but the counters
    masked = int(np.flatnonzero(~obs["action_mask"][:n].astype(bool))[0])
    obs2, r, term, trunc, _ = env.step(masked)
    assert r == 0.0 and not term and env.no_progress == 1 and not env.ci
    assert np.array_equal(obs2["node_status_vector"], status_before)
    assert np.array_equal(obs2["action_mask"], obs["action_mask"])
    # STOP on an empty candidate is a violation
    env.step(env.max_nodes)
    assert env.violations == 1 and not env.ci


def test_observation_features_in_unit_interval(bench_manifest):
    rng = np.random.default_rng(3)
    env = CIEnv({"corpus": bench_manifest, "graphs": [3], "io": [6, 3], "area_factor": 1.0})
    obs, _ = env.reset()
    for _ in range(200):
        for k in ("graph_feats", "node_dynamic_feats"):
            assert obs[k].min() >= 0.0 and obs[k].max() <= 1.0, k
        m = obs["action_mask"]
        legal = np.flatnonzero(m)
        obs, _, term, trunc, _ = env.step(int(rng.choice(legal)))
        if term or trunc:
            obs, _ = env.reset()


def test_training_sampler_is_seeded(synth_corpus):
    cfg = {"corpus": synth_corpus, "seed": 11, "sampler": {"size_min": 10, "size_max": 120}}
    a = [CIEnv(cfg) for _ in range(2)]
    seq = [[e.reset()[0]["graph_idx"][0] for _ in range(10)] for e in a]
    assert seq[0] == seq[1]


def test_unknown_config_key_rejected(synth_corpus):
    with pytest.raises(ValueError, match="unknown"):
        CIEnv({"corpus": synth_corpus, "mode": "inference"})


def test_latency_sum_at_a_cycle_boundary(tmp_path):
    """0.23+0.23+0.10+0.31+0.03+0.10 is 1.0000000000000002 in floats; the
    chain still takes one cycle (saving 5, speedup 7/2)."""
    from ci_ident.data.build import build_manifest

    ops = ["shl", "shl", "icmp", "add", "and", "icmp"]
    lines = ['digraph "c" {', '  "x" [label="load"];'] + [f'  "n{i}" [label="{o}"];' for i, o in enumerate(ops)]
    lines += ['  "x" -> "n0";'] + [f'  "n{i}" -> "n{i + 1}";' for i in range(5)] + ["}"]
    (tmp_path / "c.dot").write_text("\n".join(lines) + "\n")
    m = build_manifest([tmp_path / "c.dot"], tmp_path / "m.json")
    env = CIEnv({"corpus": str(m), "graphs": [0], "io": [2, 1], "area_factor": 1.0})
    obs, _ = env.reset()
    for i in range(1, 7):                      # grow the whole chain, then STOP
        assert obs["action_mask"][i]
        obs, *_ = env.step(i)
    if env.ci:
        env.step(env.max_nodes)
    assert env.episode_summary()["speedup"] == pytest.approx(3.5)
    assert env.episode_summary()["area_util"] == pytest.approx(1.0)   # the whole budget is usable
