"""Environment (TACO formulation): invariants checked independently of the
environment's own bookkeeping."""
import math

import numpy as np
import pytest

from _check import check_solution
from ci_ident.data import Corpus
from ci_ident.ddg.graph import area_reference, load_ddg
from ci_ident.ddg.signature import SignatureIndex
from ci_ident.env import CIEnv
from ci_ident.env.config import VARIANTS

SETTINGS = [((2, 1), 0.2), ((4, 2), 0.4), ((6, 3), 1.0)]


def _episode(env, rng, p_stop=0.15, p_any=0.0, gid=None):
    """Random policy over the mask (with probability p_any any action)."""
    obs, _ = env.reset(options=None if gid is None else {"graph_idx": gid})
    assert env.observation_space.contains(obs)
    gamma = env.config.gamma
    disc_shape, t, info = 0.0, 0, {}
    shape_prev = 0.0
    while True:
        m = obs["action_mask"]
        legal = np.flatnonzero(m[:-1])
        if rng.random() < p_any:
            a = int(rng.integers(0, env.max_nodes + 1))
        elif m[-1] and (legal.size == 0 or rng.random() < p_stop):
            a = env.max_nodes
        else:
            assert legal.size, "all-zero mask in a non-terminal state"
            a = int(rng.choice(legal))
        obs, r, term, trunc, info = env.step(a)
        f = env.stats["r_shape"] - shape_prev
        shape_prev = env.stats["r_shape"]
        disc_shape += (gamma ** t) * f
        t += 1
        if term or trunc:
            return info, disc_shape, term
        assert env.observation_space.contains(obs)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("mode", ["train", "inference"])
def test_random_episodes_are_valid(synth_corpus, variant, mode):
    corpus = Corpus.load(synth_corpus)
    rng = np.random.default_rng(hash((variant, mode)) & 0xFFFF)
    p_any = 0.2 if variant == "nomask" else 0.05
    for k, gi in enumerate(range(0, len(corpus), 3)):
        io, af = SETTINGS[k % len(SETTINGS)]
        env = CIEnv({"corpus": synth_corpus, "graphs": [gi], "variant": variant, "mode": mode,
                     "io": list(io), "area_factor": af, "gamma": 0.95})
        info, disc_shape, term = _episode(env, rng, p_any=p_any)
        g = load_ddg(corpus.dot_path(gi))
        sig = SignatureIndex.from_graph(g).signature
        # every committed CI (not only the reported ones) is valid and connected
        sp_all, _ = check_solution(g, env.committed, io[0], io[1], af * area_reference(g),
                                   signature=sig, connected=True)
        assert env.speedup() == pytest.approx(sp_all, rel=1e-12)
        sp, used = check_solution(g, env.committed_cis(), io[0], io[1], af * area_reference(g),
                                  signature=sig, connected=True)
        assert info["speedup"] == pytest.approx(sp, rel=1e-12)
        # commit payments telescope to w_perf * log S of the committed set
        w = env.config.weights.w_perf
        assert env.stats["r_commit"] == pytest.approx(w * math.log(env.speedup()), abs=1e-9)
        # the discounted shaping of an episode that terminates sums to zero
        if term:
            assert disc_shape == pytest.approx(0.0, abs=1e-9)
        if variant == "noshape":
            assert env.stats["r_shape"] == 0.0 and env.stats["r_size"] == 0.0


def test_masks(synth_corpus):
    base = {"corpus": synth_corpus, "graphs": [20], "io": [2, 1]}
    nm = CIEnv({**base, "variant": "nomask"})
    o, _ = nm.reset()
    n = nm.N
    assert o["action_mask"][:n].all() and o["action_mask"][-1] and not o["action_mask"][n:-1].any()
    tr = CIEnv({**base, "variant": "base", "mode": "train"})
    o, _ = tr.reset()
    assert not o["action_mask"][-1], "STOP is not offered on an empty candidate"
    inf = CIEnv({**base, "variant": "base", "mode": "inference"})
    o2, _ = inf.reset()
    # the inference mask is the training mask filtered by the growth predicate
    assert set(np.flatnonzero(o2["action_mask"][:-1])) <= set(np.flatnonzero(o["action_mask"][:-1]))


def test_inference_mask_never_needs_rollback(bench_manifest):
    rng = np.random.default_rng(5)
    for gi in range(8):
        env = CIEnv({"corpus": bench_manifest, "graphs": [gi], "variant": "absgain",
                     "mode": "inference", "io": [4, 2], "area_factor": 1.0})
        _episode(env, rng, p_stop=0.2)
        # with area factor 1.0 no area overflow is possible, so nothing is refused
        assert env.violations == 0


def test_illegal_actions_are_penalised_and_change_nothing(synth_corpus):
    env = CIEnv({"corpus": synth_corpus, "graphs": [12], "variant": "nomask"})
    o, _ = env.reset()
    status = o["node_status_vector"].copy()
    src = [i for i, v in enumerate(env.static.nodes) if v in env.static.input_nodes][0]
    o, r, term, trunc, _ = env.step(src)                  # a block input is never CI-legal
    assert r == pytest.approx(env.config.weights.w_invalid / env.N)
    assert np.array_equal(o["node_status_vector"], status) and not env.ci
    o, r, *_ = env.step(env.max_nodes)                    # empty STOP
    assert r == pytest.approx(env.config.weights.w_empty_stop / env.N)


def test_reward_terms_of_a_commit(tmp_path):
    """Chain x -> a(add) -> b(add) -> c(add): grow a, b, c and STOP."""
    from ci_ident.data.build import build_manifest

    (tmp_path / "c.dot").write_text('digraph "c" {\n "x" [label="load"];\n "a" [label="add"];\n'
                                   ' "b" [label="add"];\n "c" [label="add"];\n'
                                   ' "x" -> "a";\n "a" -> "b";\n "b" -> "c";\n}\n')
    m = build_manifest([tmp_path / "c.dot"], tmp_path / "m.json")
    env = CIEnv({"corpus": str(m), "graphs": [0], "variant": "absgain", "io": [1, 1],
                 "area_factor": 1.0, "gamma": 0.9})
    env.reset()
    W, N = env.config.weights, 4
    env.step(1)
    env.step(2)
    # the third growth exhausts the pool: the episode ends and the candidate
    # {a, b, c} is committed and paid in the same step
    _, r, term, *_ = env.step(3)
    assert term and env.committed == [["a", "b", "c"]]
    phi_ab = 2.0 - 1.0                                    # Delta({a,b}) = 2 - ceil(0.62)
    eta = 2.0 * (1.0 - 1.0)                               # h(3) = 1 (k_target = 2)
    expect = (W.w_ext * eta + W.w_perf * math.log(4 / 2) + (W.w_size / N) * 1.0
              + (W.w_shp / N) * (0.9 * 0.0 - phi_ab))
    assert r == pytest.approx(expect)


def test_unknown_keys_rejected(synth_corpus):
    with pytest.raises(ValueError, match="unknown"):
        CIEnv({"corpus": synth_corpus, "radius": 2})
    with pytest.raises(ValueError):
        CIEnv({"corpus": synth_corpus, "variant": "full"})


def test_training_mask_allows_temporary_port_overflow(bench_manifest):
    """Training: the regular port bound may be exceeded while growing (it is
    enforced at commit); inference: never."""
    seen = {"train": False, "inference": False}
    for mode in seen:
        rng = np.random.default_rng(11)
        for gi in range(8):
            env = CIEnv({"corpus": bench_manifest, "graphs": [gi], "variant": "base",
                         "mode": mode, "io": [2, 1], "area_factor": 1.0})
            obs, _ = env.reset()
            for _ in range(300):
                legal = np.flatnonzero(obs["action_mask"][:-1])
                a = int(rng.choice(legal)) if legal.size and rng.random() < 0.9 else env.max_nodes
                if not obs["action_mask"][a]:
                    break
                obs, _, term, trunc, _ = env.step(a)
                if len(env.ddg.input_set) > 2 or len(env.ddg.output_set) > 1:
                    seen[mode] = True
                if term or trunc:
                    break
    assert seen["train"] and not seen["inference"]


@pytest.mark.parametrize("variant", ["base", "noshape", "absgain", "nomask"])
def test_extension_rewards_of_discarded_candidates_cancel(bench_manifest, variant):
    """Growing a candidate and discarding it earns no extension reward (in
    discounted value); only committed candidates keep theirs."""
    rng = np.random.default_rng(2)
    env = CIEnv({"corpus": bench_manifest, "graphs": [4], "variant": variant,
                 "mode": "train", "io": [2, 1], "area_factor": 0.4, "gamma": 0.9})
    obs, _ = env.reset()
    gamma, t, cand, discards = 0.9, 0, 0.0, 0
    ext_prev, commits_prev = 0.0, 0
    for _ in range(400):
        legal = [i for i in np.flatnonzero(obs["action_mask"][:-1])
                 if env._selectable(env.static.nodes[i])]       # NoMask: legal growth only
        grow = len(legal) and (not env.ci or rng.random() < 0.7)
        a = int(rng.choice(legal)) if grow else env.max_nodes
        had = bool(env.ci)
        obs, r, term, trunc, _ = env.step(a)
        d = (env.stats["r_ext"] - ext_prev) * gamma ** t
        ext_prev = env.stats["r_ext"]
        t += 1
        if env.stats["commits"] > commits_prev:
            commits_prev, cand = env.stats["commits"], 0.0
        elif had and not env.ci:
            assert cand + d == pytest.approx(0.0, abs=1e-9)
            cand, discards = 0.0, discards + 1
        else:
            cand += d
        if term or trunc:
            break
    assert discards > 0
