"""SARSA baseline (ci_ident.baselines.sarsa): enumeration, conflicts, contract, deadline.

Every selection is validated with the independent checks of ``_check`` on the
original DDG.  Tests that need the C++ extensions skip when they are not
built (``python -m ci_ident.baselines.sarsa.build``).
"""
import math
import random
import time

import networkx as nx
import pytest

from _check import check_solution, contraction_is_dag
from ci_ident.baselines import sarsa
from ci_ident.ddg.graph import annotate, area_reference, load_ddg
from ci_ident.ddg.signature import SignatureIndex

needs_ext = pytest.mark.skipif(not sarsa.available(), reason="SARSA C++ extensions not built")

RESULT_KEYS = {"speedup", "area_used", "area_budget", "a_ref", "num_patterns", "num_instances",
               "instances", "pattern_of_instance", "mean_ci_size", "runtime_s", "status",
               "num_candidates", "diagnostics"}

VAL = ["add", "xor", "shl", "mul", "and", "sub", "or", "icmp"]
INV = ["load", "store", "call", "phi", "select"]


# ----------------------------------------------------------------------------- helpers

def make_ddg(edges, labels):
    g = nx.MultiDiGraph()
    for v in labels:
        g.add_node(v, label=labels[v])
    for u, v in edges:
        g.add_edge(u, v)
    annotate(g)
    return g


def rand_ddg(rng, n, pinv):
    """Random DAG: several sources and sinks, inadmissible interior nodes, parallel edges."""
    g = nx.MultiDiGraph()
    nsrc = rng.randint(1, 3)
    for i in range(n):
        g.add_node(f"n{i}")
    for i in range(nsrc, n):
        k = min(i, 1 if rng.random() < 0.45 else (2 if rng.random() < 0.8 else 3))
        for p in rng.sample(range(i), k):
            g.add_edge(f"n{p}", f"n{i}")
            if rng.random() < 0.05:
                g.add_edge(f"n{p}", f"n{i}")
    for v in g.nodes:
        if g.in_degree(v) == 0:
            g.nodes[v]["label"] = "load"
        else:
            g.nodes[v]["label"] = rng.choice(INV) if rng.random() < pinv else rng.choice(VAL)
    annotate(g)
    return g


def brute_force(g, connected):
    """Every legal, convex (connected) subset with its (|In|, |Out|), by bitmask enumeration."""
    nodes = list(g.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    N = len(nodes)
    pm, sm = [0] * N, [0] * N
    for u, v in g.edges():
        sm[idx[u]] |= 1 << idx[v]
        pm[idx[v]] |= 1 << idx[u]
    order = [idx[v] for v in nx.topological_sort(g)]
    desc, anc = [0] * N, [0] * N
    for u in reversed(order):
        for j in range(N):
            if sm[u] >> j & 1:
                desc[u] |= (1 << j) | desc[j]
    for u in order:
        for j in range(N):
            if pm[u] >> j & 1:
                anc[u] |= (1 << j) | anc[j]
    elig = [i for i, v in enumerate(nodes) if g.nodes[v]["ci_eligible"]]
    out = {}
    for code in range(1, 1 << len(elig)):
        mem = [elig[j] for j in range(len(elig)) if code >> j & 1]
        S = sum(1 << i for i in mem)
        D = A = I = 0
        for i in mem:
            D |= desc[i]
            A |= anc[i]
            I |= pm[i]
        if D & A & ~S:
            continue
        if connected:
            seen = fr = 1 << mem[0]
            while fr:
                nf = 0
                for i in range(N):
                    if fr >> i & 1:
                        nf |= pm[i] | sm[i]
                nf &= S & ~seen
                seen |= nf
                fr = nf
            if seen != S:
                continue
        nout = sum(1 for i in mem if sm[i] == 0 or sm[i] & ~S)
        out[frozenset(nodes[i] for i in mem)] = (bin(I & ~S).count("1"), nout)
    return out


def validate(g, r, I, O):
    sig = SignatureIndex.from_graph(g).signature
    sp, used = check_solution(g, r["instances"], I, O, r["area_budget"], signature=sig, connected=True)
    assert math.isclose(sp, r["speedup"], rel_tol=0, abs_tol=1e-12)
    assert abs(used - r["area_used"]) < 1e-9
    # pattern bookkeeping: equal signature <=> equal pattern index
    sigs = [sig(S) for S in r["instances"]]
    for a in range(len(sigs)):
        for b in range(len(sigs)):
            assert (sigs[a] == sigs[b]) == (r["pattern_of_instance"][a] == r["pattern_of_instance"][b])
    assert r["num_patterns"] == len(set(sigs))
    return sp


# ----------------------------------------------------------------------------- import

def test_package_imports_without_extensions():
    assert isinstance(sarsa.available(), bool)
    assert callable(sarsa.identify_sarsa)
    if not sarsa.available():
        g = make_ddg([("s", "a"), ("a", "b")], {"s": "load", "a": "add", "b": "add"})
        with pytest.raises(sarsa.ExtensionUnavailable):
            sarsa.identify_sarsa(g, 2, 1, 1.0)


# ----------------------------------------------------------------------------- enumeration

@needs_ext
@pytest.mark.parametrize("connected,ngraphs,seed", [(True, 300, 0), (False, 150, 1)])
def test_enumeration_matches_brute_force(connected, ngraphs, seed):
    from ci_ident.baselines.sarsa.enumeration import candidate_sets, enumerate_candidates

    rng = random.Random(seed)
    bounds = [(1, 1), (2, 1), (3, 1), (3, 2), (4, 2), (6, 3), (12, 6)]
    n_ref = n_inadm = 0
    for _ in range(ngraphs):
        g = rand_ddg(rng, rng.randint(8, 16), rng.choice([0.0, 0.15, 0.3]))
        n_inadm += any(not d["ci_eligible"] and g.in_degree(v) for v, d in g.nodes(data=True))
        table = brute_force(g, connected)
        for I, O in bounds:
            r = enumerate_candidates(g, I, O, connected=connected, store_all=True)
            got = [frozenset(S) for S in candidate_sets(r)]
            ref = {S for S, (a, b) in table.items() if a <= I and b <= O}
            assert len(got) == len(set(got)), "candidate generated twice"
            assert set(got) == ref, (I, O, sorted(map(sorted, ref - set(got)))[:3],
                                     sorted(map(sorted, set(got) - ref))[:3])
            assert r["valid"] == len(got) and not r["timed_out"]
            n_ref += len(ref)
    assert n_ref > 1000 and n_inadm > ngraphs // 3


@needs_ext
def test_enumeration_deadline_expired():
    from ci_ident.baselines.sarsa.enumeration import enumerate_candidates

    g = rand_ddg(random.Random(3), 16, 0.1)
    for s in (0.0, -5.0):
        r = enumerate_candidates(g, 4, 2, seconds=s)
        assert r["timed_out"] and r["valid"] == 0


# ----------------------------------------------------------------------------- conflicts

def _pair_case(through_load, pad):
    """A = {c, a1, a2}, B = {d, b1, b2}; A -> B via a1 -> b1 (optionally through a
    load L), B -> A via b2 -> a2.  Both are convex, so only the dependency
    relation can separate them.  ``pad`` single-node branches of ``s0`` sit
    between A and B in DOT order, so B comes ``pad`` positions after A in the
    canonical topological order."""
    E = [("s0", "c"), ("c", "a1"), ("c", "a2")]
    L = {"s0": "load", "c": "add", "a1": "xor", "a2": "and"}
    for i in range(pad):
        E.append(("s0", f"p{i}"))
        L[f"p{i}"] = "add"
    E += [("s1", "d"), ("d", "b1"), ("d", "b2"), ("b2", "a2")]
    L.update({"s1": "load", "d": "add", "b1": "xor", "b2": "and"})
    if through_load:
        E += [("a1", "L"), ("L", "b1")]
        L["L"] = "load"
    else:
        E += [("a1", "b1")]
    return make_ddg(E, L), [["c", "a1", "a2"], ["d", "b1", "b2"]]


def _triangle_case():
    """X, Y, Z = {r, i, o} with r -> i, r -> o; o_X -> i_Y, o_Y -> i_Z, o_Z -> i_X.
    No two instances are mutually dependent, but the three form a cycle once
    contracted."""
    E, L = [], {}
    for k in "XYZ":
        E += [(f"s{k}", f"r{k}"), (f"r{k}", f"i{k}"), (f"r{k}", f"o{k}")]
        L.update({f"s{k}": "load", f"r{k}": "add", f"i{k}": "xor", f"o{k}": "xor"})
    E += [("oX", "iY"), ("oY", "iZ"), ("oZ", "iX")]
    return make_ddg(E, L), [[f"r{k}", f"i{k}", f"o{k}"] for k in "XYZ"]


@needs_ext
@pytest.mark.parametrize("case", ["adjacent", "far_in_topo_order", "through_load", "triangle"])
def test_selection_never_returns_cyclic_set(case):
    from ci_ident.baselines.sarsa import selection

    if case == "triangle":
        g, inst = _triangle_case()
        for a in range(3):
            for b in range(a + 1, 3):
                assert contraction_is_dag(g, [inst[a], inst[b]])  # pairwise fine
        assert not contraction_is_dag(g, inst)                    # jointly cyclic
    else:
        g, inst = _pair_case(case == "through_load", 300 if case == "far_in_topo_order" else 0)
        assert not contraction_is_dag(g, inst)
        if case == "far_in_topo_order":
            from ci_ident.baselines.sarsa.enumeration import topo_order

            pos = {v: i for i, v in enumerate(topo_order(g))}
            # the first nodes of the two instances are more than 256 positions apart in topological order
            assert min(pos[v] for v in inst[1]) - min(pos[v] for v in inst[0]) > 256
    budget = 10.0
    for seed in range(5):
        r = selection.select(g, inst, [2.0] * len(inst), list(range(len(inst))), [0.1] * len(inst),
                             budget, episodes=300, seed=seed)
        chosen = [inst[c] for c in r["chosen"]]
        assert contraction_is_dag(g, chosen)
        assert len(chosen) == len(inst) - 1  # all but one are compatible
        assert r["dependency_rejections"] > 0


@needs_ext
@pytest.mark.parametrize("case", ["far_in_topo_order", "through_load", "triangle"])
def test_end_to_end_on_conflict_graphs(case):
    g = _triangle_case()[0] if case == "triangle" else _pair_case(case == "through_load", 300)[0]
    for I, O in [(2, 2), (4, 2), (6, 3)]:
        r = sarsa.identify_sarsa(g, I, O, 1.0, episodes=300, time_budget_s=60)
        validate(g, r, I, O)


@needs_ext
def test_random_graphs_end_to_end():
    rng = random.Random(7)
    rejections = 0
    for t in range(40):
        g = rand_ddg(rng, rng.randint(30, 70), 0.2)
        I, O = rng.choice([(2, 1), (4, 2), (6, 3)])
        af = rng.choice([0.2, 0.6, 1.0])
        before = {v: dict(d) for v, d in g.nodes(data=True)}
        r = sarsa.identify_sarsa(g, I, O, af, episodes=150, seed=t, time_budget_s=60)
        assert {v: dict(d) for v, d in g.nodes(data=True)} == before  # g is read only
        assert set(r) == RESULT_KEYS and r["status"] == "ok"
        validate(g, r, I, O)
        assert r["a_ref"] == area_reference(g) and r["area_budget"] == pytest.approx(af * r["a_ref"])
        rejections += r["diagnostics"]["sarsa"]["dependency_rejections"]
    assert rejections > 0  # the dependency relation was exercised


# ----------------------------------------------------------------------------- contract

@needs_ext
def test_determinism_and_seed():
    g = rand_ddg(random.Random(11), 80, 0.15)
    a = sarsa.identify_sarsa(g, 4, 2, 0.6, episodes=400, seed=3)
    b = sarsa.identify_sarsa(g, 4, 2, 0.6, episodes=400, seed=3)
    for k in RESULT_KEYS - {"runtime_s", "diagnostics"}:
        assert a[k] == b[k], k
    assert a["diagnostics"]["sarsa"]["states"] == b["diagnostics"]["sarsa"]["states"]


@needs_ext
def test_deadline_respected(bench_manifest):
    import json
    from pathlib import Path

    root = Path(bench_manifest).parent
    path = next(e["path"] for e in json.load(open(bench_manifest))["graphs"] if e["path"].endswith("sha.dot"))
    g = load_ddg(root / path)
    for budget in (0.0, -1.0):
        t = time.monotonic()
        r = sarsa.identify_sarsa(g, 12, 6, 1.0, time_budget_s=budget)
        assert r["status"] == "no_solution" and r["instances"] == [] and r["speedup"] == 1.0
        assert time.monotonic() - t < 5.0
    for budget in (0.05, 0.5, 2.0):
        t = time.monotonic()
        r = sarsa.identify_sarsa(g, 12, 6, 1.0, time_budget_s=budget)
        el = time.monotonic() - t
        assert r["status"] in ("timeout_partial", "no_solution")
        assert el < budget + 3.0, el  # graph indexing and final validation are outside the stages
        validate(g, r, 12, 6)


@needs_ext
def test_benchmark_smoke(bench_manifest):
    import json
    from pathlib import Path

    root = Path(bench_manifest).parent
    path = next(e["path"] for e in json.load(open(bench_manifest))["graphs"]
                if e["path"].endswith("computesatd.dot"))
    g = load_ddg(root / path)
    r = sarsa.identify_sarsa(g, 4, 2, 0.6, episodes=500, seed=0, time_budget_s=120)
    assert r["status"] == "ok" and r["num_candidates"] == 444 and r["num_instances"] > 0
    assert validate(g, r, 4, 2) > 1.0


@needs_ext
def test_enumeration_candidate_bound_stops_the_search(bench_manifest):
    from pathlib import Path

    from ci_ident.baselines.sarsa.enumeration import enumerate_candidates
    g = load_ddg(Path(bench_manifest).parent / "dot/mibench/susan.dot")
    full = enumerate_candidates(g, 4, 2)
    assert not full["capped"] and full["stored"] > 10
    r = enumerate_candidates(g, 4, 2, max_candidates=10)
    assert r["capped"] and not r["timed_out"] and r["stored"] == 10
    res = sarsa.identify_sarsa(g, 4, 2, 0.6, time_budget_s=60, episodes=50, max_candidates=10)
    assert res["status"] in ("timeout_partial", "no_solution")
    assert res["diagnostics"]["enumeration"]["capped"]
