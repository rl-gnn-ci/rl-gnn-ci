"""Tests of the ILP baseline (ci_ident.baselines.ilp).

The checks in validate.py are tested on their own; the MILP tests are skipped
when neither CPLEX (docplex) nor SCIP (pyscipopt) is importable.
"""

import random

import pytest

from ci_ident.baselines.ilp import validate
from ci_ident.costs import cycles
from ci_ident.ddg.graph import load_ddg

try:
    from ci_ident.baselines.ilp.enumerator import ILPEnumerator, available_solvers

    HAVE_MILP = bool(available_solvers())
except Exception:  # pragma: no cover
    HAVE_MILP = False

needs_milp = pytest.mark.skipif(not HAVE_MILP, reason="no MILP backend (docplex or pyscipopt)")


def _write(tmp_path, name, nodes, edges):
    lines = [f'digraph "{name}" {{'] + [f'  "{n}" [label="{op}"];' for n, op in nodes]
    lines += [f'  "{u}" -> "{v}";' for u, v in edges] + ["}"]
    p = tmp_path / f"{name}.dot"
    p.write_text("\n".join(lines) + "\n")
    return p


def _contraction_graph(tmp_path):
    """After {x, y, x2} is extracted, its two
    outputs x and x2 feed {z, z2, z3, z4}; on the contracted graph they are one
    input (plus s2), on the original graph three inputs."""
    nodes = [("s0", "load"), ("s1", "load"), ("s2", "load"), ("x", "add"), ("y", "add"),
             ("x2", "add"), ("z", "xor"), ("z2", "xor"), ("z3", "xor"), ("z4", "xor")]
    edges = [("s0", "x"), ("s1", "x"), ("x", "y"), ("s1", "y"), ("y", "x2"), ("x", "x2"),
             ("x", "z"), ("s2", "z"), ("x2", "z2"), ("z", "z2"), ("z2", "z3"), ("z", "z3"),
             ("z3", "z4"), ("z2", "z4")]
    return load_ddg(_write(tmp_path, "contract2", nodes, edges))


def _pair_graph(tmp_path):
    """Two convex, connected 'join' CIs A={a,b,ja}, B={c,d,jb} that depend on
    each other (a -> c and d -> b): contracting both gives a cycle."""
    nodes = [("s0", "load"), ("s1", "load"), ("s2", "load"), ("s3", "load"),
             ("a", "add"), ("b", "add"), ("ja", "add"), ("c", "add"), ("d", "add"), ("jb", "add")]
    edges = [("s0", "a"), ("s1", "b"), ("a", "ja"), ("b", "ja"),
             ("s2", "c"), ("s3", "d"), ("c", "jb"), ("d", "jb"),
             ("a", "c"), ("d", "b")]
    return load_ddg(_write(tmp_path, "pair", nodes, edges))


def _ring_graph(tmp_path):
    """Three join CIs with a_k -> b_{k+1}: pairwise schedulable, but the set of
    all three is cyclic once contracted."""
    nodes, edges = [], []
    for k in range(3):
        nodes += [(f"s{k}", "load"), (f"t{k}", "load"), (f"a{k}", "add"), (f"b{k}", "add"),
                  (f"j{k}", "add")]
        edges += [(f"s{k}", f"a{k}"), (f"t{k}", f"b{k}"), (f"a{k}", f"j{k}"), (f"b{k}", f"j{k}")]
    for k in range(3):
        edges.append((f"a{k}", f"b{(k + 1) % 3}"))
    return load_ddg(_write(tmp_path, "ring", nodes, edges))


def _random_ddg(tmp_path, name, n, rng, p_invalid=0.05):
    nodes, edges = [], []
    for i in range(n):
        preds = []
        if i > 0 and rng.random() > 0.18:
            k = 2 if rng.random() < 0.55 else 1
            preds = rng.sample(range(max(0, i - 8), i), min(k, i - max(0, i - 8)))
        edges += [(str(p), str(i)) for p in preds]
        if not preds:
            op = "load"
        elif rng.random() < p_invalid:
            op = rng.choice(["store", "call", "phi", "udiv"])
        else:
            op = rng.choice(["add", "add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "icmp"])
        nodes.append((str(i), op))
    return load_ddg(_write(tmp_path, name, nodes, edges))


# ------------------------------------------------------------- validate.py


def test_io_semantics_and_convexity(tmp_path):
    g = _contraction_graph(tmp_path)
    # inputs = distinct outside predecessors; a selected sink is an output
    assert validate.io_counts(g, ["z", "z2", "z3", "z4"]) == (3, 1)
    assert validate.io_sets(g, ["x", "y", "x2"]) == ({"s0", "s1"}, {"x", "x2"})
    assert validate.is_convex(g, ["x", "x2"]) is False  # x -> y -> x2 leaves the set
    assert validate.is_convex(g, ["x", "y", "x2"]) is True
    assert validate.is_weakly_connected(g, ["x", "z4"]) is False
    assert validate.is_admissible(g, ["x", "s0"]) is False
    assert validate.check_ci(g, ["z", "z2", "z3", "z4"], 2, 2) == ["inputs"]
    # saving = |S| - ceil(cp): add 0.31 x 3 in a chain -> ceil(0.93) = 1
    assert validate.saving(g, ["x", "y", "x2"]) == 2


def test_set_acyclicity_checks(tmp_path):
    g = _pair_graph(tmp_path)
    A, B = ["a", "b", "ja"], ["c", "d", "jb"]
    for S in (A, B):  # In = {s0, s1, d} / {s2, s3, a}; Out = {a, ja} / {d, jb}
        assert validate.check_ci(g, S, 3, 2) == []
    assert validate.mutually_dependent_pairs(g, [A, B]) == [(0, 1)]
    assert validate.set_is_acyclic(g, [A]) and not validate.set_is_acyclic(g, [A, B])
    r = _ring_graph(tmp_path)
    ring = [[f"a{k}", f"b{k}", f"j{k}"] for k in range(3)]
    assert validate.mutually_dependent_pairs(r, ring) == []
    assert sorted(validate.find_contracted_cycle(r, ring)) == [0, 1, 2]
    assert validate.set_is_acyclic(r, ring[:2])


# ------------------------------------------------------------- MILP


@needs_milp
def test_inputs_counted_on_the_original_ddg_after_contraction(tmp_path):
    """With one input binary per super-node the model returns
    {z, z2, z3, z4} (3 inputs on the original DDG) at (2, 2); the enumerator
    must not emit it, and must record why (lazy split)."""
    g = _contraction_graph(tmp_path)
    en = ILPEnumerator(g, 2, 2, area_max=None, area_cap_fracs=[1.0], lambda_grid=[0.0])
    out = en.run()
    assert ["x", "x2", "y"] in out["templates"]
    for T in out["templates"]:
        assert validate.check_ci(g, T, 2, 2) == [], T
    assert sorted(["z", "z2", "z3", "z4"]) not in [sorted(T) for T in out["templates"]]
    assert out["lazy_splits"] >= 1 and out["rejected"] == []
    # the first solve of round 1 (one input binary per super-node) returns the 3-input set
    r1 = [d for d in out["diagnostics"] if d["contraction_round"] == 1]
    assert sorted(r1[0]["candidate"]) == ["z", "z2", "z3", "z4"] and r1[0]["split_supernodes"] == 0
    assert r1[-1]["split_supernodes"] == 1 and sorted(r1[-1]["candidate"]) != ["z", "z2", "z3", "z4"]


@needs_milp
def test_selector_rejects_mutually_dependent_pair(tmp_path):
    """Both CIs fit the budget and are the same pattern; picking both
    would be optimal without the conflict constraint."""
    from ci_ident.baselines.ilp.selector import ILPSelector

    g = _pair_graph(tmp_path)
    T = [["a", "b", "ja"], ["c", "d", "jb"]]
    sel = ILPSelector(g, T, [validate.saving(g, t) for t in T], [validate.ci_area(g, t) for t in T],
                      area_max=10.0)
    r = sel.run()
    assert len(r["classes"]) == 1 and sel.pair_conflicts == [(0, 1)]
    assert len(r["chosen_instances"]) == 1


@needs_milp
def test_selector_lazy_cycle_cut(tmp_path):
    from ci_ident.baselines.ilp.selector import ILPSelector

    g = _ring_graph(tmp_path)
    T = [[f"a{k}", f"b{k}", f"j{k}"] for k in range(3)]
    sel = ILPSelector(g, T, [validate.saving(g, t) for t in T], [validate.ci_area(g, t) for t in T],
                      area_max=10.0)
    r = sel.run()
    assert sel.pair_conflicts == [] and len(sel.cycle_cuts) == 1
    chosen = [T[i] for i in r["chosen_instances"]]
    assert len(chosen) == 2 and validate.set_is_acyclic(g, chosen)


@needs_milp
@pytest.mark.parametrize("builder", [_pair_graph, _ring_graph, _contraction_graph])
def test_identify_ilp_end_to_end_small(tmp_path, builder):
    from ci_ident.baselines.ilp import identify_ilp

    g = builder(tmp_path)
    n_nodes, n_edges = g.number_of_nodes(), g.number_of_edges()
    r = identify_ilp(g, 3, 2, 1.0, time_budget_s=60)
    assert g.number_of_nodes() == n_nodes and g.number_of_edges() == n_edges  # not mutated
    assert r["status"] == "ok"
    chk = validate.check_selection(g, r["instances"], 3, 2)
    assert chk["ok"], chk
    assert r["num_instances"] >= 1


def _recomputed_speedup(g, instances):
    T_sw = g.number_of_nodes()
    saved = sum(len(S) - cycles(validate.critical_path(g, S)) for S in instances)
    return T_sw / (T_sw - saved)


@needs_milp
@pytest.mark.parametrize("seed", range(6))
def test_identify_ilp_random_valid_and_consistent(tmp_path, seed):
    """Every returned instance passes validate.py on the original graph and
    the selection is acyclic; the speedup recomputes from the instances."""
    from ci_ident.baselines.ilp import identify_ilp

    rng = random.Random(seed)
    g = _random_ddg(tmp_path, f"r{seed}", rng.randint(12, 26), rng)
    max_in, max_out = rng.choice([(2, 1), (2, 2), (3, 2), (4, 2)])
    af = rng.choice([0.25, 0.5, 1.0])
    r = identify_ilp(g, max_in, max_out, af, time_budget_s=60)
    assert r["status"] == "ok"
    for S in r["instances"]:
        assert validate.check_ci(g, S, max_in, max_out) == [], S
    assert validate.check_selection(g, r["instances"], max_in, max_out)["ok"]
    assert r["speedup"] == pytest.approx(_recomputed_speedup(g, r["instances"]), abs=1e-12)
    assert r["area_used"] <= r["area_budget"] + 1e-9
    assert r["a_ref"] == pytest.approx(sum(d["area"] for _, d in g.nodes(data=True) if d["ci_eligible"]))
    assert len(r["pattern_of_instance"]) == r["num_instances"]
    assert set(r["pattern_of_instance"]) == set(range(r["num_patterns"]))
    assert r["rejected"]["count"] == 0
    assert all(d["phase"] in ("enum", "select") for d in r["diagnostics"])


@needs_milp
def test_zero_budget_returns_no_solution(tmp_path):
    from ci_ident.baselines.ilp import identify_ilp

    g = _random_ddg(tmp_path, "b0", 30, random.Random(7))
    r = identify_ilp(g, 4, 2, 1.0, time_budget_s=0.0)
    assert r["status"] == "no_solution" and r["instances"] == [] and r["speedup"] == 1.0
    assert r["runtime_s"] < 5.0 and r["diagnostics"] == []


@needs_milp
def test_short_budget_returns_valid_partial_result(tmp_path):
    """A positive but too short budget: every solve respects the one deadline,
    the status says so, and whatever is returned is still valid."""
    from ci_ident.baselines.ilp import identify_ilp

    g = _random_ddg(tmp_path, "t0", 90, random.Random(3))
    budget = 1.5
    r = identify_ilp(g, 4, 2, 1.0, time_budget_s=budget)
    assert r["status"] in ("timeout_partial", "no_solution")
    assert r["runtime_s"] < budget + 2.0
    assert any(d["hit_time_limit"] for d in r["diagnostics"]) or r["enum_truncated"]
    for d in r["diagnostics"]:
        assert d["time_limit_s"] is not None and d["time_limit_s"] <= budget + 1e-6
    assert validate.check_selection(g, r["instances"], 4, 2)["ok"]
    assert r["speedup"] == pytest.approx(_recomputed_speedup(g, r["instances"]), abs=1e-12)


def _boundary_chain(tmp_path):
    """shl, shl, icmp, add, and, icmp: the float sum of the hardware latencies
    is 1.0000000000000002, i.e. exactly one cycle."""
    ops = ["shl", "shl", "icmp", "add", "and", "icmp"]
    nodes = [("s", "load")] + [(f"n{k}", op) for k, op in enumerate(ops)]
    edges = [("s", "n0")] + [(f"n{k}", f"n{k + 1}") for k in range(len(ops) - 1)]
    return load_ddg(_write(tmp_path, "boundary", nodes, edges))


def test_saving_uses_cycle_rounding(tmp_path):
    g = _boundary_chain(tmp_path)
    S = [f"n{k}" for k in range(6)]
    assert validate.critical_path(g, S) > 1.0  # float rounding
    assert validate.saving(g, S) == 5  # |S| - cycles(cp) = 6 - 1


@needs_milp
def test_identify_ilp_cycle_boundary(tmp_path):
    from ci_ident.baselines.ilp import identify_ilp

    g = _boundary_chain(tmp_path)
    r = identify_ilp(g, 2, 2, 1.0, time_budget_s=60)
    assert r["status"] == "ok"
    assert sorted(map(sorted, r["instances"])) == [[f"n{k}" for k in range(6)]]
    assert r["speedup"] == pytest.approx(7 / 2)


@needs_milp
def test_selector_area_budget_uses_eps(tmp_path):
    """The solver's feasibility tolerance (1e-6) admits a pattern that exceeds
    the budget by 5e-7; the selector must exclude it (lazy cut), while an
    excess within ci_ident.costs.EPS is accepted, as in the RL environment."""
    from ci_ident.baselines.ilp.selector import ILPSelector

    g = _pair_graph(tmp_path)
    T = [["a", "b", "ja"]]
    A = validate.ci_area(g, T[0])
    over = ILPSelector(g, T, [validate.saving(g, T[0])], [A], area_max=A - 5e-7)
    assert over.run()["chosen_instances"] == []
    at = ILPSelector(g, T, [validate.saving(g, T[0])], [A], area_max=A - 1e-10)
    assert at.run()["chosen_instances"] == [0] and at.area_cuts == []
