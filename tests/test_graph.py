import numpy as np
import pytest

from ci_ident.data import Corpus
from ci_ident.ddg.graph import load_ddg, node_order, static_node_features
from ci_ident.ddg.signature import SignatureIndex
from ci_ident.ddg.state import StaticDDG
from ci_ident.model.graph_store import GraphStore, build_dgl_graph


def _write(tmp_path, name, nodes, edges):
    lines = [f'digraph "{name}" {{'] + [f'  "{n}" [label="{op}"];' for n, op in nodes]
    lines += [f'  "{u}" -> "{v}";' for u, v in edges] + ["}"]
    p = tmp_path / f"{name}.dot"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_node_order_is_dot_order_not_sorted(tmp_path):
    # string labels: sorted order would put "10" before "2"
    nodes = [(str(i), "load" if i == 0 else "add") for i in range(12)]
    edges = [(str(i), str(i + 1)) for i in range(11)]
    g = load_ddg(_write(tmp_path, "chain", nodes, edges))
    assert node_order(g) == [str(i) for i in range(12)]


def test_store_matches_environment_order(synth_corpus):
    corpus = Corpus.load(synth_corpus)
    store = GraphStore(corpus)
    for gi in range(0, len(corpus), 5):
        dg = store.graph(gi)
        g = load_ddg(corpus.dot_path(gi))
        st = StaticDDG.from_graph(g)
        order = list(st.nodes)
        assert order == node_order(g)
        feats = dg.ndata["feat"].numpy()
        exp = np.asarray(static_node_features(g), dtype=np.float32)
        assert np.array_equal(feats, exp)
        src, dst = dg.edges()
        got = sorted(zip(src.tolist(), dst.tolist()))
        pos = {n: i for i, n in enumerate(order)}
        assert got == sorted((pos[u], pos[v]) for u, v in g.edges())
        # inadmissible nodes (and only those) have zero static features
        for i, n in enumerate(order):
            assert (feats[i] == 0).all() == (not st.eligible[n])


def test_store_rejects_reordered_file(tmp_path, synth_corpus):
    corpus = Corpus.load(synth_corpus)
    e = corpus.entries[3]
    text = e.path.read_text().splitlines()
    node_lines = [l for l in text if "label=" in l]
    other = [l for l in text if l not in node_lines]
    shuffled = [other[0]] + node_lines[::-1] + other[1:]
    p = tmp_path / "rev.dot"
    p.write_text("\n".join(shuffled) + "\n")
    with pytest.raises(ValueError, match="node order"):
        build_dgl_graph(p, expected_digest=e.node_order, expected_nodes=e.num_nodes)


def test_cyclic_graph_rejected(tmp_path):
    p = _write(tmp_path, "cyc", [("a", "add"), ("b", "add")], [("a", "b"), ("b", "a")])
    with pytest.raises(ValueError, match="cycle"):
        load_ddg(p)


def test_signature_is_isomorphism_invariant(tmp_path):
    g1 = load_ddg(_write(tmp_path, "g1", [("x", "load"), ("a", "add"), ("b", "mul"), ("c", "add")],
                         [("x", "a"), ("a", "b"), ("b", "c")]))
    g2 = load_ddg(_write(tmp_path, "g2", [("q", "load"), ("p", "add"), ("r", "mul"), ("s", "add")],
                         [("q", "p"), ("p", "r"), ("r", "s")]))
    s1 = SignatureIndex.from_graph(g1).signature(["a", "b"])
    s2 = SignatureIndex.from_graph(g2).signature(["p", "r"])
    assert s1 == s2
    assert SignatureIndex.from_graph(g1).signature(["b", "c"]) != s1   # mul->add vs add->mul


def test_signature_is_invariant_to_node_order(bench_manifest):
    """Same node set given in different orders -> same signature."""
    import random

    from ci_ident.data import Corpus
    from ci_ident.env import CIEnv

    corpus = Corpus.load(bench_manifest)
    rng = random.Random(0)
    for gi in range(len(corpus)):
        g = load_ddg(corpus.dot_path(gi))
        idx = SignatureIndex.from_graph(g)
        nodes = [n for n in g.nodes() if g.nodes[n]["ci_eligible"]]
        for _ in range(30):                       # random connected sets
            seed = rng.choice(nodes)
            ci, frontier = [seed], set(g.predecessors(seed)) | set(g.successors(seed))
            for _ in range(rng.randint(2, 8)):
                cand = [v for v in frontier if v not in ci and g.nodes[v]["ci_eligible"]]
                if not cand:
                    break
                v = rng.choice(cand)
                ci.append(v)
                frontier |= set(g.predecessors(v)) | set(g.successors(v))
            sigs = {idx.signature(rng.sample(ci, len(ci))) for _ in range(6)}
            assert len(sigs) == 1, (gi, ci)
