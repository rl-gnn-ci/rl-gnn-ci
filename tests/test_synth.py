import filecmp

import networkx as nx

from ci_ident.data import Corpus
from ci_ident.data.synth import SOURCE_LABEL, SynthConfig, generate_corpus
from ci_ident.ddg.graph import load_ddg


def test_reproducible_and_well_formed(tmp_path):
    cfg = SynthConfig(sizes=(10, 50, 200), per_size=2, seed=5)
    a = generate_corpus(tmp_path / "a", cfg, verbose=False)
    b = generate_corpus(tmp_path / "b", cfg, verbose=False)
    assert all(filecmp.cmp(x, y, shallow=False) for x, y in zip(a, b))
    c = Corpus.load(tmp_path / "a" / "manifest.json")
    assert sorted(c.sizes.tolist()) == [10, 10, 50, 50, 200, 200]
    for p in a:
        g = load_ddg(p)
        assert nx.is_directed_acyclic_graph(g)
        assert max(d for _, d in g.in_degree()) <= 2
        for v in g.nodes():
            assert (g.in_degree(v) == 0) == (g.nodes[v]["op"] == SOURCE_LABEL)


def test_graph_independent_of_corpus_extent(tmp_path):
    a = generate_corpus(tmp_path / "a", SynthConfig(sizes=(30,), per_size=3), manifest=None, verbose=False)
    b = generate_corpus(tmp_path / "b", SynthConfig(sizes=(10, 30), per_size=1), manifest=None, verbose=False)
    assert filecmp.cmp(a[0], b[1], shallow=False)
