"""The fast DOT reader returns exactly what networkx/pydot returns."""
from pathlib import Path

import networkx as nx
import pytest

from ci_ident.ddg.dotio import _Fallback, _fast, read_dot

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "dot"


def assert_same(path):
    ref = nx.drawing.nx_pydot.read_dot(str(path))
    g = _fast(Path(path).read_text())      # the fast path must apply to these files
    assert list(g.nodes()) == list(ref.nodes())
    for n in ref.nodes():
        assert g.nodes[n] == ref.nodes[n], n
    assert list(g.edges(keys=True)) == list(ref.edges(keys=True))


@pytest.mark.parametrize("path", sorted(BENCH.glob("*/*.dot")), ids=lambda p: p.name)
def test_benchmarks(path):
    assert_same(path)


def test_synthetic(synth_corpus):
    from ci_ident.data import Corpus
    c = Corpus.load(synth_corpus)
    for i in range(0, len(c), 4):
        assert_same(c.dot_path(i))


def test_fallback_cases(tmp_path):
    cases = {
        "chain": 'digraph g {\n "a" [label="add"];\n "b" [label="add"];\n "c" [label="add"];\n "a" -> "b" -> "c";\n}\n',
        "defaults": 'digraph g {\n node [shape=box];\n "a" [label="add"];\n}\n',
        "oneline": 'digraph g { "a" [label="add"]; "b" [label="mul"]; "a" -> "b"; }\n',
        "strict": 'strict digraph g {\n "a" -> "b";\n "a" -> "b";\n}\n',
        "keyword_case": 'digraph g {\n Node [shape=box];\n "a" [label="add"];\n}\n',
        "quoted_keyword": 'digraph g {\n "node" [label="add"];\n "a" [label="add"];\n}\n',
    }
    for name, text in cases.items():
        with pytest.raises(_Fallback):
            _fast(text)
        p = tmp_path / f"{name}.dot"
        p.write_text(text)
        ref = nx.drawing.nx_pydot.read_dot(str(p))
        g = read_dot(p)
        assert list(g.nodes()) == list(ref.nodes()) and list(g.edges()) == list(ref.edges())


def test_malformed_attribute_not_accepted():
    with pytest.raises(_Fallback):
        _fast('digraph g {\n "a" [label=x=y];\n}\n')


def test_escaped_quotes_and_multiedges(tmp_path):
    text = ('digraph "x" {\n  "n1" [label="  %1 = call i32 @f(ptr \\"s\\")", op="call"];\n'
            '  "n2" [label="add", idx=3];\n  "n1" -> "n2";\n  "n1" -> "n2";\n}\n')
    p = tmp_path / "e.dot"
    p.write_text(text)
    assert_same(p)
