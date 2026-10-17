"""Reading DDGs from DOT files.

:func:`read_dot` returns the same ``networkx.MultiDiGraph`` as
``networkx.drawing.nx_pydot.read_dot``: the same node order, the same
node-attribute strings (quoted values keep their quotes), and the same edges in
the same order.  For the line-oriented files written by the DDG pass and the
synthetic generator (one statement per line), it uses a direct line parser.
Any file outside that subset
(``strict`` graphs, subgraphs, edge chains, attribute defaults, comments,
several statements per line) falls back to pydot.  The equivalence is checked
in ``tests/test_dotio.py``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import networkx as nx

_ID = r'"(?:[^"\\]|\\.)*"|[A-Za-z0-9_.]+'
_VAL = r'"(?:[^"\\]|\\.)*"|[^\s,;=\]"]+'
_HEADER = re.compile(r'^\s*digraph\s*(' + _ID + r')?\s*\{\s*$')
_GRAPH_ATTR = re.compile(r'^\s*graph\s*\[.*\]\s*;?\s*$')  # lower case only; others fall back
_NODE = re.compile(r'^\s*(' + _ID + r')\s*(?:\[(.*)\])?\s*;?\s*$')
_EDGE = re.compile(r'^\s*(' + _ID + r')\s*->\s*(' + _ID + r')\s*(?:\[(.*)\])?\s*;?\s*$')
_ATTR = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(' + _VAL + r')\s*[,;]?')
_RESERVED = {"graph", "node", "edge", "subgraph", "digraph", "strict"}


class _Fallback(Exception):
    pass


def _attrs(text: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not text:
        return out
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = _ATTR.match(text, pos)
        if not m or m.end() == pos:
            raise _Fallback
        out[m.group(1)] = m.group(2)
        pos = m.end()
    return out


def _name(token: str) -> str:
    name = token.strip('"')
    if name.lower() in _RESERVED:      # keywords in any case, quoted or not
        raise _Fallback
    return name


def _fast(text: str) -> nx.MultiDiGraph:
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i == len(lines):
        raise _Fallback
    m = _HEADER.match(lines[i])
    if not m:
        raise _Fallback
    gname = (m.group(1) or "").strip('"')
    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Tuple[str, str, Dict[str, str]]] = []
    closed = False
    for line in lines[i + 1:]:
        s = line.strip()
        if not s:
            continue
        if closed:
            raise _Fallback
        if s == "}":
            closed = True
            continue
        if s.startswith(("#", "/*", "//")):
            raise _Fallback
        if _GRAPH_ATTR.match(s):
            continue
        e = _EDGE.match(s)
        if e:
            edges.append((_name(e.group(1)), _name(e.group(2)), _attrs(e.group(3))))
            continue
        if "->" in s.split("[", 1)[0]:
            raise _Fallback                    # edge chain or unsupported form
        n = _NODE.match(s)
        if not n:
            raise _Fallback
        nodes.setdefault(_name(n.group(1)), {}).update(_attrs(n.group(2)))
    if not closed:
        raise _Fallback
    g = nx.MultiDiGraph(name=gname)
    for v, a in nodes.items():
        g.add_node(v, **a)
    for u, v, a in edges:
        g.add_edge(u, v, **a)
    return g


def read_dot(path) -> nx.MultiDiGraph:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return _fast(text)
    except _Fallback:
        return nx.drawing.nx_pydot.read_dot(str(path))
