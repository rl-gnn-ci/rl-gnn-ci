"""Loading and annotating data-dependence graphs (DDGs).

A DDG is read from a Graphviz DOT file, one basic block per file.  Node order
is the order in which nodes appear in the DOT file; this order defines the
node index used by the environment's action space *and* by the graph tensors
the GNN reads (see :mod:`ci_ident.model.graph_store`), so both sides are built
from :func:`load_ddg` and nothing else.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Union

import networkx as nx

from ci_ident.costs import VALID_OPERATIONS, extract_operation
from ci_ident.ddg.dotio import read_dot

PathLike = Union[str, Path]


def load_ddg(dot_path: PathLike) -> nx.MultiDiGraph:
    """Parse a DOT file and annotate it (see :func:`annotate`).

    Raises ``ValueError`` if the graph is empty or not acyclic.
    """
    g = read_dot(dot_path)
    if g.number_of_nodes() == 0:
        raise ValueError(f"{dot_path}: empty DDG")
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(f"{dot_path}: DDG contains a cycle")
    annotate(g)
    return g


def annotate(g: nx.DiGraph) -> Set[str]:
    """Annotate every node with ``op``, ``hw_latency``, ``sw_latency``, ``area``
    and ``ci_eligible``; return the set of inadmissible nodes.

    A node is CI-admissible iff its operation is in
    :data:`ci_ident.costs.VALID_OPERATIONS` and it has at least one
    predecessor (a source is a block input, not a computation of the block).
    Inadmissible nodes keep one cycle of software latency, so the block's
    software time is ``T_sw = sum_v ceil(hw_latency(v)) = |V|``.
    """
    invalid: Set[str] = set()
    for n, data in g.nodes(data=True):
        op = extract_operation(data.get("label", ""), data.get("op") or None)
        data["op"] = op
        if op in VALID_OPERATIONS and g.in_degree(n) != 0:
            hw, area = VALID_OPERATIONS[op]
            data["hw_latency"] = float(hw)
            data["sw_latency"] = 1.0
            data["area"] = float(area)
            data["ci_eligible"] = True
        else:
            invalid.add(n)
            data["hw_latency"] = 1.0
            data["sw_latency"] = 1.0
            data["area"] = 0.0
            data["ci_eligible"] = False
    return invalid


def node_order(g: nx.DiGraph) -> List[str]:
    """The canonical node order (DOT insertion order)."""
    return list(g.nodes())


def node_order_digest(nodes: Iterable[str]) -> str:
    """Stable digest of a node order, used to verify that a stored graph
    tensor and the environment index nodes identically."""
    h = hashlib.sha256()
    for n in nodes:
        h.update(str(n).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def edge_index(g: nx.DiGraph) -> Tuple[List[int], List[int]]:
    """``(src, dst)`` index lists in canonical node order (parallel edges kept)."""
    pos = {n: i for i, n in enumerate(g.nodes())}
    src, dst = [], []
    for u, v in g.edges():
        src.append(pos[u])
        dst.append(pos[v])
    return src, dst


def static_node_features(g: nx.DiGraph) -> List[Tuple[float, float]]:
    """Per-node static features ``(hw_latency, area)`` in canonical order;
    ``(0, 0)`` for inadmissible nodes, so the zero pattern marks exactly the
    nodes no CI may contain."""
    out = []
    for _, d in g.nodes(data=True):
        if d["ci_eligible"]:
            out.append((float(d["hw_latency"]), float(d["area"])))
        else:
            out.append((0.0, 0.0))
    return out


def area_reference(g: nx.DiGraph) -> float:
    """``A_ref``: total area of the CI-admissible nodes."""
    return float(sum(d["area"] for _, d in g.nodes(data=True) if d["ci_eligible"]))
