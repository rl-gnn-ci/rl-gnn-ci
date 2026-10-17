"""Structural comparison of DDGs (``python -m ci_ident.ddg.compare REF CANDIDATE...``).

Reports whether one of the candidate DOT files has the same node names in the
same order, the same edges and the same operation per node as the reference.
LLVM value numbers and metadata inside the labels are not compared.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from ci_ident.ddg.graph import load_ddg, node_order


def same_structure(a, b) -> bool:
    return (node_order(a) == node_order(b)
            and Counter(a.edges()) == Counter(b.edges())
            and all(a.nodes[n]["op"] == b.nodes[n]["op"] for n in a.nodes()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ci_ident.ddg.compare",
                                 description="Structural comparison of DDGs.")
    ap.add_argument("ref", help="reference DOT file")
    ap.add_argument("candidates", nargs="+", help="candidate DOT files")
    a = ap.parse_args(argv)
    ref = load_ddg(a.ref)
    for p in a.candidates:
        try:
            g = load_ddg(p)
        except ValueError:
            continue
        if g.number_of_nodes() == ref.number_of_nodes() and same_structure(ref, g):
            print(f"IDENTICAL  {a.ref}  <-  {p}")
            return 0
    print(f"NOT FOUND  {a.ref} ({ref.number_of_nodes()} nodes) among {len(a.candidates)} candidates")
    return 1


if __name__ == "__main__":
    sys.exit(main())
