"""Build a corpus manifest from DOT files (``ci-ident-manifest``)."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

from ci_ident.ddg.graph import load_ddg, node_order, node_order_digest


def _entry(p: str) -> dict:
    g = load_ddg(p)
    return {"num_nodes": g.number_of_nodes(), "num_edges": g.number_of_edges(),
            "node_order": node_order_digest(node_order(g))}


def build_manifest(dot_files: Sequence, manifest, *, name: Optional[str] = None,
                   jobs: int = 1, max_nodes: int = 2000) -> Path:
    """Scan ``dot_files`` in the given order and write a manifest; paths are
    stored relative to the manifest's directory when possible.  Graphs larger
    than ``max_nodes`` (the action-space size) are rejected."""
    manifest = Path(manifest).expanduser().resolve()
    root = manifest.parent
    files = [str(Path(p).expanduser().resolve()) for p in dot_files]
    if not files:
        raise ValueError("no DOT files given")
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            entries = list(ex.map(_entry, files, chunksize=64))
    else:
        entries = [_entry(p) for p in files]
    graphs = []
    for p, e in zip(files, entries):
        if e["num_nodes"] > max_nodes:
            raise ValueError(f"{p}: {e['num_nodes']} nodes exceeds max_nodes={max_nodes}")
        try:
            rel = str(Path(p).relative_to(root))
        except ValueError:
            rel = p
        graphs.append({"path": rel, **e})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"name": name or manifest.stem, "root": ".", "graphs": graphs},
                                   indent=1))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="ci-ident-manifest", description="Build a corpus manifest from DOT files.")
    ap.add_argument("manifest", help="output manifest (JSON)")
    ap.add_argument("dots", nargs="+", help="DOT files or directories (searched for *.dot, sorted)")
    ap.add_argument("--name", help="corpus name (default: the manifest file name)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes (default 1)")
    a = ap.parse_args(argv)
    files = []
    for d in a.dots:
        p = Path(d)
        files += sorted(p.glob("*.dot")) if p.is_dir() else [p]
    out = build_manifest(files, a.manifest, name=a.name, jobs=a.jobs)
    print(f"wrote {out} ({len(files)} graphs)")


if __name__ == "__main__":
    main()
