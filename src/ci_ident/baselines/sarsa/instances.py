"""From enumerated candidates to CI instances and patterns.

Every stored candidate ``S`` (``|S| >= 2``) becomes an instance with

* saving ``Delta(S) = l_SW(S) - l_CI(S) = |S| - cycles(cp(S))`` (*Problem
  Formulation*: one software cycle per admissible operation, ``cp`` the longest
  hardware-latency path inside ``S``, :func:`ci_ident.costs.cycles` the whole
  cycles with tolerance :data:`ci_ident.costs.EPS`); only instances with a
  positive saving are kept;
* pattern ``signature(S)``, the direction-aware 1-WL signature
  :class:`ci_ident.ddg.signature.SignatureIndex` used by every method of the
  repository; the pattern's area is ``sum_{v in S} a(v)`` and is charged once.

Patterns are numbered in order of first appearance and instances keep the
enumeration order, so the result is deterministic.  The loop polls a
deadline; on expiry the instances built so far are returned with
``truncated=True``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import numpy as np

from ci_ident.costs import cycles
from ci_ident.ddg.signature import SignatureIndex


def build_instances(g, res: Dict[str, Any], deadline_at: float = math.inf,
                    poll: int = 256) -> Dict[str, Any]:
    """Instances and patterns of an :func:`~ci_ident.baselines.sarsa.enumeration.enumerate_candidates`
    result.  ``deadline_at`` is an absolute :func:`time.monotonic` value.

    Returns ``nodes``/``offsets`` (topological indices per kept instance),
    ``gains`` (int64), ``pattern`` (int32 per instance), ``pattern_area``
    (float64 per pattern), ``signatures`` (per pattern), ``truncated`` and
    ``num_seen`` (stored candidates processed).
    """
    gi = res["index"]
    order, area = gi.order, gi.area
    nodes, off, cp = res["nodes"], res["offsets"], res["cp"]
    sig_index = SignatureIndex.from_graph(g)
    pat_of_sig: Dict[Any, int] = {}
    pattern_area: List[float] = []
    signatures: List[Any] = []
    kept: List[int] = []
    gains: List[int] = []
    pattern: List[int] = []
    truncated = False
    n = len(off) - 1
    seen = 0
    for k in range(n):
        if k % poll == 0 and time.monotonic() >= deadline_at:
            truncated = True
            break
        seen += 1
        idx = nodes[off[k]:off[k + 1]]
        if len(idx) < 2:
            continue
        saving = len(idx) - cycles(float(cp[k]))
        if saving <= 0:
            continue
        names = [order[i] for i in idx]
        sig = sig_index.signature(names)
        p = pat_of_sig.get(sig)
        if p is None:
            p = len(pattern_area)
            pat_of_sig[sig] = p
            pattern_area.append(float(sum(area[i] for i in idx)))
            signatures.append(sig)
        kept.append(k)
        gains.append(int(saving))
        pattern.append(p)
    new_nodes, new_off = gather(nodes, off, np.asarray(kept, dtype=np.int64))
    return {
        "nodes": new_nodes,
        "offsets": new_off,
        "gains": np.asarray(gains, dtype=np.int64),
        "pattern": np.asarray(pattern, dtype=np.int32),
        "pattern_area": np.asarray(pattern_area, dtype=np.float64),
        "signatures": signatures,
        "truncated": truncated,
        "num_seen": seen,
    }


def gather(nodes: np.ndarray, off: np.ndarray, sel: np.ndarray):
    """Flat ``(nodes, offsets)`` of the segments ``sel`` of a flat segment list."""
    off = np.asarray(off, dtype=np.int64)
    starts, sizes = off[sel], off[sel + 1] - off[sel]
    new_off = np.zeros(len(sel) + 1, dtype=np.uint64)
    np.cumsum(sizes, out=new_off[1:])
    if not len(sel) or not int(sizes.sum()):
        return np.zeros(0, dtype=np.uint32), new_off
    # position i of the output reads nodes[starts[seg(i)] + (i - new_off[seg(i)])]
    rel = np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(new_off[:-1].astype(np.int64), sizes)
    return np.asarray(nodes, dtype=np.uint32)[np.repeat(starts, sizes) + rel], new_off


def priority_order(inst: Dict[str, Any]) -> np.ndarray:
    """Instance order handed to the selector: higher saving first, then larger
    instance, then enumeration order (greedy ties go to the earliest)."""
    sizes = np.diff(inst["offsets"].astype(np.int64))
    ids = np.arange(len(sizes))
    return np.lexsort((ids, -sizes, -inst["gains"])) if len(sizes) else ids


def reorder(inst: Dict[str, Any], perm: np.ndarray) -> Dict[str, Any]:
    """``inst`` with its instances permuted by ``perm`` (patterns unchanged)."""
    nodes, new_off = gather(inst["nodes"], inst["offsets"], np.asarray(perm, dtype=np.int64))
    out = dict(inst)
    out.update(nodes=nodes, offsets=new_off, gains=inst["gains"][perm],
               pattern=inst["pattern"][perm])
    return out
