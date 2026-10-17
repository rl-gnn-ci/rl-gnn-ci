"""Operator cost model and opcode classification.

This module is the single source of truth for which DDG operations may belong
to a custom instruction (CI) and what each one costs. Every method in the
package -- the RL environment, the graph tensors the GNN reads, and the ILP
baseline -- annotates graphs through :func:`ci_ident.ddg.annotate`, which reads
the tables below and nothing else.

Costs are the normalised operator library of the TACO paper (adapted from Wang and
Xiao, 2023): hardware latency is a fraction of one base-core cycle, area is
normalised to a 32x32+64 MAC.  Every admissible operation has a hardware
latency of at most one cycle, so its software latency on the single-issue
in-order core is exactly one cycle.

Division and remainder are inadmissible: they take several cycles on the base
core.  The admissible and
inadmissible tables are disjoint (checked at import).
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional, Tuple

#: Tolerance of comparisons between sums of latencies or areas.  The tables
#: below have two decimals, so exact sums are only ever off by float rounding
#: (e.g. 0.31+0.31+0.31+0.03+0.03+0.01 ... = 1.0000000000000002).
EPS = 1e-9


def cycles(latency: float) -> int:
    """Whole cycles taken by a latency-weighted path: ``ceil`` up to :data:`EPS`."""
    return int(math.ceil(latency - EPS))


#: Admissible operations: ``op -> (hardware latency [cycles], area [MAC units])``.
VALID_OPERATIONS: Dict[str, Tuple[float, float]] = {
    # arithmetic
    "add": (0.31, 0.10),
    "fadd": (0.31, 0.10),
    "sub": (0.31, 0.10),
    "fsub": (0.31, 0.10),
    "neg": (0.31, 0.10),  # fneg, or sub 0, x
    "mul": (0.96, 0.98),
    "fmul": (0.96, 0.98),
    "fma": (1.00, 1.00),  # llvm.fma.* intrinsic
    "mac": (1.00, 1.00),
    # shifts and rotations
    "shl": (0.23, 0.12),
    "lshr": (0.23, 0.12),
    "ashr": (0.23, 0.12),
    "rotl": (0.23, 0.12),  # llvm.fshl.*
    "rotr": (0.23, 0.12),  # llvm.fshr.*
    # bitwise logic
    "and": (0.03, 0.02),
    "or": (0.03, 0.02),
    "xor": (0.03, 0.02),
    "not": (0.03, 0.02),  # xor x, -1
    # comparisons
    "icmp": (0.10, 0.04),
    "fcmp": (0.10, 0.04),
    # casts
    "zext": (0.03, 0.02),
    "sext": (0.03, 0.02),
    "trunc": (0.03, 0.02),
    "bitcast": (0.03, 0.02),
    "ptrtoint": (0.03, 0.02),
    "inttoptr": (0.03, 0.02),
    # address generation
    "getelementptr": (0.31, 0.12),
}

#: Recognised but inadmissible operations, with the reason (for statistics).
INVALID_OPERATIONS: Dict[str, str] = {
    # control flow
    "br": "control", "switch": "control", "ret": "control", "invoke": "control",
    "resume": "control", "unreachable": "control", "phi": "control",
    "select": "control",
    # memory and side effects
    "load": "memory", "store": "memory", "alloca": "memory",
    "atomicrmw": "memory", "cmpxchg": "memory", "fence": "memory",
    "call": "side_effect",
    # vector
    "extractelement": "vector", "insertelement": "vector",
    "shufflevector": "vector",
    # multi-cycle on the base core
    "div": "multi_cycle", "rem": "multi_cycle", "fdiv": "multi_cycle",
    # debug
    "dbg_value": "debug", "dbg_declare": "debug", "nop": "debug",
}

if set(VALID_OPERATIONS) & set(INVALID_OPERATIONS):  # pragma: no cover
    raise ImportError("an opcode is classified both admissible and inadmissible")

#: Intrinsic calls that compute an admissible operation.
_INTRINSICS = {"llvm.fma.": "fma", "llvm.fshl.": "rotl", "llvm.fshr.": "rotr"}

_OP_AFTER_ASSIGN = re.compile(r"=\s*([A-Za-z_][A-Za-z0-9_.]*)")
_OP_AT_START = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)")
# ``xor <ty> x, -1`` (or ``xor i1 x, true``) is a bitwise NOT; the second
# operand must be exactly -1 / true (``xor x, 1`` is an ordinary xor).  LLVM
# labels may carry trailing metadata (``, !ddg.orig_bb !9``).
_NOT_OPERAND = re.compile(r"\bxor\b[^,]*,\s*(-1|true)\s*(,|$)")


def _unquote(s: Optional[str]) -> str:
    s = "" if s is None else str(s)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def extract_operation(label: Optional[str], op_attr: Optional[str] = None) -> str:
    """Normalised opcode of a DDG node.

    The explicit ``op`` attribute written by the DDG pass is preferred; the
    node label (an LLVM IR instruction such as ``%2 = add i32 %1, %0``, or a
    bare opcode as in the synthetic corpus) is parsed otherwise.
    """
    label = _unquote(label)
    if op_attr:
        op = _unquote(op_attr).strip().lower()
    else:
        m = _OP_AFTER_ASSIGN.search(label) or _OP_AT_START.match(label)
        op = (m.group(1) if m else "unknown").lower()

    if op == "call":
        for prefix, name in _INTRINSICS.items():
            if prefix in label:
                op = name
                break
    if op in ("sdiv", "udiv"):
        op = "div"
    elif op in ("srem", "urem", "frem"):
        op = "rem"
    elif op == "fneg":
        op = "neg"
    elif op == "xor" and _NOT_OPERAND.search(label):
        op = "not"
    return op


def classify(op: str) -> str:
    """``"valid"``, ``"invalid"`` or ``"unknown"`` for a normalised opcode."""
    if op in VALID_OPERATIONS:
        return "valid"
    if op in INVALID_OPERATIONS:
        return "invalid"
    return "unknown"
