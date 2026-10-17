"""Baselines the RL method is compared against.

:mod:`ci_ident.baselines.ilp` -- ILP-based CI identification (iterative
ILP-IO extraction with contraction, then a 0-1 MILP that selects a
conflict-free subset of the extracted instances).

:mod:`ci_ident.baselines.sarsa` -- exact convex enumeration, then tabular
SARSA selection of a conflict-free subset of the instances (C++ extensions,
built with ``python -m ci_ident.baselines.sarsa.build``).
"""
