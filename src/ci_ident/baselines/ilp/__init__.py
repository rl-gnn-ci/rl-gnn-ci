"""ILP-based custom-instruction identification baseline.

``identify_ilp(g, max_in, max_out, area_factor, ...)`` runs the two stages
(:mod:`.enumerator`, :mod:`.selector`) under one deadline and returns the
selected CIs with their speedup; :mod:`.validate` holds the independent
checks every returned CI and selection passes on the original DDG.
"""

from ci_ident.baselines.ilp.api import identify_ilp

__all__ = ["identify_ilp"]
