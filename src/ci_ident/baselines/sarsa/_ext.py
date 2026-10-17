"""Lazy access to the compiled extensions of the SARSA baseline.

The package imports without them; :func:`load` raises
:class:`ExtensionUnavailable` with build instructions when they are absent.
"""

from __future__ import annotations

import importlib
from typing import Tuple

BUILD_HINT = ("build them with `python -m ci_ident.baselines.sarsa.build` "
              "(needs a C++17 compiler and pybind11)")


class ExtensionUnavailable(ImportError):
    """The C++ extensions of the SARSA baseline are not built."""


#: ABI version each extension must report (``ABI_VERSION``).
ABI = {"_xiao_enum": 2, "_sarsa_core": 1}

_cache = {}


def load(name: str):
    """Import ``ci_ident.baselines.sarsa.<name>`` (``_xiao_enum`` or ``_sarsa_core``)."""
    if name not in _cache:
        try:
            mod = importlib.import_module(f"ci_ident.baselines.sarsa.{name}")
        except ImportError as e:
            raise ExtensionUnavailable(f"extension {name} is not available ({e}); {BUILD_HINT}") from e
        if getattr(mod, "ABI_VERSION", None) != ABI[name]:
            raise ExtensionUnavailable(f"extension {name} has ABI version {getattr(mod, 'ABI_VERSION', None)}, "
                                       f"expected {ABI[name]}; re-{BUILD_HINT}")
        _cache[name] = mod
    return _cache[name]


def available() -> bool:
    try:
        load("_xiao_enum")
        load("_sarsa_core")
        return True
    except ExtensionUnavailable:
        return False


def status() -> Tuple[bool, str]:
    try:
        a = load("_xiao_enum")
        b = load("_sarsa_core")
    except ExtensionUnavailable as e:
        return False, str(e)
    return True, f"SARSA extensions available: {a.__file__}, {b.__file__}"
