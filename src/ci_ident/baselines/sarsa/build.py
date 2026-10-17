"""Compile the two C++ extensions of the SARSA baseline into this package.

Usage (from anywhere; the outputs land next to this file)::

    python -m ci_ident.baselines.sarsa.build            # build both
    python -m ci_ident.baselines.sarsa.build --check    # report availability

Requirements: a C++17 compiler (``$CXX``, default ``c++``) and ``pybind11``
importable by the running interpreter (``pip install pybind11``).  The module
suffix comes from ``sysconfig`` of the running interpreter, so the extensions
are built for exactly the Python that runs this command.

Sources (``csrc/``):

* ``xiao_enum.cpp``  -> ``_xiao_enum``   exact candidate enumeration
* ``sarsa_core.cpp`` -> ``_sarsa_core``  tabular SARSA selection
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List

PKG_DIR = Path(__file__).resolve().parent
SRC_DIR = PKG_DIR / "csrc"
EXTENSIONS = {"_xiao_enum": "xiao_enum.cpp", "_sarsa_core": "sarsa_core.cpp"}
CXXFLAGS = ["-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-shared", "-fvisibility=hidden",
            "-Wall", "-Wextra"]


def ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _includes() -> List[str]:
    try:
        import pybind11
    except ImportError as e:  # pragma: no cover - environment dependent
        raise SystemExit("pybind11 is required to build the SARSA extensions: pip install pybind11") from e
    inc = [pybind11.get_include(), sysconfig.get_paths()["include"]]
    plat = sysconfig.get_paths().get("platinclude")
    if plat and plat not in inc:
        inc.append(plat)
    return [f"-I{p}" for p in inc]


def build(names=None, verbose: bool = True) -> List[Path]:
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    extra = shlex.split(os.environ.get("CXXFLAGS", ""))
    ldflags = []
    if sys.platform == "darwin":  # pragma: no cover
        ldflags = ["-undefined", "dynamic_lookup"]
    out_paths = []
    for mod, src in EXTENSIONS.items():
        if names and mod not in names:
            continue
        out = PKG_DIR / f"{mod}{ext_suffix()}"
        tmp = out.with_name(out.name + ".tmp")
        cmd = [*cxx, *CXXFLAGS, *extra, *_includes(), str(SRC_DIR / src), "-o", str(tmp), *ldflags]
        if verbose:
            print(" ".join(shlex.quote(c) for c in cmd), flush=True)
        subprocess.run(cmd, check=True)
        os.replace(tmp, out)  # atomic: a running interpreter keeps the old mapping
        out_paths.append(out)
    return out_paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ci_ident.baselines.sarsa.build",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only report whether the extensions import")
    ap.add_argument("-q", "--quiet", action="store_true", help="do not print the compiler commands")
    a = ap.parse_args(argv)
    if not a.check:
        build(verbose=not a.quiet)
    from ci_ident.baselines.sarsa import _ext

    ok, msg = _ext.status()
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
