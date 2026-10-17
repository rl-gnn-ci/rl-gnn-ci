from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"


@pytest.fixture(scope="session")
def bench_manifest():
    return str(BENCH / "manifest.json")


@pytest.fixture(scope="session")
def synth_corpus(tmp_path_factory):
    """A small synthetic corpus (sizes 10..120) built with the release generator."""
    from ci_ident.data.synth import SynthConfig, generate_corpus

    out = tmp_path_factory.mktemp("synth")
    generate_corpus(out, SynthConfig(sizes=tuple(range(10, 121, 10)), per_size=3), verbose=False)
    return str(out / "manifest.json")
