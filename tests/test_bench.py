import math
from pathlib import Path

from ci_ident.bench.grid import load_grid, summary

GRID = Path(__file__).resolve().parents[1] / "benchmarks" / "grid_240.yaml"


def test_grid_is_the_papers():
    corpus, scen = load_grid(GRID)
    assert len(scen) == 240 and len(corpus) == 8
    assert len({s[0] for s in scen}) == 8
    assert sorted({s[2] for s in scen}) == [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6)]
    assert sorted({s[3] for s in scen}) == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_summary_geomean():
    rows = [{"method": "rl", "benchmark": "a", "io": [2, 1], "speedup": 2.0, "status": "ok"},
            {"method": "rl", "benchmark": "b", "io": [2, 1], "speedup": 8.0, "status": "ok"},
            {"method": "rl", "benchmark": "b", "io": [4, 2], "speedup": None, "status": "ok"}]
    s = summary(rows)["rl:None"]
    assert s["scenarios"] == 2 and math.isclose(s["geomean_speedup"], 4.0)
