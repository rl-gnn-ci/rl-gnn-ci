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


def test_rl_runner_with_a_saved_network(tmp_path):
    import torch

    from ci_ident.bench.grid import run_rl
    from ci_ident.model import TacoNetwork
    from ci_ident.rl.infer import save_model
    from ci_ident.variants import load_variant

    corpus, scen = load_grid(GRID)
    env_cfg, mcfg = load_variant("ppo-absgain")
    torch.manual_seed(0)
    p = save_model(tmp_path / "m.pt", TacoNetwork(mcfg), env_cfg, variant="ppo-absgain")
    rows = list(run_rl(str(p), corpus, [s for s in scen if s[0] == "Susan"][:2]))
    assert len(rows) == 2 and all(r["variant"] == "ppo-absgain" for r in rows)
    assert all(r["speedup"] >= 1.0 and r["status"] == "ok" for r in rows)


def test_sarsa_rows_are_averaged_over_seeds():
    rows = [{"method": "sarsa", "benchmark": "a", "io": [2, 1], "area_factor": 0.2, "seed": s,
             "speedup": sp, "status": "ok"} for s, sp in ((0, 1.0), (1, 2.0), (2, 3.0))]
    s = summary(rows)["sarsa"]
    assert s["scenarios"] == 1 and math.isclose(s["geomean_speedup"], 2.0)
