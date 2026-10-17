# RL-GNN-CI (TACO formulation, release 1.1)

Code accompanying

> E. Hussein, B. Waschneck, C. Mayr. *Reinforcement Learning on Data-Dependence
> Graphs for Custom Instruction Identification.* ACM Transactions on
> Architecture and Code Optimization (TACO) 23(3), 2026.
> <https://doi.org/10.1145/3828552>

RL-GNN-CI identifies custom instructions (CIs) for an application-specific
processor. It works on the data-dependence graph (DDG) of a basic block. A
graph neural network encodes the DDG and the identification state, and a
reinforcement-learning policy grows CI candidates node by node and commits
them. The design space consists of convex, CI-legal, connected subgraphs
within register-port (I/O) bounds, under a global area budget that is
charged once per pattern.

This release (`v1.x`, branch `taco`) implements the formulation of the TACO
paper. The `v2.x` releases (branch `main`) implement the formulation of the
follow-up work (NorCAS 2026).

## Contents

* **The environment** (`ci_ident.env`), configured by the two ablation
  axes: constraint enforcement (strict masking or none) and reward signal
  (Base, NoShape, AbsGain). It has a training mask and an inference mask.
* **The networks of the eight variants**, {Base, NoMask, NoShape, AbsGain}
  x {PPO, DQN}:
  * a GIN, GATv2 or GCN encoder with mean, sum or attention pooling;
  * an actor-critic head, or a dueling Q head with optional NoisyNet.

  The variant configurations are in `src/ci_ident/variants/`.
* **Greedy inference** with a trained network (`ci-ident-eval`).
* **The baselines:**
  * the ILP baseline: MILP enumeration with contraction, then
    isomorphism-grouped 0-1 selection;
  * exact convex enumeration, then tabular SARSA selection of
    conflict-free instances.
* **The eight benchmark DDGs** (MiBench, MediaBench, JM H.264), the
  240-scenario evaluation grid, and the LLVM pass that extracts DDGs.
* **The synthetic DDG generator.**

The release contains no trained weights and no training pipeline.
`ci-ident-eval` and `ci-ident-bench rl` evaluate networks saved with
`ci_ident.rl.infer.save_model`.

[docs/FORMULATION.md](docs/FORMULATION.md) specifies the formulation,
[docs/VARIANTS.md](docs/VARIANTS.md) the eight variants and the learner
settings of the networks reported in the paper, and
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) the benchmark DDGs and their
extraction.

## Installation

Python 3.10 or 3.11.

```bash
pip install -e ".[ilp,sarsa,test]"           # torch 2.2.1, DGL 2.1, PySCIPOpt, pybind11
python -m ci_ident.baselines.sarsa.build     # C++17 extensions of the SARSA baseline
pytest
```

The ILP baseline uses PySCIPOpt; CPLEX is used instead if `docplex` is
importable. `requirements-lock.txt` lists the exact versions the release was
tested with.

## Usage

```python
import torch
from ci_ident.env import CIEnv
from ci_ident.model import GraphStore, TacoNetwork
from ci_ident.variants import load_variant

env_cfg, model_cfg = load_variant("dqn-absgain")
env = CIEnv({**env_cfg, "corpus": "benchmarks/manifest.json", "io": [4, 2],
             "area_factor": 0.4, "mode": "train"})
net = TacoNetwork(model_cfg, GraphStore("benchmarks/manifest.json"))
obs, _ = env.reset()
scores, value = net({k: torch.as_tensor(v[None]) for k, v in obs.items()})
```

A network is saved with
`ci_ident.rl.infer.save_model(path, net, env_cfg, variant="dqn-absgain")`.
It is evaluated in the inference mode of its variant:

```bash
ci-ident-eval my_model.pt benchmarks/manifest.json --io "4,2;6,3" --area-factors 0.4,1.0
ci-ident-bench rl    results/rl.jsonl    --model my_model.pt
ci-ident-bench ilp   results/ilp.jsonl   --time-budget 3600
ci-ident-bench sarsa results/sarsa.jsonl --time-budget 3600 --seeds 0,1,2
ci-ident-bench-summary results/*.jsonl
```

Corpora of other DDGs are built with `ci-ident-manifest`, synthetic corpora
with `ci-ident-synth`, and corpus statistics with `ci-ident-stats`.

## Repository layout

| Path | Contents |
|---|---|
| `src/ci_ident/costs.py` | operator cost model and opcode classification |
| `src/ci_ident/ddg/` | DOT loading, the identification state (I/O, convexity under contraction), the 1-WL pattern signature |
| `src/ci_ident/env/` | the environment and its configuration |
| `src/ci_ident/model/` | graph store, encoders, PPO and DQN heads |
| `src/ci_ident/variants/` | the eight variant configurations |
| `src/ci_ident/rl/infer.py` | greedy inference |
| `src/ci_ident/baselines/` | ILP and SARSA baselines |
| `src/ci_ident/bench/` | evaluation-grid runner and summary |
| `src/ci_ident/data/` | corpus manifests, synthetic generator, statistics |
| `benchmarks/` | benchmark DDGs, manifest, evaluation grid |
| `cpp/ddg-pass/`, `scripts/` | LLVM DDG pass, MiBench extraction script |
| `docs/` | formulation, variants, benchmarks |

## Citation

```bibtex
@article{hussein2026rlgnnci,
  author    = {Hussein, Eslam and Waschneck, Bernd and Mayr, Christian},
  title     = {Reinforcement Learning on Data-Dependence Graphs for Custom Instruction Identification},
  journal   = {ACM Transactions on Architecture and Code Optimization},
  volume    = {23},
  number    = {3},
  pages     = {1--25},
  month     = jul,
  publisher = {Association for Computing Machinery},
  year      = {2026},
  doi       = {10.1145/3828552}
}
```

`CITATION.cff` holds the machine-readable citation. Please state the release
version (1.1.0) you used.

## License

MIT (see `LICENSE`).
