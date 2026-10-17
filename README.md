# RL-GNN-CI: reinforcement learning on data-dependence graphs for custom-instruction identification

Code accompanying

> E. Hussein, B. Waschneck, C. Mayr. *Reinforcement Learning on Data-Dependence
> Graphs for Custom Instruction Identification.* ACM Transactions on
> Architecture and Code Optimization (TACO) 23(3), 2026.
> <https://doi.org/10.1145/3828552>

and its follow-up work

> E. Hussein, B. Waschneck, C. Mayr. *Reinforcement-Learned Custom-Instruction
> Identification in the Disjoint Design Space.* Nordic Circuits and Systems
> Conference (NorCAS), 2026 (to appear).

RL-GNN-CI identifies custom instructions (CIs) for an application-specific
processor. It works on the data-dependence graph (DDG) of a basic block. A
graph neural network (GIN) encodes the DDG and the current partial solution,
and a policy trained with proximal policy optimisation (PPO) grows CI
candidates node by node and commits them.

Every hardware constraint is enforced by the action mask, in training as in
evaluation:

* operation admissibility;
* the register-port bound (I/O);
* convexity, including set-level acyclicity of the committed CIs;
* a global area budget, charged once per distinct pattern.

The release also contains:

* the ILP baseline;
* the benchmark DDGs (MiBench, JM H.264 encoder, MediaBench) and the LLVM
  pass that extracted them;
* the synthetic training-graph generator.

> **Versions.** `v2.x` (this branch, `main`) implements the formulation of
> the follow-up work. It has an undiscounted objective whose return is
> `5 log S` (S the block speedup), one strict mask, candidate-conditioned node
> features, and a full training pipeline.
>
> The formulation of the TACO paper, with its eight variants
> ({Base, NoMask, NoShape, AbsGain} x {PPO, DQN}) and the SARSA baseline, is
> release `v1.1.0` (tag `v1.1.0`, branch `taco`).
>
> [docs/VERSIONS.md](docs/VERSIONS.md) compares the two.

## Installation

Python 3.10 or 3.11, CPU or CUDA.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[train,ilp,test]"      # core + RLlib + SCIP + pytest
pytest                                  # about 1 min on 2 cores; `pytest -m slow` adds a PPO smoke run (under 1 min)
```

The core install (environment, model, evaluation) needs only torch 2.2 and
DGL 2.1. `[train]` adds Ray RLlib 2.34 (old API stack) and `[ilp]` adds
PySCIPOpt; CPLEX is used instead if `docplex` is importable.
`requirements-lock.txt` lists the exact versions the release was tested with.
For CUDA, install the torch and DGL wheels that match your CUDA version first.

## Quick start

```bash
# 1. synthetic training corpus: 200 graphs per size, |V| = 10, 12, ..., 2000 (~9 GB; --per-size 20 for ~0.9 GB)
ci-ident-synth data/synth --jobs 8          # writes data/synth/dot/*.dot and data/synth/manifest.json
ci-ident-stats data/synth/manifest.json --sample 2000   # report what you train on

# 2. train (reference configuration; one cell io=(4,2), area factor 0.4, radius 1)
ci-ident-train configs/ppo.yaml runs/ppo_s0 --corpus data/synth/manifest.json --seed 0
#    -> runs/ppo_s0/{config.json, progress.jsonl, rllib/, model_iter_*.pt}

# 3. evaluate on the 240-scenario grid (8 DDGs x 6 port bounds x 5 area factors)
ci-ident-bench rl  results/rl_s0.jsonl  --model runs/ppo_s0/model_iter_003000.pt
ci-ident-bench ilp results/ilp.jsonl    --time-budget 3600
ci-ident-bench-summary results/rl_s0.jsonl results/ilp.jsonl
```

To evaluate single graphs, use
`ci-ident-eval MODEL.pt benchmarks/manifest.json --io "4,2;2,1" --area-factors 0.2,1.0`.
In Python, call `ci_ident.rl.infer.evaluate`.

Any DDG in DOT format can be used. Build a manifest with
`ci-ident-manifest my.json path/to/dots/` (the format is in
[docs/FORMULATION.md](docs/FORMULATION.md#graphs)). Node labels are LLVM IR
instructions or bare opcodes; an `op` attribute takes precedence.

## Repository layout

| Path | Contents |
|---|---|
| `src/ci_ident/costs.py` | operator cost model and opcode classification |
| `src/ci_ident/ddg/` | DOT loading and annotation, per-episode DDG state (I/O, convexity under contraction), 1-WL pattern signature |
| `src/ci_ident/env/` | the Gymnasium environment (`CIEnv`) and its configuration |
| `src/ci_ident/model/` | GIN policy/value network and the graph store |
| `src/ci_ident/rl/` | PPO training (RLlib), callbacks, greedy evaluation |
| `src/ci_ident/baselines/ilp/` | ILP baseline: MILP enumeration with contraction, isomorphism-grouped 0-1 selection, independent validation |
| `src/ci_ident/bench/` | scenario-grid runner and summary |
| `src/ci_ident/data/` | corpus manifests, synthetic generator, statistics |
| `configs/ppo.yaml` | the reference training configuration (every key, with defaults) |
| `benchmarks/` | the 8 benchmark DDGs, their manifest, the 240-scenario grid |
| `cpp/ddg-pass/` | LLVM pass plug-in that extracts per-basic-block DDGs |
| `scripts/extract_mibench.sh` | re-extracts the MiBench DDGs and compares them with `benchmarks/` |
| `docs/` | formulation, comparison with the TACO formulation, reproduction, benchmarks |

## Trained models

This repository contains no trained weights. `ci-ident-train` produces
exported models (`model_iter_*.pt`). Each one carries its provenance:

* the package version;
* the full configuration;
* the seed and the iteration;
* the SHA-256 of the training-corpus manifest.

A model loads with strict parameter matching only.
[docs/REPRODUCING.md](docs/REPRODUCING.md) gives the training procedure and
its compute budget.

## Citation

The v2 formulation (this release):

```bibtex
@inproceedings{hussein2026disjoint,
  author    = {Hussein, Eslam and Waschneck, Bernd and Mayr, Christian},
  title     = {Reinforcement-Learned Custom-Instruction Identification in the Disjoint Design Space},
  booktitle = {Nordic Circuits and Systems Conference (NorCAS)},
  year      = {2026},
  note      = {To appear}
}
```

The TACO formulation and the benchmark setting:

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
version (2.0.0) you used.

## License

MIT (see `LICENSE`).
