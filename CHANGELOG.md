# Changelog

## 2.0.0 — 2026-10-17

The formulation of the follow-up work (see `docs/VERSIONS.md`). The TACO
formulation with its eight variants is release 1.1 (tag `v1.1.0`).

* **Environment.**
  * One strict mask for training and evaluation: port bound, convexity
    under contraction, area budget, and growth radius.
  * An undiscounted return equal to `5 log S_final`: commit payments plus
    an unrounded potential that is charged back.
  * Candidate-conditioned node features.
* **Network.** A GIN encoder (4 x 96, LayerNorm) with policy and value heads.
* **Training.** PPO training (`ci-ident-train`, RLlib 2.34 old API stack)
  with the reference configuration (`configs/ppo.yaml`). Exported models
  carry their provenance.
* **Evaluation.** Greedy evaluation without Ray (`ci-ident-eval`), and the
  240-scenario grid (`ci-ident-bench`).
* **ILP baseline.** The MILP enumeration with contraction and the
  isomorphism-grouped 0-1 selection. Every result is validated on the
  original DDG.
* **Data.**
  * Corpus manifests with node-order digests.
  * A DOT reader.
  * The synthetic generator, with statistics.
  * The benchmark DDGs, the LLVM pass, and a MiBench re-extraction script.

## 1.1.0 — 2026-10-17

The TACO formulation (eight variants, networks, inference, ILP and SARSA
baselines). See the `v1.1.0` tag.
