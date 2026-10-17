# Changelog

## 1.1.0 — 2026-10-17

* **Environment.** The environment of the TACO formulation:
  * strict masking or none, combined with the Base, NoShape or AbsGain
    reward;
  * training and inference masks;
  * the FeasGrow and FeasCommit predicates, and termination conditions.
* **Networks.** The networks of the eight variants:
  * GIN, GATv2 and GCN encoders with mean, sum or attention pooling, and an
    opcode embedding;
  * an actor-critic head, or a dueling Q head over the feasible actions
    with optional NoisyNet.
* **Variants.** The variant configurations are in `src/ci_ident/variants/`.
* **Baselines.** Two baselines, each validated independently on the
  original DDG:
  * the ILP: MILP enumeration with contraction, then isomorphism-grouped
    0-1 selection;
  * the SARSA baseline: exact convex enumeration with tabular SARSA
    selection.
* **Data and evaluation.**
  * Corpus manifests with node-order digests.
  * A DOT reader.
  * The synthetic generator, with corpus statistics.
  * The 240-scenario evaluation grid, and greedy inference.
* **Benchmarks.** The benchmark DDGs, the LLVM pass, and a MiBench
  extraction script.
