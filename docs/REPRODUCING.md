# Reproducing and extending

## Scope

This release trains and evaluates the v2 formulation (see
[FORMULATION.md](FORMULATION.md)). The formulation of the TACO paper is
release `v1.1.0` ([VERSIONS.md](VERSIONS.md)).

The test suite checks:

* **Invariants.** On every tested episode, every committed CI passes
  independent checks on the original DDG: port bound, convexity, set-level
  acyclicity, admissibility, area budget. The episode return equals
  `5 * log S` (`pytest`).
* **Pipeline.** Training, export, resuming and evaluation run end to end
  (`pytest -m slow`).

`configs/ppo.yaml` is the reference configuration: entropy coefficient
0.01 -> 0.0005 exponentially over 244 iterations, RLlib's adaptive KL
penalty, and the corpus of the synthetic generator.
`scripts/extract_mibench.sh` re-extracts the MiBench DDGs
([BENCHMARKS.md](BENCHMARKS.md)).

## Steps

```bash
pip install -e ".[train,ilp,test]"
pytest && pytest -m slow

# training corpus: 200 graphs per size (199,200 graphs, about 9 GB of DOT);
# --per-size 20 gives a 19,920-graph corpus (about 0.9 GB) with the same size distribution
ci-ident-synth data/synth --jobs 16
ci-ident-stats data/synth/manifest.json --sample 5000 > data/synth/stats.json

# PPO, one run per seed
for s in 0 1 2; do
  ci-ident-train configs/ppo.yaml runs/ppo_s$s --corpus data/synth/manifest.json \
      --seed $s --num-env-runners 16 --num-gpus 1
done

# the 240-scenario grid
for s in 0 1 2; do
  ci-ident-bench rl results/rl_s$s.jsonl --model runs/ppo_s$s/model_iter_003000.pt
done
ci-ident-bench ilp results/ilp.jsonl --time-budget 3600
ci-ident-bench-summary results/*.jsonl
```

To choose a model, use `progress.jsonl` (`episode_reward_mean` is
5 times the mean log-speedup of the training episodes) or a held-out synthetic manifest
evaluated with `ci-ident-eval`. Do not select on the benchmark grid you
report.

## Compute

* **Training budget.** The reference run is 3000 iterations of 4096
  environment steps, about 12.3 M steps.
* **Measured on one CPU core** (sampling with one environment and the
  network on CPU): about 210 steps/s at |V| = 50, 140 steps/s at 200,
  57 steps/s at 800, and 22 steps/s at 2000. The per-step network forward
  dominates.
* **Estimate.** With 16 environment runners and one GPU for learning, a run
  takes on the order of a day.
* **Memory.** Each process caches parsed graphs (env `cache_size`, default
  512, about 0.7 MB per 400-node graph) and DGL graphs (the graph store,
  1024 graphs, about 0.2 MB each). Plan for about 0.6 GB per environment
  runner on top of Ray and PyTorch, i.e. roughly 16 GB for 16 runners.
* **ILP.** With the reference budget of one hour per scenario the full
  grid is up to 240 h with one solver process. Scenarios are independent:
  split them with `--only` and run several processes.

## Determinism

* Synthetic graphs are byte-reproducible from `(seed, size, replica)`.
* Environment sampling is seeded from the run seed, the worker index and
  the vector index.
* RLlib training is seeded (`.debugging(seed=...)`) but is not bit-exact
  across hardware or numbers of runners.
* Evaluation is deterministic: greedy, with no sampling.
