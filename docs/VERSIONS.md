# Releases

Release 2.x implements the v2 formulation (NorCAS). Release 1.x (tag
`v1.1.0`, branch `taco`) implements the formulation of the TACO paper with
its eight variants.

## Common to both releases

* The cost model and the opcode classification.
* The DDG semantics.
* The validity conditions:
  * convexity, with committed CIs contracted, so the committed set is
    acyclic;
  * operation legality;
  * port bounds, with block sinks counted as outputs;
  * connectivity, at growth radius 1.
* Area charged once per 1-WL pattern, with the budget `alpha * A_ref`.
* The speedup model.
* The benchmark DDGs and the 240-scenario grid.
* The synthetic generator, and the graph-identity checks of the graph
  store.

## Differences

| | v1.x (TACO) | v2.x (NorCAS) |
|---|---|---|
| Masks | training mask (frontier, post-hoc checks, regular port bound and free-node convexity at commit); inference mask with FeasGrow | one mask for training and evaluation, containing every condition including area |
| Objective | discounted (`gamma < 1`) | undiscounted (`gamma = 1`) |
| Commit reward | `w_perf * delta log S` + size term | `5 * delta log S` |
| Shaping | gain (`rho` or `Delta`) minus I/O and convexity terms, discounted | saving `l_SW - cp(C)` minus an optional area price, undiscounted, charged back at every close |
| Other terms | extension reward with modulator, size term, penalties | none |
| Node features | status, opcode, latency, area | status, latency, area, and the candidate-conditioned `[depth_C, latency, height_C] / S` |
| Global features | 12 | 16, including a one-step look-ahead of the best saving gain and `cp(C)/S` |
| Growth | connected (radius 1) | radius `r` (default 1; `r >= 2` admits disjoint CIs) |
| Algorithms | PPO and dueling DQN, eight variants | PPO |
| Training | environment, network and variant configurations | `ci-ident-train` (RLlib), training cell (4, 2) / 0.4, log-uniform graph sizes |
| Baselines | ILP, SARSA | ILP |
