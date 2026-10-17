# Formulation (release 2.0)

This document specifies what the code computes. File names in parentheses
refer to the source files.

## Graphs

A DDG is one basic block in Graphviz DOT format (`ddg/graph.py`). Node order
is the order of appearance in the file. It defines the node index of the
action space and of the GNN input. Both sides are built from the same file
through `load_ddg`, and the graph store checks a digest of the order
(`model/graph_store.py`). A corpus is a JSON manifest:

```json
{"name": "...", "root": ".", "graphs": [{"path": "dot/x.dot", "num_nodes": 10, "num_edges": 11, "node_order": "<16 hex>"}]}
```

`ci-ident-manifest` builds one. The index of a graph is its position in
`graphs`.

## Cost model and admissibility (`costs.py`)

Every node gets a hardware latency `l(v)` (a fraction of a base-core cycle),
an area `a(v)` (in units of a 32x32+64 MAC), and a software latency of one
cycle. A node is **admissible** iff its operation is in `VALID_OPERATIONS`
and it has at least one predecessor. Graph sources are block inputs.

| Operations | `l(v)` | `a(v)` |
|---|---|---|
| add, fadd, sub, fsub, neg | 0.31 | 0.10 |
| mul, fmul | 0.96 | 0.98 |
| fma, mac | 1.00 | 1.00 |
| shl, lshr, ashr, rotl, rotr | 0.23 | 0.12 |
| and, or, xor, not | 0.03 | 0.02 |
| icmp, fcmp | 0.10 | 0.04 |
| zext, sext, trunc, bitcast, ptrtoint, inttoptr | 0.03 | 0.02 |
| getelementptr | 0.31 | 0.12 |

Division, remainder and `select` are inadmissible, as are memory, control,
call, vector and debug operations. Inadmissible nodes keep one cycle of
software latency, so the software time of a block is `T_sw = |V|`. In the
benchmark DDGs this includes the boundary nodes the DDG pass materialises for
values produced outside the block (`kind="external_in"`; for example 205 of
the 844 nodes of AES).

## Objective

For a CI `C`, `cp(C)` is the longest latency-weighted path inside `C`, and
`saving(C) = |C| - ceil(cp(C))`. `ceil` and the area-budget comparison are
taken with a tolerance of 1e-9 (`costs.cycles`, `costs.EPS`). For a set of committed CIs,
`S = T_sw / (T_sw - sum_C saving(C))` (`env/env.py: speedup`).

Area is charged once per **pattern**. Two CIs share a pattern iff their
1-WL signatures agree (`ddg/signature.py`). The signature uses five rounds,
starting from (op, in-degree, out-degree, topological level) inside the CI.
It consists of the multiset of final node colours and the multiset of
(producer colour, consumer colour) edge pairs.
The budget is `A_budget = area_factor * A_ref`, where `A_ref` is the total
area of the admissible nodes.

## Validity of a CI (`ddg/state.py`)

* **Ports.** `In(C)` is the set of distinct nodes outside `C` with a successor
  in `C`. `Out(C)` is the set of nodes of `C` with a successor outside `C`, or
  with no successor at all (a DDG sink is live-out). The bound is
  `|In(C)| <= max_in` and `|Out(C)| <= max_out`. Counts are taken on the original
  DDG.
* **Convexity under contraction.** No path leaves `C` and re-enters it. Paths
  through committed CIs count, and every committed CI is treated as one
  atomic node. This makes the committed set acyclic as a whole: no
  instruction can depend on another CI that depends on it.
* **Area.** The candidate's area plus the area already charged must not
  exceed `A_budget`. While a candidate is being built, its area is counted
  even if it would repeat a committed pattern.

## Decision process (`env/env.py`)

One episode is one DDG. A **candidate** `C` is grown one node at a time. The
action space is `Discrete(max_nodes + 1)`: action `i < |V|` adds node `i`,
action `max_nodes` is STOP, and the indices in between are padding.

**Mask** (identical in training and evaluation):

* A node action is admissible iff the node is free, admissible, and lies
  within `radius` hops of `C` in the undirected DDG (any free node when `C`
  is empty). Its addition must also keep the port bound, convexity under
  contraction, and the area budget.
  `radius = 1` restricts the policy to connected CIs; `radius >= 2` admits
  disjoint CIs.
* STOP is admissible iff `C` is non-empty and committable, or `C` is
  non-empty and no node action remains.

Every candidate reached through admissible actions is committable
(`tests/test_env.py`).

**Reachability.** The mask requires the port bound after every growth step,
and port counts are not monotone in the candidate. A CI whose every growth
order passes through an intermediate set beyond the bound is not reachable.
Example: `x -> a, b, d -> c` with source `x`. `{a, b, c, d}` has
`|In| = |Out| = 1` and is valid at `(2, 1)`. But every growth path to it
passes through a set that violates the bound: from `{a}` the only neighbour
in the pool is `c`, and `{a, c}` has three inputs. The same holds from `{b}`
and from `{d}`, and `{c}` alone already has three inputs.

**STOP** commits `C`. This fixes its nodes, contracts it for later
convexity checks, and charges its area if its pattern is new.

**Reward** (`RewardConfig`):

* On a commit: `commit_coef * [log S_after - log S_before]`.
* On a growth step:
  `(shape_coef / |V|) * [Phi(C_{t+1}) - Phi(C_t)]`, where
  `Phi(C) = (|C| - cp(C)) - area_price * A(C) / A_budget`. The critical path
  is taken before rounding.
* When a candidate is committed, discarded or abandoned, its accumulated
  potential is charged back: `(shape_coef / |V|) * [0 - Phi(C)]`.

The growth payments of every candidate sum to zero. The episode
return is exactly `commit_coef * log S_final` (`tests/test_env.py` checks
this on every episode). The return is undiscounted, and PPO uses `gamma = 1`.
`area_price` is 0 in the reference configuration.

**End of episode**:

* **Terminated.** The pool of free admissible nodes is exhausted, or no
  admissible action remains (an absorbing dead end).
* **Truncated.** One of these limits is reached:
  * `max_steps_factor * |V|` decisions (default 10);
  * `clip(ceil(12 * log2 max(4, |V|)), 6, 70)` consecutive decisions without
    progress (the potential not increasing);
  * 4 consecutive states with a non-empty candidate and no node action;
  * `3 * |V|` refused actions.

A non-empty committable candidate is committed, and paid, when the pool runs
out or the episode ends. Otherwise it is discarded. The truncation limits
bound episodes of callers that ignore the mask.

**Off-mask actions** (a caller that ignores the mask). None of them leads to
an invalid commit:

* A node action outside the frontier changes nothing.
* A growth that breaks the port bound or the area budget, or places a fixed
  node between candidate nodes, is rolled back and counted as a violation.
* A growth that leaves a free node between candidate nodes is kept. The
  candidate is then not committable: it is discarded at STOP (a violation) or at the end of the
  episode.
* A STOP on an empty or uncommittable candidate counts as a violation.

**Reported result** (`episode_summary`). Reported quantities count only the
committed CIs with more than one node and a positive saving. A one-node CI
saves nothing, because every admissible operation takes one cycle. These
quantities are the speedup, the area used (once per pattern), the area
budget, `A_ref`, the number of patterns, the number of instances, and the
mean CI size.

## Observation

| Key | Shape | Contents |
|---|---|---|
| `node_status_vector` | `[max_nodes]` | status code per node (`ddg/status.py`, 11 codes: free, free input of `C`, in `C`, in `C` and output, in `C` and permanent output, committed, committed input of `C`, DDG source, DDG source input of `C`, inadmissible, inadmissible input of `C`) |
| `node_dynamic_feats` | `[max_nodes, 3]` | `[depth_C(v), l(v), height_C(v)] / S_V`: `depth_C(v)` (`height_C(v)`) is the longest latency-weighted path into (out of) `v` through `C`, and `S_V = max(1, ceil(cp(V)))`. |
| `graph_feats` | `[16]` | all in [0, 1]: `|V|/max_nodes`; pool fraction; area used including `C` over `max(A_budget, 1)`; saving ratio of `C`; `|In|`, `|permIn|`, `|Out|`, `|permOut|` over their bounds; frontier size over `|V|`; admissible node actions over `|V|`; no-progress counter over its limit; `|C|` over a target size; the best saving gain of one admissible node, squashed as `1 - 1/(1+g)`; area utilisation if `C` were committed (pattern-deduplicated); `cp(C)/S_V`; `ceil(cp(C))/S_V` |
| `action_mask` | `[max_nodes + 1]` | the mask |
| `graph_idx`, `num_nodes` | `[1]` | corpus index and `|V|` (the model reads the static graph from its store and checks `|V|`) |

## Network (`model/backbone.py`)

* **Input per node.** `[Embedding(11, 8) -> Linear(8, 96) -> LeakyReLU]`
  projected to 48 dimensions, concatenated with a 48-dimensional projection
  of `[l(v), a(v)]` (zero for inadmissible nodes) and the three dynamic
  features.
* **Encoder.** Four GIN layers with sum aggregation over predecessors (DGL
  passes messages along edge direction, i.e. data flow), a learnable
  epsilon, and the MLP `Linear-LayerNorm-ReLU-Linear`. Successors enter
  through the candidate-conditioned features and the graph embedding. Each
  layer is followed by LayerNorm, a residual to the encoder input, and
  LeakyReLU(0.2).
* **Graph embedding.** The mean over nodes, then
  `Linear-LeakyReLU-LayerNorm`, then concatenation of 48-dimensional
  projections of the embedding and of the 16 graph features.
* **Heads.**
  * Node logit: `q_head([h_v || g])`, with
    `q_head = Linear(192,192)-LeakyReLU-LayerNorm-Linear(192,1)`.
  * STOP logit: `q_head(Linear(96,192)(g))`.
  * Value: `Linear(96,96)-LeakyReLU-Linear(96,1)` of `g`.
  * Masked logits are set to -1e9.
* **Size.** 162,574 parameters.

All normalisation is per sample (LayerNorm), so `pi(a|s)` is a function of
`s` (`tests/test_model.py`).

## Training (`rl/train.py`, `configs/ppo.yaml`)

* **Algorithm.** PPO on RLlib 2.34 (old API stack) with `gamma = 1` and
  GAE `lambda = 1`. The learning rate is 1e-4, the clip parameter 0.3, and
  the value-function clip 20.
* **KL penalty.** A KL penalty with initial coefficient 0.2 and target
  0.015, adapted by RLlib.
* **Entropy coefficient.** `0.01 * (0.05)^(it/244)`, then 0.0005.
* **Batches and rollouts.** Gradient clipping at a global norm of 1. The
  train batch is 4096, the minibatch 512, with 2 SGD epochs. Rollouts use
  16 env runners and fragments of 8 steps (`truncate_episodes`).
* **Training cell.** `io = (4, 2)`, `area_factor = 0.4`, `radius = 1`.
  Graph sizes are drawn log-uniformly from [10, 2000] (`SizeSampler`,
  `per_size`, exponent 1).

## Evaluation (`rl/infer.py`)

Evaluation uses the same environment. Only the graph, the port bound and the
area factor change, and the policy takes the most probable admissible
action.
