# Formulation

This document specifies what the code computes. Paper sections are given in
*italics*.

## DDGs and cost model (*Problem Formulation*)

* **Graphs.** A DDG is one basic block in DOT format (`ddg/graph.py`). Node
  order is file order, and it indexes both the action space and the GNN
  input. Each corpus manifest records a digest of the order, which the
  graph store checks (`model/graph_store.py`).
* **Operations** (`costs.py`). The operator library defines the hardware
  latency `l_hw(v)` and area `a(v)` of the admissible operations:
  * add, fadd, sub, fsub, neg: 0.31 / 0.10
  * mul, fmul: 0.96 / 0.98
  * fma, mac: 1.00 / 1.00
  * shifts and rotates: 0.23 / 0.12
  * bitwise logic: 0.03 / 0.02
  * comparisons: 0.10 / 0.04
  * casts: 0.03 / 0.02
  * getelementptr: 0.31 / 0.12

  Memory, control, call, vector, debug, `select`, division and remainder
  operations are inadmissible, and so are graph sources (block inputs).
  Every node takes one software cycle, so `T_sw = |V|`.
* **Latency and saving.** `cp(S)` is the longest latency-weighted path
  inside `S`, and `l_CI(S) = ceil(cp(S))`. Cycle counts and area
  comparisons use a tolerance of 1e-9 (`costs.cycles`, `costs.EPS`). The
  per-instance saving is `Delta(S) = |S| - l_CI(S)` and the block speedup
  is `T_sw / (T_sw - sum Delta)`.
* **Ports.** `In(S)` is the set of distinct nodes outside `S` with a
  successor in `S`. `Out(S)` is the set of nodes of `S` with a successor
  outside `S` or with no successor.
* **Convexity.** Convexity is taken with every committed CI contracted to
  one node, so a committed set is acyclic as a whole.
* **Patterns.** Two CIs are instances of one pattern iff their
  direction-aware 1-WL signatures agree (*WL-signature algorithm*, K = 5,
  `ddg/signature.py`). The signature is the multiset of final node colours
  together with the multiset of (producer colour, consumer colour) pairs of
  the CI's edges. Area is charged once per pattern; the budget is
  `alpha * A_ref`, where `A_ref` is the area of all CI-legal nodes.

## MDP (*MDP Formulation*, `env/env.py`)

* **State.**
  * The status of every node, one of 11 codes (`ddg/status.py`): free,
    in the candidate or committed, refined by the node's port role (input,
    output or permanent output of the candidate, block input, inadmissible).
  * The candidate `C`.
  * The global features `f_t`, 12 values in [0, 1]:
    * `|V|/max_nodes`;
    * the free-pool fraction;
    * area utilisation (candidate included);
    * `rho(C)`;
    * `|In|`, `|permIn|`, `|Out|`, `|permOut|` over their bounds;
    * frontier size over `|V|`;
    * admissible node actions over `|V|`;
    * `h(|C|)`;
    * committed speedup as `1 - 1/S`.
* **Actions.** `V ∪ {STOP}`: `Discrete(max_nodes + 1)`, with STOP the last
  index.
* **FeasGrow(C, v)** holds when:
  * `v` is free and CI-legal;
  * `v` is adjacent to `C`, or `C` is empty;
  * `C + v` is convex and within the port bound.
* **FeasCommit(C)** holds when:
  * `C` is non-empty, convex and within the port bound;
  * the area of a new pattern fits the remaining budget (a pattern already
    committed costs no area);
  * `Delta(C) >= 0`.
* **Transitions.**
  * A legal growth adds the node.
  * An illegal node action leaves the candidate unchanged.
  * A rolled-back growth adds the node to the refusal set of the current
    candidate, which is cleared when the candidate is committed or
    discarded.
  * STOP commits a feasible candidate, and discards an infeasible one.
  * An empty STOP changes nothing.
* **Termination.**
  * The episode terminates when the free pool is exhausted.
  * It also terminates when the candidate is empty and no free node can
    start a feasible CI, that is, when every free node has permanent ports
    beyond the bound, or an area beyond the remaining budget, or (with the
    inference mask) no admissible action.
  * It is truncated after `10 |V|` decisions.
  * A candidate still open at the end is committed if feasible and
    discarded otherwise.
* **Discount.** `gamma` is set by the variant configuration.

## Reward (*Reward Design*)

| Term | When | Value |
|---|---|---|
| commit | a commit (STOP, or at the end of the episode) | `w_perf [log S_{t+1} - log S_t] + (w_size/N) h(|C|)` |
| extension | legal growth | `w_ext * eta` |
| shaping | every transition | `(w_shp/N) [gamma Phi(s') - Phi(s)]` |
| discard | a discard (STOP, or at the end of the episode) | minus the candidate's extension rewards, each compounded by `1/gamma` per step since it was paid |
| illegal node action | | `w_invalid / N` |
| infeasible STOP | | `w_bad_stop / N` |
| empty STOP | | `w_empty_stop / N` |

The terms in the table use the following definitions:

* `N = |V|`, and `S` is the block speedup of the committed set.
* `eta = max(G(C'), 0) * (1 - h(|C'|)) * 1[FeasCommit(C')]` for the grown
  candidate `C'`.
* `h(k) = min(1, k / k_target)`, where `k_target` is the smaller of
  `N_in + N_out + 1` and the number of median-area free nodes that fit the
  remaining budget, and at least 2.
* `G = rho = Delta / l_SW` (Base, NoMask), or `G = Delta` (AbsGain).
* `Phi(C) = G(C) - lambda_io d_io(C) - lambda_conv d_conv(C)`, and
  `Phi = 0` without a candidate and in terminal states:
  * `d_io = max(|In| - N_in, 0) + max(|Out| - N_out, 0)`;
  * `d_conv` is the fraction of free nodes lying between candidate nodes.
* **NoShape:** `eta = 1`, `w_shp = 0` and `w_size = 0`.

Over an episode, the commit terms sum to `w_perf log S_final`. The
discounted shaping terms of a terminated episode sum to zero. A discarded
candidate's extension rewards have zero discounted value.

## Action masks (*Constrained Action Masking*)

| | training (`mode="train"`) | inference (`mode="inference"`) |
|---|---|---|
| Base, NoShape, AbsGain | Free, CI-legal nodes adjacent to `C` (all free CI-legal nodes when `C` is empty), not refused for `C`. After a growth, a permanent port beyond the bound, a fixed node between candidate nodes, or an area overflow is rolled back. The regular port bound and convexity with respect to free nodes are enforced at commit. | The training mask filtered by FeasGrow. An area overflow is rolled back. |
| NoMask | every node and STOP | every node and STOP |

In the masked variants STOP is admissible iff `C` is non-empty. The area
check during growth counts the candidate's own area.

## Network (*Graph Encoder*, *RL Parameterizations*; `model/network.py`)

* **Node input.** `[status embedding (8) || opcode embedding (8) || l_hw || a]`,
  where `l_hw` and `a` are zero for inadmissible nodes. The input is
  linearly projected to the hidden width `H`, with ReLU.
* **Encoder.**
  * `L` layers of GIN, GATv2 or GCN, with messages following the DDG edges.
  * Each layer is followed by LayerNorm, a residual to the projected input,
    and ReLU.
  * GIN uses the MLP `Linear-LayerNorm-ReLU-Linear` with a learnable
    epsilon.
  * GATv2 concatenates its heads. GCN uses symmetric normalisation. Both
    use self-loops.
* **Pooling.** Mean, sum or attention (softmax-gated sum) pooling gives `g`,
  and `g' = [g || ReLU(W_f f_t)]`.
* **Action features.** `u(v) = [z_v || g']` and `u(STOP) = [e_STOP || g']`,
  where `e_STOP` is a learned vector. One MLP,
  `Linear(3H, 2H) - ReLU - LayerNorm - Linear(2H, 1)`, scores all actions.
* **PPO.**
  * Logits: `MLP_pi(u) / tau`.
  * Value: `MLP_V(g')`, where `MLP_V` is `Linear(2H, 2H) - ReLU - LayerNorm - Linear(2H, 1)`.
* **DQN.**
  * `Q(a) = V(g') + A(u(a)) - mean over feasible a' of A(u(a'))`.
  * Optional NoisyNet: factorised Gaussian noisy layers in both streams,
    with noise in training mode only.
* **Masking.** Inadmissible actions score `-1e9`.

All normalisation is LayerNorm, and the network has no dropout.

## Inference (`rl/infer.py`)

The network takes the highest-scoring admissible action under the inference
mask of its variant until the episode ends.
