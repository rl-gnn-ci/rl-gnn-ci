# Variants

Each variant is one file in `src/ci_ident/variants/`, loaded with
`ci_ident.variants.load_variant(name)`. A file holds the environment
configuration (variant, discount, reward weights) and the network
configuration.

| Variant | Mask | Reward | Encoder | Layers | Hidden | Pooling | gamma |
|---|---|---|---|---:|---:|---|---:|
| `dqn-base` | strict | eta, shaping, size term, `G = rho` | GIN | 4 | 128 | attention | 0.99 |
| `dqn-nomask` | none | as Base | GATv2, 4 heads | 3 | 96 | sum | 0.99 |
| `dqn-noshape` | strict | `eta = 1`, no shaping, no size term | GIN | 4 | 96 | mean | 0.99 |
| `dqn-absgain` | strict | eta, shaping, size term, `G = Delta` | GATv2, 2 heads, NoisyNet | 3 | 128 | attention | 0.95 |
| `ppo-base` | strict | as dqn-base | GATv2, 4 heads | 2 | 96 | attention | 0.995 |
| `ppo-nomask` | none | as Base | GCN | 3 | 160 | mean | 0.995 |
| `ppo-noshape` | strict | as dqn-noshape | GIN | 2 | 160 | sum | 0.995 |
| `ppo-absgain` | strict | as dqn-absgain | GIN | 2 | 160 | sum | 0.995 |

The head width is twice the hidden width.

## Reward weights

| Weight | default | `dqn-noshape` | `dqn-absgain` |
|---|---:|---:|---:|
| `w_perf` | 8.0 | 10.0 | 8.0 |
| `w_size` | 1.0 | (off) | 0.98852 |
| `w_ext` | 0.3 | 0.12089 | 0.57520 |
| `w_shp` | 0.1 | (off) | 0.31063 |
| `lambda_io` | 0.5 | (off) | 0.05985 |
| `lambda_conv` | 0.5 | (off) | 0.09070 |
| `w_invalid` | -0.2 | -0.2 | -0.2 |
| `w_bad_stop` | -0.5 | -1.0 | -0.5 |
| `w_empty_stop` | -1.0 | -1.0 | -1.0 |

The other six variants use the defaults. The full-precision values are in
the variant files.

## Learner settings of the reported networks

**DQN.** Dueling double DQN with prioritised replay (capacity 2e5).

| | range | `dqn-noshape` | `dqn-absgain` |
|---|---|---|---|
| PER `alpha` / `beta` | [0.46, 0.69] / [0.27, 0.45] | 0.456 / 0.367 | 0.551 / 0.447 |
| n-step | {1, 3} | 3 | 1 |
| target-network update | 4k-16k steps | 8000 | 8000 |
| learning rate (Adam, exponential decay) | [2.7e-5, 1.6e-4] | 6.05e-5, x0.9998 per 1024 steps | 1.57e-4, x0.9920 per 1024 steps |
| exploration | epsilon-greedy 1.0 -> 0.01-0.02, or NoisyNet | epsilon 1.0 -> 0.02 over 2e5 steps | epsilon 1.0 -> 0.02 over 1e4 steps, NoisyNet (sigma0 0.5) |
| train batch / training intensity | | 256 / 2 | 256 / 2 |
| loss / gradient clip | | Huber / 20 | Huber / none |

**PPO.**
* `gamma = 0.995`, GAE `lambda` in {0.98, 1.0};
* Adam with learning rate (1.8-6.7)e-4;
* clip ratio 0.15-0.20;
* KL coefficient 0.1-0.2;
* entropy coefficient decaying from 0.005-0.02 to 0.0001-0.005.

**Budget.** About 25 M environment steps per variant.

The networks compute the dueling combination over the feasible actions and
return masked Q-values; a DQN learner uses them without its own dueling head
(in RLlib: `dueling=False`, `hiddens=[]`).
