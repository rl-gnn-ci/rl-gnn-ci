# Benchmark DDGs

`benchmarks/dot/` holds the eight basic-block DDGs of the evaluation.
`benchmarks/manifest.json` indexes them, and `benchmarks/grid_240.yaml`
defines the scenario grid: 8 DDGs x 6 port bounds
{(2,1), (4,2), (6,3), (8,4), (10,5), (12,6)} x 5 area factors
{0.2, 0.4, 0.6, 0.8, 1.0}.

| Name | Suite | Function, block | File | \|V\| | \|E\| |
|---|---|---|---|---:|---:|
| Rijndael-AES | MiBench security/rijndael | `encrypt`, largest block | `dot/mibench/aes.dot` | 844 | 799 |
| SHA | MiBench security/sha | `sha_transform`, largest block | `dot/mibench/sha.dot` | 1215 | 1753 |
| Susan | MiBench automotive/susan | `susan_principle`, largest block | `dot/mibench/susan.dot` | 303 | 303 |
| MD5 | MiBench security/pgp | `MD5Transform`, block 0 | `dot/mibench/md5.dot` | 666 | 1060 |
| Blowfish | MiBench security/blowfish | `BF_encrypt`, largest block | `dot/mibench/blowfish.dot` | 372 | 402 |
| HadamardSAD8x8 | JM H.264 encoder (lencod) | `HadamardSAD8x8`, block 0 | `dot/jm_lencod/hadamardsad8x8.dot` | 384 | 575 |
| computeSATD | JM H.264 encoder (lencod) | `computeSATD`, largest block | `dot/jm_lencod/computesatd.dot` | 355 | 428 |
| MPEG2_Transform | MediaBench 2 mpeg2enc | `itransform`, largest block | `dot/mediabench/mpeg2_transform.dot` | 731 | 809 |

## Extraction

`cpp/ddg-pass/` is an LLVM (15 or later) new-pass-manager plug-in. It
writes one DOT file per basic block. Nodes are instructions, carrying the
IR text as label plus `op`, `kind` and I/O annotations. Edges are def-use
dependences inside the block.

```bash
cmake -S cpp/ddg-pass -B build/ddg -DLLVM_DIR=$(llvm-config --cmakedir) && cmake --build build/ddg
```

Settings for MiBench:

* **Front end:** `-std=gnu99 -O3 -fno-vectorize -fno-slp-vectorize -finline-functions -funroll-loops`, plus the forced includes and per-benchmark defines of
  `scripts/extract_mibench.sh`.
* **IR pipeline:** `opt -passes='default<O3>,cgscc(inline),globaldce'`.
* **Pass:** `-ddg-emit-input-boundary=true -ddg-split-bb=true -ddg-include-literals=false -ddg-constants-are-input=false -ddg-exclude-mem=true -ddg-exclude-calls=false -ddg-include-casts=true -ddg-gep-as-compute=true -ddg-annotate-io=true`.

The JM and MediaBench graphs use the same settings with
`-ddg-exclude-mem=false`.

`scripts/extract_mibench.sh <mibench-root> <DDGPass.so> <out>` extracts the
five MiBench kernels (sources: <https://github.com/embecosm/mibench>). It
compares each shipped graph with the extracted ones using
`python -m ci_ident.ddg.compare`, which checks node names and order, edges,
and the operation per node.
