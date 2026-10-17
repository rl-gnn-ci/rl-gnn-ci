#!/usr/bin/env bash
# Re-extract the five MiBench DDGs of benchmarks/dot/mibench from the MiBench
# sources with the DDG pass (cpp/ddg-pass) and compare them with the shipped
# graphs (operations, edges and node order; LLVM value numbers in the labels
# may differ).
#
#   scripts/extract_mibench.sh <mibench-root> <DDGPass.so> <out-dir>
#
# MiBench: https://github.com/embecosm/mibench ; tested with clang/opt 18.
set -euo pipefail
ROOT=$(cd "$1" && pwd); PASS=$(readlink -f "$2"); OUT=$(mkdir -p "$3" && cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")/.." && pwd)
CFLAGS=(-std=gnu99 -O3 -fno-vectorize -fno-slp-vectorize -finline-functions -funroll-loops
        -include stdlib.h -include stdio.h -include string.h -include ctype.h
        -include unistd.h -include fcntl.h -w)
PREOPT='default<O3>,cgscc(inline),globaldce'
DDG=(-ddg-emit-input-boundary=true -ddg-split-bb=true -ddg-include-literals=false
     -ddg-constants-are-input=false -ddg-exclude-mem=true -ddg-exclude-calls=false
     -ddg-include-casts=true -ddg-gep-as-compute=true -ddg-annotate-io=true)

extract() {  # name shipped-dot extra-cflags sources...
  local name=$1 shipped=$2 extra=$3; shift 3
  local d="$OUT/$name"; mkdir -p "$d"; local bcs=()
  for s in "$@"; do
    clang "${CFLAGS[@]}" $extra -emit-llvm -c "$ROOT/$s" -o "$d/$(basename "$s" .c).bc"
    bcs+=("$d/$(basename "$s" .c).bc")
  done
  if [ ${#bcs[@]} -gt 1 ]; then llvm-link "${bcs[@]}" -o "$d/linked.bc"; else cp "${bcs[0]}" "$d/linked.bc"; fi
  opt -passes="$PREOPT" "$d/linked.bc" -o "$d/opt.bc"
  (cd "$d" && opt -load-pass-plugin="$PASS" -passes=ddgpass -disable-output -ddg-quiet=true \
       -ddg-out-dir="$d" "${DDG[@]}" "$d/opt.bc")
  python -m ci_ident.ddg.compare "$HERE/benchmarks/dot/mibench/$shipped" "$d"/*.dot || true
}

extract sha      sha.dot      "-DUNROLL_LOOPS -DNDEBUG" security/sha/sha.c security/sha/sha_driver.c
extract aes      aes.dot      "-DNDEBUG" security/rijndael/aes.c
extract blowfish blowfish.dot "-DNDEBUG" security/blowfish/bf_encrypt.c security/blowfish/bf_enc.c security/blowfish/bf_skey.c
extract md5      md5.dot      "-DNDEBUG" security/pgp/src/md5.c
extract susan    susan.dot    "-DNDEBUG -DPPC -std=gnu89" automotive/susan/susan.c
