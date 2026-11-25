#!/bin/bash
# Latency per stage, cold start and peak RSS of the small-backbone candidates, under the lock:
#   python -m bench.heavy --lane cpu --need 30 --floor 12 -- bash bench/opt/run_profile_small.sh <rev> <out dir>
# Every backbone is loaded through PRONUNCIADA_BACKBONE (so the default's pre-optimized-graph load is
# not used by any of them, including "fp32-file", the reference for load times); "fp32" = the default.
set -u
REV="$1"; OUT="$2"
mkdir -p "$OUT"
P="python -m bench.opt.profile"
C() { python -m bench.opt.candidates spec "$1" --code "$REV"; }
log() { echo "$(date '+%H:%M:%S') load $(sysctl -n vm.loadavg) $*"; }
F32="{\"name\":\"fp32\",\"code\":\"$REV\",\"precision\":\"fp32\"}"
CF="$F32,$(C fp32-file),$(C int8-shipped-vs-fp32),$(C nb8-sym-b32),$(C nb8-sym-b128),$(C dq8-sym-ch-noqdq),$(C mix-A),$(C mix-B),$(C mix-C)"
log start
$P latency --configs "$CF" --langs en,de,ja,zh --threads 1,2,4 --n 10 --repeats 2 --out "$OUT/opt_profile_small.json"; log latency done
$P rss --configs "$CF" --langs en,zh --threads 1,4 --n 10 --out "$OUT/opt_rss_small.json"; log rss done
log all done
