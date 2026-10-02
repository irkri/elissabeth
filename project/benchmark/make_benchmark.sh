#!/usr/bin/env bash
# The canonical runs behind the two reports. Run from project/; each writes
# its CSV (+ .json) next to this script, then renders the HTML.
#
#   bash benchmark/make_benchmark.sh speed      # ~2 h on a 3090
#   bash benchmark/make_benchmark.sh stability  # ~45 min
#
# --memory-cap-gb keeps the speed run off memory other jobs on a shared GPU
# hold; cells beyond it report OOM. A new implementation is benchmarked with
# --impls eager compiled <name> (and --side-impls for the rank/settings runs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/richard/miniconda3/envs/viptorch/bin/python}
TAG=${TAG:-01}

case "${1:-}" in
  speed)
    $PY benchmark/benchmark.py --out benchmark/liss$TAG.csv \
        --memory-cap-gb 12 --cold-compile > benchmark/liss$TAG.out 2>&1
    $PY benchmark/plot_benchmark.py benchmark/liss$TAG.csv
    ;;
  stability)
    $PY benchmark/stability.py --out benchmark/stability$TAG.csv \
        > benchmark/stability$TAG.out 2>&1
    $PY benchmark/plot_stability.py benchmark/stability$TAG.csv
    ;;
  *)
    echo "usage: $0 speed|stability" >&2
    exit 1
    ;;
esac
