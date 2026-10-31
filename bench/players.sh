#!/bin/bash
# Live players per GPU, one process per player (how Waypoint is hosted without a shared engine):
# N = 1, 2, 4, ... concurrent players.py workers until a player falls below FLOOR fps or a worker fails
# (out of memory). One JSON line per N in OUT/players.jsonl.
#
#   players.sh MODEL_DIR SEEDS_DIR OUT [FLOOR]          CRUKX=1 to apply crukx_waypoint in every worker
set -u
MODEL=$1; SEEDS=($2/*.jpg); O=$3; FLOOR=${4:-10}; mkdir -p $O
cd "$(dirname "$0")"
PY=${PYTHON:-python}; FLAG=${CRUKX:+--crukx}
for N in 1 2 4 6 8 12 16 24 32 48; do
  B=$O/barrier$N; rm -rf $B; mkdir -p $B
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $O/mem$N.txt & SMI=$!
  PIDS=()
  for ((i = 0; i < N; i++)); do
    timeout 900 $PY players.py --model-dir $MODEL --seed-image ${SEEDS[$((i % ${#SEEDS[@]}))]} --barrier $B \
      --rank $i --n $N --out $O/n${N}_r$i.json $FLAG > $O/n${N}_r$i.log 2>&1 & PIDS+=($!)
  done
  FAIL=0; for p in "${PIDS[@]}"; do wait $p || FAIL=1; done; kill $SMI
  $PY - $O $N $FAIL >> $O/players.jsonl <<'EOF'
import json, sys
from pathlib import Path
o, n, fail = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rs = [json.loads(p.read_text()) for p in sorted(o.glob(f'n{n}_r*.json'))]
mem = [int(x) for x in (o / f'mem{n}.txt').read_text().split() if x.isdigit()]
fps = [r['fps_mean'] for r in rs]
print(json.dumps({'players': n, 'failed': bool(fail) or len(rs) < n, 'fps_per_player_min': min(fps, default=0),
                  'aggregate_fps': round(sum(fps), 1), 'gpu_mem_peak_gb': round(max(mem, default=0) / 1024, 1),
                  'gpu': rs[0]['gpu'] if rs else None, 'crukx': rs[0]['crukx'] if rs else None}))
EOF
  LAST=$(tail -1 $O/players.jsonl); echo "$LAST"; rm -rf $B
  echo "$LAST" | $PY -c "import json,sys;d=json.load(sys.stdin);sys.exit(0 if not d['failed'] and d['fps_per_player_min']>=$FLOOR else 1)" || break
done
