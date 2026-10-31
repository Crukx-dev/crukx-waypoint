#!/bin/bash
# The full GPU matrix on one machine: stock Waypoint vs Crukx (if crukx_waypoint is installed), measured and recorded.
#
#   run_matrix.sh MODELS_DIR OUT      MODELS_DIR holds model720/ and model360/ (see README)
#
# Stages: play (human-like, Biome worlds, video), fps and latency (720p), fidelity (isolated + history
# against an FP32 reference), players per GPU (one process each), batching probe; with Crukx also the
# shared engine and the two-player fork demo. Each stage prints "<name> exit <code>".
set -u
M=$(realpath $1); O=$(realpath -m $2); mkdir -p $O; cd "$(dirname "$0")"
B=bench; PY=${PYTHON:-python}
export TORCHINDUCTOR_CACHE_DIR=$O/kcache/inductor TRITON_CACHE_DIR=$O/kcache/triton TRITON_CACHE_AUTOTUNING=1
step() { N=$1; shift; "$@" > $O/$N.log 2>&1; echo "$N exit $?"; }

[ -d seeds ] || bash fetch_seeds.sh
CONFIGS=stock; $PY -c "import crukx_waypoint" 2>/dev/null && CONFIGS="stock crukx"
flag() { [ $1 = crukx ] && echo --crukx; }
DEMO="seeds/neon_alley_night.jpg seeds/highland_castle_loch.jpg"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $O/env.txt

for RES in 720 360; do for C in $CONFIGS; do
  step play_${RES}_$C $PY $B/play.py --model-dir $M/model$RES --seeds $DEMO --tag ${RES}_$C --out $O/play $(flag $C)
done; done
for C in $CONFIGS; do
  step fps_720_$C $PY $B/fps.py --model-dir $M/model720 --out $O/fps_720_$C.json $(flag $C)
  step latency_720_$C $PY $B/latency.py --model-dir $M/model720 --out $O/latency_720_$C.json $(flag $C)
done
[ -f $O/golden_720.pt ] || step golden_720 $PY $B/fidelity.py golden --model-dir $M/model720 --golden $O/golden_720.pt
for T in isolated history; do for C in $CONFIGS; do
  step fidelity_720_${T}_$C $PY $B/fidelity.py $T --model-dir $M/model720 --golden $O/golden_720.pt \
    --out $O/fidelity_720_${T}_$C.json $(flag $C)
done; done
step players_720 bash $B/players.sh $M/model720 $PWD/seeds $O/players_720
for RES in 720 360; do
  step batch_probe_$RES $PY $B/batch_probe.py --model-dir $M/model$RES --seed-image seeds/neon_alley_night.jpg \
    --out $O/batch_probe_$RES.json
done
if [[ $CONFIGS == *crukx* ]]; then
  for RES in 720 360; do
    step sessions_$RES $PY $B/sessions.py --model-dir $M/model$RES --seeds $DEMO --out $O/sessions_$RES.json
  done
  step fork $PY $B/fork_demo.py --model-dir $M/model720 --seed-image seeds/neon_alley_night.jpg --out $O/fork
  rm -rf $O/fork/saves
fi
echo MATRIX_DONE
