#!/usr/bin/env bash
set -euo pipefail

PY=/data1/ycf/.conda/envs/3dgs/bin/python
CODE=/data1/ycf/academic/github_run/gaussian-splatting-datasetonline-20260908
STAGE=/data3/ycf/Data/MyData-datasetonline-stage
SOURCE=$STAGE/2026-09-03_datasetOnline_scene1_high_area1/input
RUNS=/data1/ycf/academic/github_run/RUN_RESULTS/2026-09-08_datasetOnline_scene1_high_area1
EXPECTED_CODE=${1:?Pass the verified code commit}
EXPECTED_DATA=110edd23446512f73ba6cac62e0f3b4de7d079b9
GPU=3
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

[[ $(git -C "$CODE" rev-parse HEAD) == "$EXPECTED_CODE" ]]
[[ $(git -C "$STAGE" rev-parse HEAD) == "$EXPECTED_DATA" ]]
[[ -z $(git -C "$CODE" status --porcelain) ]]
mkdir -p "$RUNS"
exec 9>"$RUNS/run.lock"
flock -n 9 || { echo 'This experiment is already running'; exit 1; }
[[ ! -e "$RUNS/status.txt" ]] || { echo 'Existing experiment: inspect before restarting'; exit 1; }
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED rc=%s time=%s\n" "$rc" "$(date -Is)" > "$RUNS/status.txt"; fi' EXIT
cd "$CODE"
printf 'RUNNING preflight %s\n' "$(date -Is)" > "$RUNS/status.txt"
printf 'code=%s\ndata=%s\ngpu=%s\ndata_device=cpu\npytorch_memory_fraction=0.5\n' \
  "$EXPECTED_CODE" "$EXPECTED_DATA" "$GPU" > "$RUNS/provenance.txt"

# Keep the shared GPU's PyTorch allocator below half of its physical memory.
run_python() {
  printf '%q ' "$PY" "$@" >> "$RUNS/commands.log"
  printf '\n' >> "$RUNS/commands.log"
  "$PY" -u -c 'import runpy, sys, torch; torch.cuda.set_per_process_memory_fraction(0.5, 0); script = sys.argv.pop(1); sys.argv[0] = script; runpy.run_path(script, run_name="__main__")' "$@"
}

check_gpu() {
  local free
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ "$free" -ge 14000 ]] || { echo "GPU $GPU has only $free MiB free; stopping before launch"; return 1; }
}

"$PY" - "$SOURCE" "$RUNS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from utils.colmap_seeds import build_colmap_seed_masks

source, runs = map(Path, sys.argv[1:])
extr = read_extrinsics_binary(str(source / 'sparse/0/images.bin'))
intr = read_intrinsics_binary(str(source / 'sparse/0/cameras.bin'))
names = sorted(image.name for image in extr.values())
assert len(names) == len(set(names)) == 99, len(names)
assert all(camera.model == 'PINHOLE' for camera in intr.values())
for name in names:
    with Image.open(source / 'images' / name) as image:
        assert image.size == (1600, 1175), (name, image.size)
seeds = build_colmap_seed_masks(str(source), target_hw=(1175, 1600), radius_px=15)
assert all(Path(name).stem in seeds and seeds[Path(name).stem].any() for name in names)
split = {'train': [n for i, n in enumerate(names) if i % 8],
         'test': [n for i, n in enumerate(names) if not i % 8]}
assert len(split['train']) == 86 and len(split['test']) == 13
(runs / 'frame_split.json').write_text(json.dumps(split, indent=2) + '\n')
files = [source / 'images' / n for n in names] + sorted((source / 'sparse/0').glob('*.bin'))
checksums = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(runs / 'input_sha256.json').write_text(json.dumps(checksums, indent=2) + '\n')
print('PRECHECK OK: 99 images; 86 train / 13 test; 1600x1175; 99 non-empty COLMAP seeds')
PY

check_gpu
run_python train.py -s "$SOURCE" -m "$RUNS/smoke_wm_v2" -r 1 --eval \
  --data_device cpu --disable_viewer --iterations 2 --test_iterations 2 --save_iterations 2 \
  --use_weight_map --weight_map_method wm_v2 --weight_map_mode online \
  --weight_map_alpha 1.0 --weight_map_beta 1.0 --weight_densify_thr 0.3 \
  2>&1 | tee "$RUNS/smoke_wm_v2.log"

for method in 3dgs_original wm_v2; do
  check_gpu
  extra=()
  if [[ "$method" == wm_v2 ]]; then
    extra=(--use_weight_map --weight_map_method wm_v2 --weight_map_mode online
      --weight_map_alpha 1.0 --weight_map_beta 1.0 --weight_densify_thr 0.3)
  fi
  printf 'RUNNING train %s %s\n' "$method" "$(date -Is)" > "$RUNS/status.txt"
  start=$(date +%s)
  run_python train.py -s "$SOURCE" -m "$RUNS/$method" -r 1 --eval \
    --data_device cpu --disable_viewer --iterations 30000 \
    --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 \
    --checkpoint_iterations 7000 15000 "${extra[@]}" \
    2>&1 | tee "$RUNS/${method}_train.log"
  printf '%s\n' "$(( $(date +%s) - start ))" > "$RUNS/${method}_train_seconds.txt"
  printf 'RUNNING evaluate %s %s\n' "$method" "$(date -Is)" > "$RUNS/status.txt"
  run_python render.py -m "$RUNS/$method" --iteration 30000 --eval --skip_train \
    2>&1 | tee "$RUNS/${method}_render.log"
  run_python metrics.py -m "$RUNS/$method" 2>&1 | tee "$RUNS/${method}_metrics.log"
  "$PY" - "$RUNS/$method/results.json" <<'PY'
import json, math, sys
with open(sys.argv[1]) as f:
    result = json.load(f)['ours_30000']
assert all(isinstance(result[k], (int, float)) and math.isfinite(result[k]) for k in ('PSNR', 'SSIM', 'LPIPS')), result
print('METRICS OK:', result)
PY
done

"$PY" "$CODE/scripts/summarize_datasetonline.py" "$RUNS"
printf 'COMPLETE %s\n' "$(date -Is)" > "$RUNS/status.txt"
