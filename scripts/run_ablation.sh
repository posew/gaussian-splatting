#!/bin/bash
#
# run_ablation.sh - 水下3DGS消融实验脚本
#
# 按照 CHANGE_PLAN.md 消融实验设计，依次运行 A~F 六组实验，
# 每组：训练 → 渲染 → 计算指标(PSNR/SSIM/LPIPS)，最终汇总输出。
#
# 用法:
#   bash scripts/run_ablation.sh -s /path/to/dataset
#   bash scripts/run_ablation.sh -s /path/to/dataset --iterations 7000 --gpu 1
#   bash scripts/run_ablation.sh -s /path/to/dataset --skip_train   # 只跑渲染+指标
#

set -euo pipefail

# ──────────────── 默认参数 ────────────────────────────────────────────
DATA_SOURCE=""
OUTPUT_ROOT="output/ablation"
ITERATIONS=30000
TEST_ITERATIONS="7000 30000"
SAVE_ITERATIONS="7000 30000"
GPU_ID=0
SKIP_TRAIN=false
SKIP_RENDER=false
SKIP_METRICS=false
EVAL_ITER=-1  # -1 表示使用最后保存的迭代

# ──────────────── 解析命令行参数 ──────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source)        DATA_SOURCE="$2"; shift 2 ;;
        -o|--output)        OUTPUT_ROOT="$2"; shift 2 ;;
        --iterations)       ITERATIONS="$2"; shift 2 ;;
        --test_iterations)  TEST_ITERATIONS="$2"; shift 2 ;;
        --gpu)              GPU_ID="$2"; shift 2 ;;
        --skip_train)       SKIP_TRAIN=true; shift ;;
        --skip_render)      SKIP_RENDER=true; shift ;;
        --skip_metrics)     SKIP_METRICS=true; shift ;;
        --eval_iter)        EVAL_ITER="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/run_ablation.sh -s /path/to/dataset [options]"
            exit 1
            ;;
    esac
done

if [[ -z "$DATA_SOURCE" ]]; then
    echo "Error: --source (-s) is required."
    echo "Usage: bash scripts/run_ablation.sh -s /path/to/dataset"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${OUTPUT_ROOT}/summary_${TIMESTAMP}.txt"
mkdir -p "$OUTPUT_ROOT"

echo "=============================================="
echo "  水下3DGS消融实验"
echo "  数据集:    $DATA_SOURCE"
echo "  输出目录:  $OUTPUT_ROOT"
echo "  迭代次数:  $ITERATIONS"
echo "  GPU:       $GPU_ID"
echo "  时间戳:    $TIMESTAMP"
echo "=============================================="

# ──────────────── 实验组定义 ──────────────────────────────────────────
#
# 格式: "实验名|训练脚本|额外参数"
#
EXPERIMENTS=(
    "A_Baseline|train.py|"
    "B_WeightLoss|train_weighted.py|--use_weight_map --weight_densify_thr 0"
    "C_WeightDensify|train_weighted.py|--use_weight_map --weight_densify_thr 0.3"
    "D_WeightPrune|train_weighted.py|--use_weight_map --weight_densify_thr 0.3 --weight_prune_thr 0.2"
    "E_NormalUniform|train_weighted.py|--use_weight_map --weight_densify_thr 0.3 --weight_prune_thr 0.2 --use_normal_loss --lambda_normal 0.01"
)

# ──────────────── 运行函数 ────────────────────────────────────────────

run_experiment() {
    local exp_name="$1"
    local train_script="$2"
    local extra_args="$3"
    local model_path="${OUTPUT_ROOT}/${exp_name}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  实验组: $exp_name"
    echo "  脚本:   $train_script"
    echo "  参数:   $extra_args"
    echo "  模型:   $model_path"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── 训练 ──
    if [[ "$SKIP_TRAIN" == false ]]; then
        echo "[$(date +%H:%M:%S)] 开始训练 ${exp_name} ..."
        local start_time=$(date +%s)

        python "$train_script" \
            -s "$DATA_SOURCE" \
            -m "$model_path" \
            --iterations "$ITERATIONS" \
            --test_iterations $TEST_ITERATIONS \
            --save_iterations $SAVE_ITERATIONS \
            --eval \
            $extra_args \
            2>&1 | tee "${model_path}/train_log.txt"

        local end_time=$(date +%s)
        local duration=$(( end_time - start_time ))
        echo "[$(date +%H:%M:%S)] 训练完成, 耗时 ${duration}s"
        echo "${exp_name}_train_time=${duration}s" >> "$SUMMARY_FILE"
    else
        echo "[SKIP] 跳过训练"
    fi

    # ── 渲染 ──
    if [[ "$SKIP_RENDER" == false ]]; then
        echo "[$(date +%H:%M:%S)] 开始渲染 ${exp_name} ..."
        python render.py \
            -m "$model_path" \
            --iteration "$EVAL_ITER" \
            --skip_train \
            2>&1 | tee "${model_path}/render_log.txt"
        echo "[$(date +%H:%M:%S)] 渲染完成"
    else
        echo "[SKIP] 跳过渲染"
    fi

    # ── 指标计算 ──
    if [[ "$SKIP_METRICS" == false ]]; then
        echo "[$(date +%H:%M:%S)] 计算指标 ${exp_name} ..."
        python metrics.py \
            -m "$model_path" \
            2>&1 | tee "${model_path}/metrics_log.txt"
        echo "[$(date +%H:%M:%S)] 指标计算完成"
    else
        echo "[SKIP] 跳过指标计算"
    fi
}

# ──────────────── 执行所有实验 ────────────────────────────────────────

echo "" >> "$SUMMARY_FILE"
echo "消融实验报告 - $TIMESTAMP" >> "$SUMMARY_FILE"
echo "数据集: $DATA_SOURCE" >> "$SUMMARY_FILE"
echo "迭代: $ITERATIONS" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name train_script extra_args <<< "$exp_entry"
    run_experiment "$exp_name" "$train_script" "$extra_args"
done

# ──────────────── 汇总结果 ────────────────────────────────────────────

echo ""
echo "=============================================="
echo "  汇总所有实验指标"
echo "=============================================="

{
    echo ""
    echo "==================== 实验结果汇总 ===================="
    printf "%-20s  %8s  %8s  %8s\n" "Experiment" "PSNR" "SSIM" "LPIPS"
    echo "------------------------------------------------------"
} >> "$SUMMARY_FILE"

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name _ _ <<< "$exp_entry"
    results_file="${OUTPUT_ROOT}/${exp_name}/results.json"

    if [[ -f "$results_file" ]]; then
        psnr_val=$(python3 -c "
import json, sys
with open('$results_file') as f:
    d = json.load(f)
for method in d.values():
    print(f\"{method.get('PSNR', 'N/A'):.4f}\")
    break
" 2>/dev/null || echo "N/A")

        ssim_val=$(python3 -c "
import json, sys
with open('$results_file') as f:
    d = json.load(f)
for method in d.values():
    print(f\"{method.get('SSIM', 'N/A'):.4f}\")
    break
" 2>/dev/null || echo "N/A")

        lpips_val=$(python3 -c "
import json, sys
with open('$results_file') as f:
    d = json.load(f)
for method in d.values():
    print(f\"{method.get('LPIPS', 'N/A'):.4f}\")
    break
" 2>/dev/null || echo "N/A")

        printf "%-20s  %8s  %8s  %8s\n" "$exp_name" "$psnr_val" "$ssim_val" "$lpips_val" | tee -a "$SUMMARY_FILE"
    else
        printf "%-20s  %8s  %8s  %8s\n" "$exp_name" "N/A" "N/A" "N/A" | tee -a "$SUMMARY_FILE"
    fi
done

echo "------------------------------------------------------" | tee -a "$SUMMARY_FILE"

# ── 统计 Gaussian 数量 ──
{
    echo ""
    echo "==================== Gaussian 数量 ===================="
    printf "%-20s  %12s\n" "Experiment" "Num Gaussians"
    echo "------------------------------------------------------"
} >> "$SUMMARY_FILE"

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name _ _ <<< "$exp_entry"

    ply_file=$(find "${OUTPUT_ROOT}/${exp_name}/point_cloud" -name "point_cloud.ply" 2>/dev/null | sort -r | head -1)
    if [[ -n "$ply_file" ]]; then
        num_points=$(python3 -c "
from plyfile import PlyData
p = PlyData.read('$ply_file')
print(p.elements[0].count)
" 2>/dev/null || echo "N/A")
        printf "%-20s  %12s\n" "$exp_name" "$num_points" | tee -a "$SUMMARY_FILE"
    else
        printf "%-20s  %12s\n" "$exp_name" "N/A" | tee -a "$SUMMARY_FILE"
    fi
done

echo "" | tee -a "$SUMMARY_FILE"
echo "完整报告已保存至: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo ""
echo "全部消融实验完成!"
