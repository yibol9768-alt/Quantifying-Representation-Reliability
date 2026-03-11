#!/bin/bash
# ================================================================
# Fusion Experiments Runner
# ================================================================
# 系统融合实验运行脚本
# 用法: bash experiments/run_fusion_experiments.sh

set -e  # 遇到错误立即退出

# ------------------------------------------------
# 1. 初始化
# ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 加载配置
source "${SCRIPT_DIR}/config.sh"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "========================================"
echo "Starting Fusion Experiments"
echo "========================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "========================================"
echo ""

# 检查模型和数据是否已下载
echo "[Step 0] Checking models and data..."
if [ ! -d "${STORAGE_DIR}/models" ]; then
    echo "ERROR: Models directory not found at ${STORAGE_DIR}/models"
    echo "Please run: python download_models.py --models --storage_dir ${STORAGE_DIR}"
    exit 1
fi

if [ ! -d "${STORAGE_DIR}/data/${DATASET}" ]; then
    echo "ERROR: Dataset ${DATASET} not found at ${STORAGE_DIR}/data/${DATASET}"
    echo "Please run: python download_models.py --${DATASET} --storage_dir ${STORAGE_DIR}"
    exit 1
fi
echo "✓ Models and data found"
echo ""

# 创建实验日志摘要文件
SUMMARY_FILE="${RESULTS_DIR}/experiment_summary.txt"
echo "Fusion Experiments Summary - $(date)" > "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# ------------------------------------------------
# 2. 定义运行函数
# ------------------------------------------------
run_experiment() {
    local dataset=$1
    local num_models=$2
    local models=$3
    local method=$4
    local log_file="${LOG_DIR}/exp_${dataset}_${num_models}models_${method}.log"

    echo ">>> Running: ${dataset} | ${num_models} models [${models}] | ${method}"
    echo ">>> Log: ${log_file}"

    # 运行实验
    python main.py \
        --dataset "${dataset}" \
        --model fusion \
        --fusion_method "${method}" \
        --fusion_models "${models}" \
        --storage_dir "${STORAGE_DIR}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --cache_dtype "${CACHE_DTYPE}" \
        --seed "${SEED}" \
        --results_dir "${RESULTS_DIR}" \
        2>&1 | tee "${log_file}"

    # 检查是否成功
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Success: ${dataset} | ${num_models} models + ${method}" | tee -a "$SUMMARY_FILE"
        return 0
    else
        echo "✗ Failed: ${dataset} | ${num_models} models + ${method}" | tee -a "$SUMMARY_FILE"
        return 1
    fi
}

run_single_model() {
    local dataset=$1
    local model=$2
    local log_file="${LOG_DIR}/exp_${dataset}_1model_${model}.log"

    echo ">>> Running: ${dataset} | Single model ${model}"
    echo ">>> Log: ${log_file}"

    # 运行单模型实验
    python main.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --storage_dir "${STORAGE_DIR}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --cache_dtype "${CACHE_DTYPE}" \
        --seed "${SEED}" \
        --results_dir "${RESULTS_DIR}" \
        2>&1 | tee "${log_file}"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Success: ${dataset} | Single model ${model}" | tee -a "$SUMMARY_FILE"
        return 0
    else
        echo "✗ Failed: ${dataset} | Single model ${model}" | tee -a "$SUMMARY_FILE"
        return 1
    fi
}

# ------------------------------------------------
# 3. 运行实验
# ------------------------------------------------

# 计算总实验数
TOTAL_DATASETS=${#CLIP_DATASETS[@]}
TOTAL_SINGLE_MODELS=1  # 只跑CLIP作为单模型baseline
TOTAL_MODEL_COMBOS=10   # 1-10个模型
TOTAL_METHODS=${#ALL_METHODS[@]}
TOTAL_EXPERIMENTS=$((TOTAL_DATASETS * (TOTAL_SINGLE_MODELS + TOTAL_MODEL_COMBOS * TOTAL_METHODS)))

echo "========================================"
echo "Experiment Plan"
echo "========================================"
echo "Datasets: ${TOTAL_DATASETS} (${CLIP_DATASETS[*]})"
echo "Model counts: 1-10"
echo "Fusion methods: ${TOTAL_METHODS} (${ALL_METHODS[*]})"
echo "Total experiments: ${TOTAL_EXPERIMENTS}"
echo "========================================"
echo ""

# ========== Phase 1: 单模型 Baseline ==========
echo ""
echo "========================================"
echo "Phase 1: Single Model Baselines (CLIP)"
echo "========================================"

for dataset in "${CLIP_DATASETS[@]}"; do
    echo ""
    echo "--- Dataset: ${dataset} ---"
    run_single_model "$dataset" "clip"
    echo ""
done

# ========== Phase 2: 逐步增加模型数量 ==========
echo ""
echo "========================================"
echo "Phase 2: Varying Number of Models"
echo "========================================"

# 定义模型数量和对应组合
declare -a model_combinations=(
    "1:${MODELS_1}"
    "2:${MODELS_2}"
    "3:${MODELS_3}"
    "4:${MODELS_4}"
    "5:${MODELS_5}"
    "6:${MODELS_6}"
    "7:${MODELS_7}"
    "8:${MODELS_8}"
    "9:${MODELS_9}"
    "10:${MODELS_10}"
)

# 遍历每个数据集
for dataset in "${CLIP_DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Dataset: ${dataset}"
    echo "========================================"

    # 对每种模型数量，运行所有融合方法
    for combo in "${model_combinations[@]}"; do
        num_models="${combo%%:*}"
        models="${combo#*:}"

        echo ""
        echo "--- ${dataset}: ${num_models} model(s) ---"

        for method in "${ALL_METHODS[@]}"; do
            run_experiment "$dataset" "$num_models" "$models" "$method"
            echo ""
        done
    done
done

# ------------------------------------------------
# 4. 收集结果
# ------------------------------------------------
echo ""
echo "[Step 4] Collecting results..."
python "${SCRIPT_DIR}/collect_results.py" \
    --results_dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/results_table.csv" \
    --markdown "${RESULTS_DIR}/results_summary.md"

echo ""
echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo "Results saved to: ${RESULTS_DIR}"
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "Experiment breakdown:"
echo "  - Datasets: ${TOTAL_DATASETS}"
echo "  - Single model experiments: ${TOTAL_DATASETS}"
echo "  - Fusion experiments: $((TOTAL_DATASETS * TOTAL_MODEL_COMBOS * TOTAL_METHODS))"
echo "  - Total: ${TOTAL_EXPERIMENTS}"
echo "========================================"
