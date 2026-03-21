#!/bin/bash
# ================================================================
# 一键运行完整实验流程
#
# 包含三个阶段：
#   阶段 1: 提取 14 个模型的冻结特征
#   阶段 2: 跑 9 种选择方法的排序
#   阶段 3: 按每种排序从 k=1 累加到 k=10 跑融合训练
#
# 用法：
#   STORAGE_DIR=/root/autodl-tmp/feature_workspace bash experiments/run_full_pipeline.sh
#
# 断点续跑：
#   脚本会自动跳过已完成的步骤（特征文件已存在/结果已存在）
# ================================================================

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?请设置 STORAGE_DIR，例如 export STORAGE_DIR=/root/autodl-tmp/feature_workspace}"
FEATURE_DIR="${STORAGE_DIR}/data/features"
SELECTION_DIR="${STORAGE_DIR}/results/selection_comparison"
FUSION_DIR="${STORAGE_DIR}/results/fusion_scaling"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
VAL_RATIO="${VAL_RATIO:-0.2}"
VAL_SEED="${VAL_SEED:-42}"
DELETE_FEATURES_AFTER_SELECTION="${DELETE_FEATURES_AFTER_SELECTION:-0}"
FUSION_CLEANUP_CACHE="${FUSION_CLEANUP_CACHE:-0}"

# 14 个模型（排除 clip_large）
MODELS="clip,dino,mae,siglip,convnext,data2vec,vit,swin,beit,dinov2_large,dinov2_small,mae_large,deit_small,resnet50"
# country211 is much slower; keep it optional so the main pipeline can finish first.
DATASETS="${DATASETS:-stl10,pets,eurosat,dtd,gtsrb,svhn}"
DATASETS_TO_RUN="${DATASETS}"
MAX_K=10

echo "================================================================"
echo "  完整实验流程"
echo "================================================================"
echo "  存储目录:   ${STORAGE_DIR}"
echo "  模型数:     14"
echo "  数据集:     $(echo "${DATASETS_TO_RUN}" | tr ',' '\n' | wc -l | tr -d ' ')"
echo "  最大融合数: ${MAX_K}"
echo "  Epochs:     ${EPOCHS}"
echo "  Val Ratio:  ${VAL_RATIO}"
echo "  Val Seed:   ${VAL_SEED}"
echo "  阶段2后删特征: ${DELETE_FEATURES_AFTER_SELECTION}"
echo "  融合后删cache: ${FUSION_CLEANUP_CACHE}"
echo "================================================================"
echo ""

# ================================================================
# 阶段 1: 提取特征
# ================================================================
echo "================================================================"
echo "  阶段 1/3: 提取冻结特征"
echo "================================================================"

python experiments/extract_features.py \
    --storage_dir "${STORAGE_DIR}" \
    --output_dir "${FEATURE_DIR}" \
    --datasets "${DATASETS}" \
    --models "${MODELS}" \
    --batch_size "${BATCH_SIZE}" \
    --validation_ratio "${VAL_RATIO}" \
    --split_seed "${VAL_SEED}" \
    --device "${DEVICE}"

echo ""
echo "[阶段 1 完成] 特征已保存到 ${FEATURE_DIR}"
echo ""

# ================================================================
# 阶段 2: 选择方法排序
# ================================================================
echo "================================================================"
echo "  阶段 2/3: 运行选择方法对比"
echo "================================================================"

mkdir -p "${SELECTION_DIR}"

python3 - "${FEATURE_DIR}" "${SELECTION_DIR}" "${MAX_K}" "${DATASETS_TO_RUN}" <<'PYTHON_SCRIPT'
import json
import os
import subprocess
import sys

feature_dir = sys.argv[1]
selection_dir = sys.argv[2]
max_k = sys.argv[3]
datasets = sys.argv[4].split(",")

required_methods = {
    "Ours_LogME_CKA",
    "GBC_CKA",
    "HScore_CKA",
    "LogME_SVCCA",
    "mRMR",
    "JMI",
    "Relevance_Only",
    "Random",
    "All_Models",
}

def is_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return False
    present = {k for k, v in payload.items() if isinstance(v, dict) and "selected" in v}
    return required_methods.issubset(present)

for dataset in datasets:
    out_path = os.path.join(selection_dir, f"{dataset}.json")
    if is_complete(out_path):
        print(f"[SKIP] {dataset}: selection results already complete")
        continue

    print(f"[RUN] {dataset}: selection comparison")
    subprocess.run([
        "python", "experiments/run_selection_comparison.py",
        "--data_root", feature_dir,
        "--datasets", dataset,
        "--max_models", max_k,
        "--selection_split", "train",
        "--output_dir", selection_dir,
    ], check=True)

print(f"[阶段 2 完成] 排序结果已保存到 {selection_dir}")
PYTHON_SCRIPT
echo ""

if [ "${DELETE_FEATURES_AFTER_SELECTION}" = "1" ]; then
    echo "[清理] 删除选择阶段特征目录: ${FEATURE_DIR}"
    rm -rf "${FEATURE_DIR}"
    echo ""
fi

# ================================================================
# 阶段 3: 按排序做融合训练
# ================================================================
echo "================================================================"
echo "  阶段 3/3: 按选择排序做累加融合训练"
echo "================================================================"

mkdir -p "${FUSION_DIR}"

python3 - "${SELECTION_DIR}" "${DATASETS}" "${MAX_K}" "${STORAGE_DIR}" "${EPOCHS}" "${BATCH_SIZE}" "${SEED}" "${FUSION_DIR}" "${VAL_RATIO}" "${VAL_SEED}" <<'PYTHON_SCRIPT'
import hashlib
import json
import os
import subprocess
import sys

selection_dir = sys.argv[1]
datasets = sys.argv[2].split(",")
max_k = int(sys.argv[3])
storage_dir = sys.argv[4]
epochs = sys.argv[5]
batch_size = sys.argv[6]
seed = sys.argv[7]
fusion_dir = sys.argv[8]
val_ratio = sys.argv[9]
val_seed = sys.argv[10]

METHODS = [
    "Ours_LogME_CKA",
    "GBC_CKA",
    "HScore_CKA",
    "LogME_SVCCA",
    "mRMR",
    "JMI",
    "Relevance_Only",
    "Random",
    "All_Models",
]

total_runs = 0
completed_runs = 0
failed_runs = 0

for dataset in datasets:
    result_file = os.path.join(selection_dir, f"{dataset}.json")
    if not os.path.exists(result_file):
        print(f"[SKIP] {dataset}: 没有选择结果文件")
        continue

    with open(result_file) as f:
        selection_results = json.load(f)

    for method in METHODS:
        if method not in selection_results:
            print(f"[SKIP] {dataset}/{method}: 方法不存在")
            continue

        entry = selection_results[method]
        if "selected" not in entry:
            print(f"[SKIP] {dataset}/{method}: 没有选择结果")
            continue

        ordering = entry["selected"]

        for k in range(2, min(max_k + 1, len(ordering) + 1)):
            models_str = ",".join(ordering[:k])
            tag = f"{dataset}_{method}_k{k}"
            marker_hash = hashlib.sha1(
                f"{models_str}|seed={seed}|val={val_ratio}|valseed={val_seed}".encode("utf-8")
            ).hexdigest()[:10]
            marker = os.path.join(fusion_dir, f"{tag}_{marker_hash}.done")

            if os.path.exists(marker):
                completed_runs += 1
                continue

            total_runs += 1
            print(f"\n[RUN] {tag}: {models_str}")

            cmd = [
                "python", "main.py",
                "--storage_dir", storage_dir,
                "--dataset", dataset,
                "--model", "fusion",
                "--fusion_method", "gated",
                "--fusion_models", models_str,
                "--epochs", epochs,
                "--batch_size", batch_size,
                "--seed", seed,
                "--validation_ratio", val_ratio,
                "--validation_seed", val_seed,
                "--cache_dtype", "fp32",
                "--disable_fewshot",
                "--results_dir", fusion_dir,
            ]
            if os.environ.get("FUSION_CLEANUP_CACHE", "0") == "1":
                cmd.append("--cleanup_cache")

            try:
                subprocess.run(cmd, check=True)
                with open(marker, "w") as f:
                    f.write(f"{models_str}\n")
                completed_runs += 1
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] {tag}: {e}")
                failed_runs += 1
            except KeyboardInterrupt:
                print("\n[中断] 下次运行会从断点继续")
                sys.exit(1)

print(f"\n{'=' * 60}")
print("融合训练完成")
print(f"  总尝试: {total_runs}")
print(f"  已完成: {completed_runs}")
print(f"  失败:   {failed_runs}")
print(f"  结果:   {fusion_dir}")
print(f"{'=' * 60}")
PYTHON_SCRIPT

echo ""
echo "================================================================"
echo "  全部实验完成！"
echo "================================================================"
echo "  特征:     ${FEATURE_DIR}"
echo "  选择排序: ${SELECTION_DIR}"
echo "  融合结果: ${FUSION_DIR}"
echo ""
echo "  下一步: 收集结果画图"
echo "    python experiments/collect_fusion_results.py \\"
echo "      --selection_dir ${SELECTION_DIR} \\"
echo "      --fusion_dir ${FUSION_DIR} \\"
echo "      --single_model_dir ${STORAGE_DIR}/results"
echo "================================================================"
