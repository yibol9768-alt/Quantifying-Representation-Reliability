#!/bin/bash
# ================================================================
# Third-Term Ablation
#
# Reuses extracted features from an existing storage workspace and runs
# only the low-order three-term selector, then compares it against the
# already-finished two-term baseline.
# ================================================================

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?请设置 STORAGE_DIR，例如 export STORAGE_DIR=/root/autodl-tmp/feature_workspace_paper6_20260322}"
FEATURE_DIR="${FEATURE_DIR:-${STORAGE_DIR}/data/features}"
DATASETS="${DATASETS:-stl10,pets,eurosat,dtd,gtsrb,svhn}"
MAX_K="${MAX_K:-10}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.2}"
VAL_SEED="${VAL_SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
ETA_COND="${ETA_COND:-1.0}"
COND_PCA_DIM="${COND_PCA_DIM:-32}"
COND_MIN_CLASS_SAMPLES="${COND_MIN_CLASS_SAMPLES:-8}"
COND_REG="${COND_REG:-1e-3}"

BASE_SELECTION_DIR="${BASE_SELECTION_DIR:-${STORAGE_DIR}/results/selection_comparison}"
BASE_FUSION_DIR="${BASE_FUSION_DIR:-${STORAGE_DIR}/results/fusion_scaling}"

ABLATION_ROOT="${ABLATION_ROOT:-${STORAGE_DIR}/results/third_term_ablation}"
SELECTION_DIR="${SELECTION_DIR:-${ABLATION_ROOT}/selection_comparison}"
FUSION_DIR="${FUSION_DIR:-${ABLATION_ROOT}/fusion_scaling}"
PAPER_TABLES_DIR="${PAPER_TABLES_DIR:-${ABLATION_ROOT}/paper_tables}"

mkdir -p "${SELECTION_DIR}" "${FUSION_DIR}" "${PAPER_TABLES_DIR}"

echo "================================================================"
echo "  Third-Term Ablation"
echo "================================================================"
echo "  Storage:    ${STORAGE_DIR}"
echo "  Features:   ${FEATURE_DIR}"
echo "  Datasets:   ${DATASETS}"
echo "  Selection:  ${SELECTION_DIR}"
echo "  Fusion:     ${FUSION_DIR}"
echo "  Eta_cond:   ${ETA_COND}"
echo "================================================================"
echo ""

python3 experiments/run_selection_comparison.py \
    --data_root "${FEATURE_DIR}" \
    --datasets "${DATASETS}" \
    --max_models "${MAX_K}" \
    --selection_split train \
    --output_dir "${SELECTION_DIR}" \
    --methods "Ours_LogME_CKA_3Term" \
    --include_third_term \
    --eta_cond "${ETA_COND}" \
    --conditional_pca_dim "${COND_PCA_DIM}" \
    --conditional_min_class_samples "${COND_MIN_CLASS_SAMPLES}" \
    --conditional_reg "${COND_REG}"

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

METHOD = "Ours_LogME_CKA_3Term"
total_runs = 0
completed_runs = 0
failed_runs = 0

for dataset in datasets:
    result_file = os.path.join(selection_dir, f"{dataset}.json")
    if not os.path.exists(result_file):
        print(f"[SKIP] {dataset}: no selection file")
        continue

    with open(result_file) as f:
        selection_results = json.load(f)

    if METHOD not in selection_results or "selected" not in selection_results[METHOD]:
        print(f"[SKIP] {dataset}: no {METHOD} ordering")
        continue

    ordering = selection_results[METHOD]["selected"]

    for k in range(2, min(max_k + 1, len(ordering) + 1)):
        models_str = ",".join(ordering[:k])
        tag = f"{dataset}_{METHOD}_k{k}"
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
            "python3", "main.py",
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
            "--device", os.environ.get("DEVICE", "cuda:0"),
        ]

        try:
            subprocess.run(cmd, check=True)
            with open(marker, "w", encoding="utf-8") as f:
                f.write(models_str + "\n")
            completed_runs += 1
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {tag}: {e}")
            failed_runs += 1
        except KeyboardInterrupt:
            print("\n[INTERRUPT] resume is safe")
            sys.exit(1)

print(f"\n{'=' * 60}")
print("Third-term fusion runs complete")
print(f"  attempted: {total_runs}")
print(f"  completed: {completed_runs}")
print(f"  failed:    {failed_runs}")
print(f"  output:    {fusion_dir}")
print(f"{'=' * 60}")
PYTHON_SCRIPT

python3 experiments/collect_third_term_ablation.py \
    --base_selection_dir "${BASE_SELECTION_DIR}" \
    --base_fusion_dir "${BASE_FUSION_DIR}" \
    --ablation_selection_dir "${SELECTION_DIR}" \
    --ablation_fusion_dir "${FUSION_DIR}" \
    --single_model_dir "${STORAGE_DIR}/results" \
    --output_dir "${PAPER_TABLES_DIR}" \
    --fusion_method gated \
    --max_k "${MAX_K}"

echo ""
echo "Ablation table saved to ${PAPER_TABLES_DIR}"
