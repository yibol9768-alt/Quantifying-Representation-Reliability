#!/bin/bash
# Compare model selection methods across multiple datasets.
#
# Usage:
#   bash experiments/run_selection_comparison.sh [DATA_ROOT] [DATASETS]
#
# Examples:
#   bash experiments/run_selection_comparison.sh ./data/features "dtd,eurosat,flowers102,food101,pets,sun397,ucf101"
#   bash experiments/run_selection_comparison.sh ./data/features "dtd,eurosat"

set -e

DATA_ROOT="${1:-./data/features}"
DATASETS="${2:-dtd,eurosat,flowers102,food101,pets,sun397,ucf101}"
MAX_MODELS="${3:-6}"
OUTPUT_DIR="result/selection_comparison"

echo "============================================"
echo "Model Selection Method Comparison"
echo "============================================"
echo "Data root:  ${DATA_ROOT}"
echo "Datasets:   ${DATASETS}"
echo "Max models: ${MAX_MODELS}"
echo "Output:     ${OUTPUT_DIR}"
echo "============================================"

python experiments/run_selection_comparison.py \
    --data_root "${DATA_ROOT}" \
    --datasets "${DATASETS}" \
    --max_models "${MAX_MODELS}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "Done. Results saved to ${OUTPUT_DIR}/"
