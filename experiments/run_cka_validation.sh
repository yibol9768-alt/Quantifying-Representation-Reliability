#!/usr/bin/env bash
# CKA Validation: compare CKA-selected model subsets vs baselines.
#
# Usage:
#   bash experiments/run_cka_validation.sh [--quick]
#
# Requires: results/cka/selection_results.json (from run_cka_analysis.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Defaults
SELECTION_JSON="results/cka/selection_results.json"
DEVICE="cuda:0"
DATASETS="stl10,gtsrb,svhn,pets,eurosat,dtd,country211"
RESULTS_DIR="results/cka_validation"
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK=true; DATASETS="stl10,gtsrb,svhn"; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --selection-json) SELECTION_JSON="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$SELECTION_JSON" ]]; then
    echo "Error: $SELECTION_JSON not found. Run run_cka_analysis.py first."
    exit 1
fi

echo "=========================================="
echo "CKA Validation Experiments"
echo "=========================================="
echo "Selection JSON: $SELECTION_JSON"
echo "Datasets: $DATASETS"
echo "Device: $DEVICE"
echo "Quick mode: $QUICK"
echo ""

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

# Helper: run a fusion experiment
run_fusion() {
    local dataset="$1"
    local models="$2"
    local tag="$3"
    local fewshot_args="$4"

    echo "  [$tag] models=$models $fewshot_args"
    python main.py \
        --model fusion \
        --models "$models" \
        --fusion_method concat \
        --dataset "$dataset" \
        --device "$DEVICE" \
        --epochs 50 \
        --lr 0.001 \
        $fewshot_args \
        2>&1 | tail -1
}

# For each dataset, compare CKA-selected vs all-6 vs random-3
for DATASET in "${DATASET_ARRAY[@]}"; do
    echo "--- Dataset: $DATASET ---"

    # Extract CKA-selected models for this dataset from JSON
    CKA_MODELS=$(python3 -c "
import json, sys
with open('$SELECTION_JSON') as f:
    data = json.load(f)
# Prefer task_adaptive, fallback to greedy
sel = data.get('task_adaptive', {}).get('$DATASET',
      data.get('greedy', {}).get('$DATASET', []))
print(','.join(sel))
")

    ALL_MODELS="clip,dino,mae,siglip,convnext,data2vec"
    RANDOM_MODELS=$(python3 -c "
import random; random.seed(42)
models = ['clip','dino','mae','siglip','convnext','data2vec']
random.shuffle(models)
print(','.join(models[:3]))
")

    echo "  CKA-selected: $CKA_MODELS"
    echo "  All 6 models: $ALL_MODELS"
    echo "  Random 3: $RANDOM_MODELS"
    echo ""

    # 10-shot experiments
    echo "  [10-shot setting]"
    run_fusion "$DATASET" "$CKA_MODELS" "cka_10shot" "--fewshot_min 10 --fewshot_max 10"
    run_fusion "$DATASET" "$ALL_MODELS" "all6_10shot" "--fewshot_min 10 --fewshot_max 10"
    run_fusion "$DATASET" "$RANDOM_MODELS" "rand3_10shot" "--fewshot_min 10 --fewshot_max 10"

    # Full-data experiments
    if [[ "$QUICK" == "false" ]]; then
        echo "  [full-data setting]"
        run_fusion "$DATASET" "$CKA_MODELS" "cka_full" ""
        run_fusion "$DATASET" "$ALL_MODELS" "all6_full" ""
        run_fusion "$DATASET" "$RANDOM_MODELS" "rand3_full" ""
    fi

    echo ""
done

echo "=========================================="
echo "Validation complete. Check results above."
echo "=========================================="
