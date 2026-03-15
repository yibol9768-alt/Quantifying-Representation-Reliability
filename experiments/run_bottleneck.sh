#!/bin/bash
# Redundancy-Aware Feature Bottleneck Experiments
#
# Compares 6-model fusion with different bottleneck strategies:
#   1. Concat (no bottleneck, baseline)
#   2. Linear Bottleneck (learnable 4608 → 512)
#   3. VIB (Variational Information Bottleneck)
#   4. Also compare best subset (4 models, no bottleneck)
#
# Hypothesis: 6-model + bottleneck ≥ best-k-model concat
# because bottleneck removes redundancy while retaining all information.
#
# Usage:
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_bottleneck.sh
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_bottleneck.sh --quick

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?Please set STORAGE_DIR}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
CACHE_DTYPE="${CACHE_DTYPE:-fp32}"

BASE_CMD="python main.py --storage_dir $STORAGE_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED --cache_dtype $CACHE_DTYPE"

ALL_MODELS="clip,dino,mae,siglip,convnext,data2vec"
FOUR_MODELS="clip,dino,mae,siglip"  # peak models from scaling experiments

DATASETS_FULL="stl10 pets eurosat dtd gtsrb svhn country211"
DATASETS_QUICK="stl10 gtsrb svhn"

BOTTLENECK_DIMS="256 512 1024"

run_one() {
    local dataset=$1
    local method=$2
    local models=$3
    local extra_args=$4
    local tag=$5

    echo ""
    echo "============================================================"
    echo "  $dataset | $method | $tag"
    echo "============================================================"

    $BASE_CMD --dataset "$dataset" --model fusion \
        --fusion_method "$method" --fusion_models "$models" \
        --disable_fewshot $extra_args \
        || echo "FAILED: $dataset $method $tag"
}

run_dataset() {
    local dataset=$1

    echo ""
    echo "############################################################"
    echo "  DATASET: $dataset"
    echo "############################################################"

    # Baseline 1: 6-model Concat (no bottleneck)
    run_one "$dataset" "concat" "$ALL_MODELS" "" "6m_concat"

    # Baseline 2: 4-model Concat (best subset from scaling)
    run_one "$dataset" "concat" "$FOUR_MODELS" "" "4m_concat"

    # Linear Bottleneck: sweep dimensions
    for dim in $BOTTLENECK_DIMS; do
        run_one "$dataset" "linear_bottleneck" "$ALL_MODELS" \
            "--bottleneck_dim $dim" "6m_linear_bn${dim}"
    done

    # VIB: sweep β values
    for beta in 0.001 0.01 0.1; do
        run_one "$dataset" "vib" "$ALL_MODELS" \
            "--bottleneck_dim 512 --router_aux_weight $beta" "6m_vib_b${beta}"
    done
}

quick_mode() {
    echo "=== Quick: Bottleneck Experiments (3 datasets) ==="
    echo "Runs: 3 datasets × (2 baselines + 3 linear + 3 VIB) = 24 runs"
    for ds in $DATASETS_QUICK; do
        run_dataset "$ds"
    done
}

full_mode() {
    echo "=== Full: Bottleneck Experiments (7 datasets) ==="
    echo "Runs: 7 datasets × (2 baselines + 3 linear + 3 VIB) = 56 runs"
    for ds in $DATASETS_FULL; do
        run_dataset "$ds"
    done
}

case "${1:---quick}" in
    --quick) quick_mode ;;
    --full)  full_mode ;;
    *)
        echo "Usage: $0 [--quick|--full]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Bottleneck experiments complete!"
echo "  Results in: $STORAGE_DIR/results/"
echo "============================================================"
