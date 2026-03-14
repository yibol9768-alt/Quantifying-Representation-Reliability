#!/bin/bash
# CKA-Guided vs Original Order Scaling Experiment
#
# Compares two model addition orderings:
#   Original: clip → dino → mae → siglip → convnext → data2vec (fixed)
#   CKA:      per-dataset greedy ordering from CKA analysis (max diversity)
#
# Uses Concat fusion (no learnable fusion params → clean ordering signal).
# Runs both 10-shot and full-data to test across data regimes.
#
# Usage:
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_cka_scaling.sh
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_cka_scaling.sh --quick
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_cka_scaling.sh --fulldata-only

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?Please set STORAGE_DIR}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
CACHE_DTYPE="${CACHE_DTYPE:-fp32}"

BASE_CMD="python main.py --storage_dir $STORAGE_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED --cache_dtype $CACHE_DTYPE"

DATASETS_FULL="stl10 pets eurosat dtd gtsrb svhn country211"
DATASETS_QUICK="stl10 gtsrb svhn"

# ---- Original ordering (fixed, from previous scaling experiments) ----
ORIG_STEPS=(
    "clip"
    "clip,dino"
    "clip,dino,mae"
    "clip,dino,mae,siglip"
    "clip,dino,mae,siglip,convnext"
    "clip,dino,mae,siglip,convnext,data2vec"
)

# ---- CKA-guided per-dataset orderings ----
# Source: result/cka_patch_pca_full_fix_20260314_220955/selection_results.json
# Greedy selection from clip, each step adds the model with lowest avg CKA
# to the current set.
get_cka_steps() {
    local dataset=$1
    case "$dataset" in
        stl10)
            echo "clip clip,mae clip,mae,data2vec clip,mae,data2vec,convnext clip,mae,data2vec,convnext,siglip clip,mae,data2vec,convnext,siglip,dino" ;;
        gtsrb)
            echo "clip clip,mae clip,mae,dino clip,mae,dino,siglip clip,mae,dino,siglip,convnext clip,mae,dino,siglip,convnext,data2vec" ;;
        svhn)
            echo "clip clip,mae clip,mae,convnext clip,mae,convnext,dino clip,mae,convnext,dino,data2vec clip,mae,convnext,dino,data2vec,siglip" ;;
        pets)
            echo "clip clip,mae clip,mae,convnext clip,mae,convnext,data2vec clip,mae,convnext,data2vec,siglip clip,mae,convnext,data2vec,siglip,dino" ;;
        eurosat)
            echo "clip clip,mae clip,mae,convnext clip,mae,convnext,dino clip,mae,convnext,dino,data2vec clip,mae,convnext,dino,data2vec,siglip" ;;
        dtd)
            echo "clip clip,mae clip,mae,data2vec clip,mae,data2vec,dino clip,mae,data2vec,dino,siglip clip,mae,data2vec,dino,siglip,convnext" ;;
        country211)
            echo "clip clip,mae clip,mae,convnext clip,mae,convnext,siglip clip,mae,convnext,siglip,data2vec clip,mae,convnext,siglip,data2vec,dino" ;;
        *)
            echo "ERROR: unknown dataset $dataset" >&2; exit 1 ;;
    esac
}

# ---- Runner ----
run_one() {
    local dataset=$1
    local models=$2
    local n_models=$3
    local fewshot_flag=$4  # "" or "--disable_fewshot"
    local order_tag=$5     # "orig" or "cka"
    local tag
    if [ -z "$fewshot_flag" ]; then tag="10shot"; else tag="fulldata"; fi

    echo ""
    echo "============================================================"
    echo "  $dataset | ${n_models}m ($models) | $tag | $order_tag"
    echo "============================================================"

    if [ "$n_models" -eq 1 ]; then
        $BASE_CMD --dataset "$dataset" --model clip \
            $fewshot_flag \
            || echo "FAILED: $dataset ${n_models}m $tag $order_tag"
    else
        $BASE_CMD --dataset "$dataset" --model fusion \
            --fusion_method concat --fusion_models "$models" \
            $fewshot_flag \
            || echo "FAILED: $dataset ${n_models}m $tag $order_tag"
    fi
}

run_scaling_pair() {
    # Run both orderings for one dataset + one data setting
    local dataset=$1
    local fewshot_flag=$2

    # Original order
    for i in "${!ORIG_STEPS[@]}"; do
        run_one "$dataset" "${ORIG_STEPS[$i]}" "$((i + 1))" "$fewshot_flag" "orig"
    done

    # CKA order
    local cka_steps_str
    cka_steps_str=$(get_cka_steps "$dataset")
    read -ra CKA_STEPS <<< "$cka_steps_str"
    for i in "${!CKA_STEPS[@]}"; do
        run_one "$dataset" "${CKA_STEPS[$i]}" "$((i + 1))" "$fewshot_flag" "cka"
    done
}

# ---- Modes ----
quick_mode() {
    echo "=== Quick: CKA Scaling (3 datasets, full-data only) ==="
    echo "Runs: 3 datasets × 6 steps × 2 orderings = 36 runs"
    echo "(step 1 is shared — CLIP solo — so effectively 3 × 5 × 2 + 3 = 33 unique)"
    echo ""
    for ds in $DATASETS_QUICK; do
        run_scaling_pair "$ds" "--disable_fewshot"
    done
}

full_mode() {
    echo "=== Full: CKA Scaling (7 datasets, full-data only) ==="
    echo "Runs: 7 datasets × 6 steps × 2 orderings = 84 runs"
    echo ""
    for ds in $DATASETS_FULL; do
        run_scaling_pair "$ds" "--disable_fewshot"
    done
}

both_mode() {
    echo "=== Both: CKA Scaling (3 datasets, 10-shot + full-data) ==="
    echo "Runs: 3 datasets × 6 steps × 2 orderings × 2 settings = 72 runs"
    echo ""
    for ds in $DATASETS_QUICK; do
        run_scaling_pair "$ds" ""                  # 10-shot
        run_scaling_pair "$ds" "--disable_fewshot"  # full data
    done
}

fulldata_only_mode() {
    echo "=== Full-data only: CKA Scaling (7 datasets) ==="
    echo "Runs: 7 datasets × 6 steps × 2 orderings = 84 runs"
    echo ""
    for ds in $DATASETS_FULL; do
        run_scaling_pair "$ds" "--disable_fewshot"
    done
}

case "${1:---quick}" in
    --quick)          quick_mode ;;
    --full)           full_mode ;;
    --both)           both_mode ;;
    --fulldata-only)  fulldata_only_mode ;;
    *)
        echo "Usage: $0 [--quick|--full|--both|--fulldata-only]"
        echo "  --quick:          3 datasets, full-data only (36 runs)"
        echo "  --full:           7 datasets, full-data only (84 runs)"
        echo "  --both:           3 datasets, 10-shot + full-data (72 runs)"
        echo "  --fulldata-only:  7 datasets, full-data only (84 runs)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  CKA Scaling experiments complete!"
echo "  Results saved in: $STORAGE_DIR/results/"
echo "  "
echo "  Next: run 'python experiments/collect_results.py' to"
echo "  aggregate and compare original vs CKA orderings."
echo "============================================================"
