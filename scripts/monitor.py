#!/usr/bin/env python3
"""
Monitor experiment progress in real-time
"""
import os
import time
import re
from pathlib import Path
from datetime import datetime, timedelta

LOG_DIR = Path("logs")
FEATURES_DIR = Path("features")
CHECKPOINTS_DIR = Path("outputs/checkpoints")


def parse_log(log_file):
    """Parse log file for progress information"""
    if not log_file.exists():
        return {"status": "pending", "progress": 0, "message": "Not started"}

    with open(log_file, "r") as f:
        lines = f.readlines()

    # Check for completion
    if any("100%" in line or "Done!" in line or "Training complete" in line for line in lines):
        return {"status": "done", "progress": 100, "message": "Complete"}

    # Check for errors
    if any("Error" in line or "Traceback" in line or "Exception" in line for line in lines[-10:]):
        return {"status": "error", "progress": 0, "message": "Error detected"}

    # Extract progress percentage
    for line in reversed(lines[-50:]):
        match = re.search(r'(\d+)%', line)
        if match:
            return {"status": "running", "progress": int(match.group(1)), "message": "Running"}

    # Check if file is being written to
    if lines and any("Extracting" in line or "Training" in line or "Epoch" in line for line in lines[-10:]):
        return {"status": "running", "progress": 50, "message": "In progress"}

    return {"status": "running", "progress": 10, "message": "Started"}


def get_file_age(filepath):
    """Get file age in seconds"""
    if not filepath.exists():
        return None
    mtime = filepath.stat().st_mtime
    return time.time() - mtime


def format_duration(seconds):
    """Format seconds as readable duration"""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def check_features():
    """Check which features exist"""
    feature_files = list(FEATURES_DIR.glob("*.pt")) if FEATURES_DIR.exists() else []
    return {f.stem: f.stat().st_size for f in feature_files}


def check_checkpoints():
    """Check which checkpoints exist"""
    ckpt_files = list(CHECKPOINTS_DIR.glob("*.pth")) if CHECKPOINTS_DIR.exists() else []
    return {f.stem: f.stat().st_size for f in ckpt_files}


def display_status():
    """Display experiment status"""
    os.system('clear')

    print("=" * 80)
    print(" " * 25 + "EXPERIMENT MONITOR")
    print("=" * 80)
    print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Datasets and models
    datasets = ["cifar10", "cifar100", "flowers102", "pets"]
    models = ["clip", "dino", "mae"]

    # Feature extraction status
    print("=== FEATURE EXTRACTION ===")
    features = check_features()

    for ds in datasets:
        row = f"{ds:12} |"
        for m in models:
            key = f"{ds}_{m}_train"
            train_exists = key in features
            key_test = f"{ds}_{m}_test"
            test_exists = key_test in features

            if train_exists and test_exists:
                row += f" {m.upper():4} ✓ |"
            elif train_exists:
                row += f" {m.upper():4} ▸ |"
            else:
                row += f" {m.upper():4} · |"
        print(row)

    # Single model training status
    print()
    print("=== SINGLE MODEL TRAINING ===")
    checkpoints = check_checkpoints()

    for ds in datasets:
        row = f"{ds:12} |"
        for m in models:
            key = f"{ds}_{m}_single"
            if key in checkpoints:
                row += f" {m.upper():4} ✓ |"
            else:
                log_file = LOG_DIR / f"{ds}_{m}_train.log"
                info = parse_log(log_file)
                if info["status"] == "running":
                    age = get_file_age(log_file)
                    if age and age < 60:
                        row += f" {m.upper():4} ▶ |"
                    else:
                        row += f" {m.upper():4} ⏳ |"
                else:
                    row += f" {m.upper():4} · |"
        print(row)

    # Fusion model status
    print()
    print("=== FUSION MODELS ===")
    fusions = [
        ("clip+dino", "clip_dino"),
        ("clip+mae", "clip_mae"),
        ("dino+mae", "dino_mae"),
        ("all3", "clip_dino_mae")
    ]

    for ds in datasets:
        row = f"{ds:12} |"
        for name, key_part in fusions:
            key = f"{ds}_{key_part}_fusion"
            if key in checkpoints:
                row += f" {name:8} ✓ |"
            else:
                log_file = LOG_DIR / f"{ds}_{key_part}_fusion.log"
                info = parse_log(log_file)
                if info["status"] == "running":
                    row += f" {name:8} ▶ |"
                else:
                    row += f" {name:8} · |"
        print(row)

    # Recent logs
    print()
    print("=== RECENT ACTIVITY ===")

    log_files = sorted(LOG_DIR.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
    for log_file in log_files:
        age = get_file_age(log_file)
        info = parse_log(log_file)
        status_symbol = {"done": "✓", "running": "▶", "error": "✗", "pending": "·"}.get(info["status"], "?")
        print(f"  {status_symbol} {log_file.name:40} ({format_duration(age)} ago)")

    # Summary
    print()
    print("=== SUMMARY ===")

    total_features = len(datasets) * len(models) * 2  # train + test
    done_features = len([k for k in features.keys() if '_train' in k or '_test' in k]) // 2

    total_single = len(datasets) * len(models)
    done_single = len([k for k in checkpoints.keys() if '_single' in k])

    total_fusion_2 = len(datasets) * 3
    done_fusion_2 = len([k for k in checkpoints.keys() if any(x in k for x in ['_clip_dino_', '_clip_mae_', '_dino_mae_'])])

    total_fusion_3 = len(datasets)
    done_fusion_3 = len([k for k in checkpoints.keys() if 'clip_dino_mae' in k])

    total_all = total_features + total_single + total_fusion_2 + total_fusion_3
    done_all = done_features + done_single + done_fusion_2 + done_fusion_3

    print(f"  Features:     {done_features}/{total_features}")
    print(f"  Single Models: {done_single}/{total_single}")
    print(f"  2-Model Fusion: {done_fusion_2}/{total_fusion_2}")
    print(f"  3-Model Fusion: {done_fusion_3}/{total_fusion_3}")
    print(f"  ─────────────────────────────")
    print(f"  Total Progress: {done_all}/{total_all} ({100*done_all/total_all:.1f}%)")

    print()
    print("Legend: ✓ Done | ▶ Running | ⏳ Waiting | · Pending | ✗ Error")
    print("=" * 80)


def main():
    """Main monitor loop"""
    try:
        while True:
            display_status()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
