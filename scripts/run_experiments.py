#!/usr/bin/env python3
"""
Run experiments on multiple datasets in parallel
"""
import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# Datasets to run
DATASETS = ["cifar100", "flowers102", "pets"]
MODELS = ["clip", "dino", "mae"]
SPLITS = ["train", "test"]

# Log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def run_command(cmd, log_file):
    """Run command and log output"""
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    return process


def extract_features(dataset, model, split):
    """Extract features for a dataset/model/split combination"""
    cmd = [
        "python3", "scripts/1_extract_single.py",
        "--model", model,
        "--dataset", dataset,
        "--split", split
    ]
    log_file = LOG_DIR / f"{dataset}_{model}_{split}.log"
    print(f"[START] {dataset} {model} {split}")
    process = run_command(cmd, log_file)
    return process, log_file


def main():
    """Run all experiments"""
    print("=" * 60)
    print("Multi-Dataset Multi-Model Experiment Runner")
    print("=" * 60)
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Splits: {', '.join(SPLITS)}")
    print(f"Total combinations: {len(DATASETS) * len(MODELS) * len(SPLITS)}")
    print("=" * 60)

    # Track running processes
    running = {}
    completed = []
    failed = []

    # Phase 1: Extract features
    print("\n=== Phase 1: Feature Extraction ===\n")

    for dataset in DATASETS:
        for model in MODELS:
            for split in SPLITS:
                key = f"{dataset}_{model}_{split}"
                process, log_file = extract_features(dataset, model, split)
                running[key] = {"process": process, "log": log_file, "start": time.time()}

                # Limit concurrent processes (increased for better GPU utilization)
                while len(running) >= 16:
                    time.sleep(5)
                    done = []
                    for k, v in running.items():
                        ret = v["process"].poll()
                        if ret is not None:
                            if ret == 0:
                                completed.append(k)
                                elapsed = time.time() - v["start"]
                                print(f"[DONE] {k} ({elapsed:.1f}s)")
                            else:
                                failed.append(k)
                                print(f"[FAIL] {k}")
                            done.append(k)
                    for k in done:
                        del running[k]

    # Wait for remaining processes
    print("\nWaiting for feature extraction to complete...")
    while running:
        time.sleep(5)
        done = []
        for k, v in running.items():
            ret = v["process"].poll()
            if ret is not None:
                if ret == 0:
                    completed.append(k)
                    elapsed = time.time() - v["start"]
                    print(f"[DONE] {k} ({elapsed:.1f}s)")
                else:
                    failed.append(k)
                    print(f"[FAIL] {k}")
                done.append(k)
        for k in done:
            del running[k]

    print(f"\n=== Feature Extraction Complete ===")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    # Phase 2: Train single models
    print("\n=== Phase 2: Training Single Models ===\n")
    running.clear()

    for dataset in DATASETS:
        for model in MODELS:
            cmd = [
                "python3", "scripts/3_train_single.py",
                "--model", model,
                "--dataset", dataset
            ]
            log_file = LOG_DIR / f"{dataset}_{model}_train.log"
            key = f"{dataset}_{model}"
            print(f"[START] {key} training")
            process = run_command(cmd, log_file)
            running[key] = {"process": process, "log": log_file, "start": time.time()}

            # Limit concurrent training (uses more memory)
            while len(running) >= 8:
                time.sleep(5)
                done = []
                for k, v in running.items():
                    ret = v["process"].poll()
                    if ret is not None:
                        if ret == 0:
                            elapsed = time.time() - v["start"]
                            print(f"[DONE] {k} ({elapsed:.1f}s)")
                        else:
                            print(f"[FAIL] {k}")
                        done.append(k)
                for k in done:
                    del running[k]

    # Wait for training to complete
    while running:
        time.sleep(5)
        done = []
        for k, v in running.items():
            ret = v["process"].poll()
            if ret is not None:
                if ret == 0:
                    elapsed = time.time() - v["start"]
                    print(f"[DONE] {k} ({elapsed:.1f}s)")
                else:
                    print(f"[FAIL] {k}")
                done.append(k)
        for k in done:
            del running[k]

    # Phase 2.5: Merge features for fusion models
    print("\n=== Phase 2.5: Merging Features for Fusion ===\n")

    two_model_combos = [
        ("clip", "dino"),
        ("clip", "mae"),
        ("dino", "mae")
    ]

    for dataset in DATASETS:
        for model1, model2 in two_model_combos:
            for split in SPLITS:
                cmd = [
                    "python3", "scripts/2_extract_multi.py",
                    "--models", model1, model2,
                    "--dataset", dataset,
                    "--split", split
                ]
                log_file = LOG_DIR / f"{dataset}_{model1}_{model2}_merge_{split}.log"
                key = f"{dataset}_{model1}_{model2}_{split}"
                print(f"[START] {key} merge")
                process = run_command(cmd, log_file)
                running[key] = {"process": process, "log": log_file, "start": time.time()}

                while len(running) >= 8:
                    time.sleep(5)
                    done = []
                    for k, v in running.items():
                        ret = v["process"].poll()
                        if ret is not None:
                            if ret == 0:
                                elapsed = time.time() - v["start"]
                                print(f"[DONE] {k} ({elapsed:.1f}s)")
                            else:
                                print(f"[FAIL] {k}")
                            done.append(k)
                    for k in done:
                        del running[k]

    # Wait for merging to complete
    while running:
        time.sleep(5)
        done = []
        for k, v in running.items():
            ret = v["process"].poll()
            if ret is not None:
                if ret == 0:
                    elapsed = time.time() - v["start"]
                    print(f"[DONE] {k} ({elapsed:.1f}s)")
                else:
                    print(f"[FAIL] {k}")
                done.append(k)
        for k in done:
            del running[k]

    # Phase 3: Train fusion models
    print("\n=== Phase 3: Training Fusion Models ===\n")

    for dataset in DATASETS:
        for model1, model2 in two_model_combos:
            cmd = [
                "python3", "scripts/4_train_fusion.py",
                "--models", model1, model2,
                "--dataset", dataset
            ]
            log_file = LOG_DIR / f"{dataset}_{model1}_{model2}_fusion.log"
            key = f"{dataset}_{model1}_{model2}"
            print(f"[START] {key} fusion")
            process = run_command(cmd, log_file)
            running[key] = {"process": process, "log": log_file, "start": time.time()}

            while len(running) >= 8:
                time.sleep(5)
                done = []
                for k, v in running.items():
                    ret = v["process"].poll()
                    if ret is not None:
                        if ret == 0:
                            elapsed = time.time() - v["start"]
                            print(f"[DONE] {k} ({elapsed:.1f}s)")
                        else:
                            print(f"[FAIL] {k}")
                        done.append(k)
                for k in done:
                    del running[k]

    while running:
        time.sleep(5)
        done = []
        for k, v in running.items():
            ret = v["process"].poll()
            if ret is not None:
                if ret == 0:
                    elapsed = time.time() - v["start"]
                    print(f"[DONE] {k} ({elapsed:.1f}s)")
                else:
                    print(f"[FAIL] {k}")
                done.append(k)
        for k in done:
            del running[k]

    # Phase 3.5: Merge three-model features
    print("\n=== Phase 3.5: Merging Three-Model Features ===\n")

    for dataset in DATASETS:
        for split in SPLITS:
            cmd = [
                "python3", "scripts/2_extract_multi.py",
                "--models", "clip", "dino", "mae",
                "--dataset", dataset,
                "--split", split
            ]
            log_file = LOG_DIR / f"{dataset}_clip_dino_mae_merge_{split}.log"
            key = f"{dataset}_3model_{split}"
            print(f"[START] {key} merge")
            process = run_command(cmd, log_file)
            running[key] = {"process": process, "log": log_file, "start": time.time()}

            while len(running) >= 3:
                time.sleep(5)
                done = []
                for k, v in running.items():
                    ret = v["process"].poll()
                    if ret is not None:
                        if ret == 0:
                            elapsed = time.time() - v["start"]
                            print(f"[DONE] {k} ({elapsed:.1f}s)")
                        else:
                            print(f"[FAIL] {k}")
                        done.append(k)
                for k in done:
                    del running[k]

    while running:
        time.sleep(5)
        done = []
        for k, v in running.items():
            ret = v["process"].poll()
            if ret is not None:
                if ret == 0:
                    elapsed = time.time() - v["start"]
                    print(f"[DONE] {k} ({elapsed:.1f}s)")
                else:
                    print(f"[FAIL] {k}")
                done.append(k)
        for k in done:
            del running[k]

    # Phase 4: Three-model fusion
    print("\n=== Phase 4: Three-Model Fusion ===\n")
    for dataset in DATASETS:
        cmd = [
            "python3", "scripts/4_train_fusion.py",
            "--models", "clip", "dino", "mae",
            "--dataset", dataset
        ]
        log_file = LOG_DIR / f"{dataset}_clip_dino_mae_fusion.log"
        key = f"{dataset}_3model"
        print(f"[START] {key} fusion")
        process = run_command(cmd, log_file)
        running[key] = {"process": process, "log": log_file, "start": time.time()}

        while len(running) >= 3:
            time.sleep(5)
            done = []
            for k, v in running.items():
                ret = v["process"].poll()
                if ret is not None:
                    if ret == 0:
                        elapsed = time.time() - v["start"]
                        print(f"[DONE] {k} ({elapsed:.1f}s)")
                    else:
                        print(f"[FAIL] {k}")
                    done.append(k)
            for k in done:
                del running[k]

    while running:
        time.sleep(5)
        done = []
        for k, v in running.items():
            ret = v["process"].poll()
            if ret is not None:
                if ret == 0:
                    elapsed = time.time() - v["start"]
                    print(f"[DONE] {k} ({elapsed:.1f}s)")
                else:
                    print(f"[FAIL] {k}")
                done.append(k)
        for k in done:
            del running[k]

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
