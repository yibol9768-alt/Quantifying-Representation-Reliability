"""Experiment result tracking and persistence."""

import csv
import json
from datetime import datetime
from pathlib import Path


def get_learning_rate(optimizer) -> float:
    """Get the current learning rate from the first param group."""
    return float(optimizer.param_groups[0]["lr"])


def init_result_tracker(args, config, training_mode: str, cache_root=None):
    """Create result files and seed them with run metadata.

    Expects args to have pre-computed attributes:
        - run_name (str or None): explicit run name
        - run_basename (str): auto-generated basename
        - checkpoint_name (str): checkpoint file path
        - results_dir (str): output directory
    """
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"{args.run_basename}_{timestamp}"
    json_path = results_root / f"{run_name}.json"
    csv_path = results_root / f"{run_name}.csv"

    payload = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "training_mode": training_mode,
        "dataset": args.dataset,
        "num_classes": config.num_classes,
        "model": args.model,
        "fusion_method": args.fusion_method if args.model == "fusion" else None,
        "fusion_models": args.fusion_model_list if args.model == "fusion" else None,
        "checkpoint_path": str(Path(args.checkpoint_name).resolve()),
        "results_json": str(json_path.resolve()),
        "results_csv": str(csv_path.resolve()),
        "storage_dir": args.storage_dir,
        "model_dir": args.model_dir,
        "data_dir": args.data_dir,
        "cache_dir": args.cache_dir,
        "cache_root": str(cache_root.resolve()) if cache_root is not None else None,
        "config": vars(args),
        "history": [],
        "summary": {},
    }

    tracker = {
        "json_path": json_path,
        "csv_path": csv_path,
        "payload": payload,
    }
    flush_result_tracker(tracker)
    return tracker


def flush_result_tracker(tracker):
    """Persist result tracker JSON and CSV to disk."""
    payload = tracker["payload"]
    with tracker["json_path"].open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    history = payload["history"]
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "best_acc",
        "is_best",
        "lr",
    ]
    with tracker["csv_path"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fieldnames})


def record_epoch_result(
    tracker,
    *,
    epoch: int,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    best_acc: float,
    is_best: bool,
    optimizer,
):
    """Append one epoch of metrics and persist result files."""
    tracker["payload"]["history"].append({
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "best_acc": float(best_acc),
        "is_best": bool(is_best),
        "lr": get_learning_rate(optimizer),
    })
    tracker["payload"]["last_updated_at"] = datetime.now().isoformat(timespec="seconds")
    flush_result_tracker(tracker)


def finalize_result_tracker(
    tracker,
    *,
    best_acc: float,
    best_epoch: int,
    checkpoint_path=None,
    cache_root=None,
    cache_removed: bool = False,
):
    """Mark the run complete and store its final summary."""
    tracker["payload"]["status"] = "completed"
    tracker["payload"]["completed_at"] = datetime.now().isoformat(timespec="seconds")
    tracker["payload"]["summary"] = {
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "cache_root": str(cache_root.resolve()) if cache_root is not None else None,
        "cache_removed": bool(cache_removed),
    }
    flush_result_tracker(tracker)
