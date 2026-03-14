"""Unified training loop for offline cache and online modes."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from src.models.fusion import get_extractor
from src.models.classifier import MLPClassifier
from src.data.dataset import get_dataloaders
from src.training.cache import (
    CachedShardDataset,
    GroupedShardSampler,
    build_split_cache,
    cleanup_cache_dir,
    move_to_device,
)
from src.training.results import (
    init_result_tracker,
    record_epoch_result,
    finalize_result_tracker,
)


def has_trainable_extractor(extractor) -> bool:
    """Check whether the cached extractor still has trainable layers."""
    return any(param.requires_grad for param in extractor.parameters())


def save_training_checkpoint(checkpoint_path, classifier, extractor):
    """Save classifier and any trainable post-backbone fusion layers."""
    payload = {
        "classifier": classifier.state_dict(),
    }
    if has_trainable_extractor(extractor):
        payload["extractor"] = extractor.state_dict()
    checkpoint_path = Path(checkpoint_path)
    torch.save(payload, checkpoint_path)
    return checkpoint_path.resolve()


def build_cached_loaders(args, extractor, device, use_fp16):
    """Build disk-backed offline caches and dataloaders."""
    cache_root = Path(args.cache_dir) / args.cache_name
    train_split_dir = cache_root / "train"
    test_split_dir = cache_root / "test"

    print(f"\nLoading dataset {args.dataset}...")
    fewshot = not args.disable_fewshot
    image_train_loader, image_test_loader = get_dataloaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.loader_model_type,
        fewshot_min=args.fewshot_min if fewshot else None,
        fewshot_max=args.fewshot_max if fewshot else None,
        seed=args.seed,
    )

    storage_dtype = torch.float16 if args.cache_dtype == "fp16" else torch.float32

    print(f"\n[Step 1/3] Building train cache at {train_split_dir}")
    build_split_cache(
        extractor=extractor,
        dataloader=image_train_loader,
        split_dir=train_split_dir,
        split="train",
        device=device,
        storage_dtype=storage_dtype,
        use_fp16=use_fp16,
        recache=args.rebuild_cache,
    )

    print(f"\n[Step 2/3] Building test cache at {test_split_dir}")
    build_split_cache(
        extractor=extractor,
        dataloader=image_test_loader,
        split_dir=test_split_dir,
        split="test",
        device=device,
        storage_dtype=storage_dtype,
        use_fp16=use_fp16,
        recache=args.rebuild_cache,
    )

    extractor.release_backbones()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_dataset = CachedShardDataset(train_split_dir)
    test_dataset = CachedShardDataset(test_split_dir)
    train_sampler = GroupedShardSampler(train_dataset, shuffle=True, seed=args.seed)
    test_sampler = GroupedShardSampler(test_dataset, shuffle=False, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    return train_loader, test_loader, cache_root


def _forward_step(extractor, inputs, use_cache, device):
    """Get features from either cached inputs or raw images."""
    if use_cache:
        inputs = move_to_device(inputs, device)
        return extractor.forward_from_cache(inputs)
    else:
        return extractor(inputs.to(device, non_blocking=True))


def print_gpu_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB used / {reserved:.2f}GB reserved")


def train(args, config):
    """Unified training function for both offline cache and online modes.

    Args:
        args: Parsed CLI arguments. Must have pre-computed attributes:
            - fusion_kwargs: dict of fusion method kwargs
            - checkpoint_name: str for checkpoint file path
            - cache_name: str for cache directory name
            - loader_model_type: str for data transforms
        config: Config dataclass with num_classes etc.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None
    use_cache = not args.no_precompute

    print(f"\nDevice: {device}")
    print(f"Mixed precision: {use_fp16}")
    if use_cache:
        print(f"Cache dtype: {args.cache_dtype}")
    print_gpu_usage()

    # Load extractor
    print(f"\nLoading {args.model} model...")
    extractor = get_extractor(
        args.model,
        fusion_method=args.fusion_method,
        fusion_models=args.fusion_model_list,
        fusion_kwargs=args.fusion_kwargs,
        model_dir=args.model_dir,
    ).to(device)
    print(f"Feature dimension: {extractor.feature_dim}")
    print_gpu_usage()

    # Build data loaders
    cache_root = None
    if use_cache:
        train_loader, test_loader, cache_root = build_cached_loaders(
            args, extractor, device, use_fp16
        )
    else:
        fewshot = not args.disable_fewshot
        print(f"\nLoading dataset {args.dataset}...")
        train_loader, test_loader = get_dataloaders(
            args.dataset,
            args.data_dir,
            args.batch_size,
            args.num_workers,
            args.loader_model_type,
            fewshot_min=args.fewshot_min if fewshot else None,
            fewshot_max=args.fewshot_max if fewshot else None,
            seed=args.seed,
        )

    result_tracker = init_result_tracker(
        args, config,
        "offline_cache" if use_cache else "online",
        cache_root=cache_root,
    )
    print(f"Results JSON: {result_tracker['json_path']}")
    print(f"Results CSV: {result_tracker['csv_path']}")

    # Classifier
    classifier = MLPClassifier(
        feature_dim=extractor.feature_dim,
        num_classes=config.num_classes,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Optimizer
    trainable_ext = has_trainable_extractor(extractor)
    optim_params = list(classifier.parameters())
    if trainable_ext:
        ext_params = [p for p in extractor.parameters() if p.requires_grad]
        optim_params.extend(ext_params)
        print(f"Trainable fusion params: {sum(p.numel() for p in ext_params):,}")

    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = 0
    best_checkpoint_path = None

    mode_label = "offline cache" if use_cache else "online"
    print("\n" + "=" * 60)
    print(f"Starting training ({mode_label} mode)...")
    print("=" * 60)

    for epoch in range(args.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if trainable_ext:
            extractor.train()
        else:
            extractor.eval()
        classifier.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in pbar:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            if trainable_ext:
                if use_fp16:
                    with autocast():
                        features = _forward_step(extractor, inputs, use_cache, device)
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                        if hasattr(extractor, 'aux_loss') and extractor.aux_loss is not None:
                            loss = loss + args.router_aux_weight * extractor.aux_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    features = _forward_step(extractor, inputs, use_cache, device)
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    if hasattr(extractor, 'aux_loss') and extractor.aux_loss is not None:
                        loss = loss + args.router_aux_weight * extractor.aux_loss
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    if use_fp16:
                        with autocast():
                            features = _forward_step(extractor, inputs, use_cache, device)
                    else:
                        features = _forward_step(extractor, inputs, use_cache, device)

                if use_fp16:
                    with autocast():
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*train_correct/train_total:.1f}%"
            })

        scheduler.step()

        # Evaluate
        classifier.eval()
        extractor.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.to(device, non_blocking=True)

                if use_fp16:
                    with autocast():
                        features = _forward_step(extractor, inputs, use_cache, device)
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                else:
                    features = _forward_step(extractor, inputs, use_cache, device)
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)

                test_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / train_total
        test_loss = test_loss / test_total
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        is_best = test_acc > best_acc
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_checkpoint_path = save_training_checkpoint(
                args.checkpoint_name, classifier, extractor,
            )

        record_epoch_result(
            result_tracker,
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            best_acc=best_acc,
            is_best=is_best,
            optimizer=optimizer,
        )

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Test {test_acc:.2f}% | Best {best_acc:.2f}%")

    # Cleanup and finalize
    cache_removed = False
    if use_cache and args.cleanup_cache:
        cleanup_cache_dir(cache_root)
        cache_removed = True
        print(f"\nRemoved offline cache: {cache_root}")

    finalize_result_tracker(
        result_tracker,
        best_acc=best_acc,
        best_epoch=best_epoch,
        checkpoint_path=best_checkpoint_path,
        cache_root=cache_root if use_cache else None,
        cache_removed=cache_removed,
    )
    return best_acc
