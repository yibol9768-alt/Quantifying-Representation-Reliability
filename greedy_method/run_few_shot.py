"""
run_few_shot.py
===============
最终 few-shot 鲁棒性评估实验入口。

对每个 K ∈ {1, 5, 10, 16}：
  • 用贪心搜索找到的最优模型集（或手动指定）构建 MultiViewFusion
  • 在 train 特征上进行情节式训练（原型欧氏距离损失）
  • 在 test 特征上采样 600 个情节评估，报告均值准确率 ± 95% CI

无全量数据训练，无 MLP 分类头，纯 few-shot + 原型距离范式。

用法示例
--------
  # 使用贪心搜索结果中的最优模型集
  python run_few_shot.py --dataset dtd \\
      --greedy_result results/dtd/greedy_search/greedy_result.json

  # 只取贪心结果的前 5 个模型
  python run_few_shot.py --dataset dtd \\
      --greedy_result results/dtd/greedy_search/greedy_result.json \\
      --top_k_models 5

  # 手动指定模型
  python run_few_shot.py --dataset dtd \\
      --models dinov2_base clip_vit_base vit_base
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import torch

from configs.config import ALL_MODELS, DATASET_REGISTRY, TrainingConfig
from data.build_dataset import build_dataset
from engine.feature_cache import FeatureCache, build_cache_for_dataset
from engine.few_shot_engine import FewShotEngine, build_episode_loader
from models.encoder_zoo import build_encoder_zoo
from models.fusion_network import MultiViewFusion
from analysis.plotter import plot_few_shot_accuracy
from utils.misc import set_seed, resolve_device, setup_logging, save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Few-Shot 原型欧氏距离评估"
    )
    # 数据
    p.add_argument("--dataset",     type=str, default="dtd",
                   choices=list(DATASET_REGISTRY))
    p.add_argument("--data_root",   type=str, default="/root/autodl-tmp/data/raw")
    p.add_argument("--cache_dir",   type=str, default="/root/autodl-tmp/data/feature_cache")
    p.add_argument("--results_dir", type=str, default="/root/autodl-tmp/results")
    p.add_argument("--log_dir",     type=str, default="/root/autodl-tmp/logs")

    # 模型选择（三选一）
    p.add_argument("--models",        type=str, nargs="+", default=None)
    p.add_argument("--greedy_result", type=str, default=None,
                   help="贪心结果 JSON 路径，自动读取最优模型集")
    p.add_argument("--top_k_models",  type=int, default=None,
                   help="若指定 greedy_result，只取前 k 个模型")

    # 网络架构
    p.add_argument("--fusion_dim", type=int, default=512)

    # Few-shot 评估参数
    p.add_argument("--k_values",          type=int, nargs="+", default=[1, 5, 10, 16])
    p.add_argument("--n_way",             type=int, default=None,
                   help="N-way（默认使用数据集实际类别数）")
    p.add_argument("--n_query",           type=int, default=15)
    p.add_argument("--eval_episodes",     type=int, default=600,
                   help="测试情节数（越多 CI 越窄）")
    p.add_argument("--eval_epochs",       type=int, default=20,
                   help="每个 K 值的训练轮数")
    p.add_argument("--eval_train_episodes", type=int, default=200,
                   help="每轮训练情节数")

    # 优化器
    p.add_argument("--lr", type=float, default=1e-3)

    # 硬件
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--force_cache", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    # N-way 自动检测：未指定时使用数据集实际类别数
    if args.n_way is None:
        args.n_way = DATASET_REGISTRY[args.dataset]["num_classes"]

    cfg = TrainingConfig(
        fusion_dim          = args.fusion_dim,
        learning_rate       = args.lr,
        data_root           = args.data_root,
        cache_dir           = args.cache_dir,
        eval_k_values       = args.k_values,
        eval_n_way          = args.n_way,
        eval_n_query        = args.n_query,
        eval_episodes       = args.eval_episodes,
        eval_epochs         = args.eval_epochs,
        eval_train_episodes = args.eval_train_episodes,
        log_dir             = args.log_dir,
        results_dir         = args.results_dir,
        device              = args.device,
        seed                = args.seed,
    )

    results_dir = os.path.join(cfg.results_dir, args.dataset, "few_shot")
    setup_logging(os.path.join(cfg.log_dir, f"fewshot_{args.dataset}"),
                  f"fewshot_{args.dataset}")
    set_seed(cfg.seed)

    # ------------------------------------------------------------------
    # 确定模型集
    # ------------------------------------------------------------------
    if args.greedy_result:
        with open(args.greedy_result) as f:
            greedy = json.load(f)
        model_names: List[str] = greedy["selection_order"]
        if args.top_k_models:
            model_names = model_names[: args.top_k_models]
        logger.info(f"使用贪心结果模型集: {model_names}")
    elif args.models:
        model_names = args.models
    else:
        model_names = ALL_MODELS

    logger.info("=" * 60)
    logger.info(f"数据集   : {args.dataset}")
    logger.info(f"模型集   : {model_names}")
    logger.info(f"N-way    : {cfg.eval_n_way},  K 值: {cfg.eval_k_values}")
    logger.info(f"测试情节 : {cfg.eval_episodes} episodes")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 加载数据集 & 缓存特征
    # ------------------------------------------------------------------
    logger.info("加载数据集 …")
    train_ds, val_ds, test_ds = build_dataset(args.dataset, cfg)

    cache    = FeatureCache(cfg.cache_dir, args.dataset)
    encoders = build_encoder_zoo(model_names)

    build_cache_for_dataset(
        encoders    = encoders,
        datasets    = {"train": train_ds, "val": val_ds, "test": test_ds},
        cache       = cache,
        batch_size  = 256,
        num_workers = cfg.num_workers,
        device      = device,
        force       = args.force_cache,
    )
    del encoders
    torch.cuda.empty_cache()

    train_feats, train_labels = cache.load_split("train", model_names)
    test_feats,  test_labels  = cache.load_split("test",  model_names)

    # ------------------------------------------------------------------
    # 逐 K 值实验
    # ------------------------------------------------------------------
    all_means: List[float] = []
    all_cis:   List[float] = []
    all_results = {}

    for k_shot in cfg.eval_k_values:
        logger.info(f"\n{'─'*50}")
        logger.info(f"K-shot = {k_shot}")

        # 每个 K 值独立初始化 fusion
        fusion = MultiViewFusion(model_names, cfg.fusion_dim)
        engine = FewShotEngine(
            fusion       = fusion,
            n_way        = cfg.eval_n_way,
            k_shot       = k_shot,
            n_query      = cfg.eval_n_query,
            device       = device,
            lr           = cfg.learning_rate,
            weight_decay = cfg.weight_decay,
        )

        # 训练：每个 epoch 不同 seed 保证情节多样性
        for epoch in range(cfg.eval_epochs):
            train_loader = build_episode_loader(
                features_dict = train_feats,
                labels        = train_labels,
                n_way         = cfg.eval_n_way,
                k_shot        = k_shot,
                n_query       = cfg.eval_n_query,
                n_episodes    = cfg.eval_train_episodes,
                seed          = cfg.seed + epoch * 13,
            )
            loss, acc = engine.train_episodes(train_loader, cfg.eval_train_episodes)
            logger.info(
                f"  Epoch [{epoch+1:02d}/{cfg.eval_epochs}] "
                f"loss={loss:.4f}  train_acc={acc:.2%}"
            )

        # 测试
        test_loader = build_episode_loader(
            features_dict = test_feats,
            labels        = test_labels,
            n_way         = cfg.eval_n_way,
            k_shot        = k_shot,
            n_query       = cfg.eval_n_query,
            n_episodes    = cfg.eval_episodes,
            seed          = cfg.seed + 99999,
        )
        mean_acc, ci_95 = engine.evaluate_episodes(test_loader, cfg.eval_episodes)
        logger.info(f"  K={k_shot}  Test acc = {mean_acc:.2%} ± {ci_95:.2%}")

        all_means.append(mean_acc)
        all_cis.append(ci_95)
        all_results[f"{k_shot}-shot"] = {"mean": mean_acc, "ci_95": ci_95}

    # ------------------------------------------------------------------
    # 保存结果 & 绘图
    # ------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)

    save_json(
        {
            "dataset":    args.dataset,
            "models":     model_names,
            "n_way":      cfg.eval_n_way,
            "k_values":   cfg.eval_k_values,
            "results":    all_results,
        },
        os.path.join(results_dir, "few_shot_result.json"),
    )

    plot_few_shot_accuracy(
        k_values             = cfg.eval_k_values,
        accuracies           = all_means,
        confidence_intervals = all_cis,
        dataset_name         = args.dataset,
        model_label          = "+".join(model_names[:3]) + (
            f" +{len(model_names)-3} more" if len(model_names) > 3 else ""
        ),
        save_path = os.path.join(results_dir, "few_shot_accuracy.png"),
    )

    # 控制台汇总
    print(f"\nFew-Shot 结果（{args.dataset.upper()}，{cfg.eval_n_way}-way）")
    print(f"{'K-shot':>8}  {'准确率':>10}  {'95% CI':>8}")
    print("-" * 34)
    for k, mean, ci in zip(cfg.eval_k_values, all_means, all_cis):
        print(f"{k:>7}-shot  {mean:>9.2%}  ±{ci:>6.2%}")
    print()

    logger.info("Few-shot 实验完成。")


if __name__ == "__main__":
    main()
