"""
run_greedy_search.py
====================
贪心前向模型选择实验入口。

完整流程
--------
  1. 解析 CLI 参数
  2. 加载 / 下载目标数据集
  3. 提取并缓存各编码器特征（已存在则跳过）
  4. 贪心前向搜索：每步用 few-shot 情节准确率打分，选最优模型加入集合
  5. 保存 JSON 结果 + 图像到 results/<dataset>/greedy_search/

打分范式：仅使用 few-shot 情节 + 原型欧氏距离，无全量数据训练，无 MLP 分类头。

用法示例
--------
  python run_greedy_search.py --dataset dtd
  python run_greedy_search.py --dataset pets --search_k 5 --search_epochs 15
  python run_greedy_search.py --dataset eurosat --models deit_small dinov2_base vit_base
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from configs.config import ALL_MODELS, DATASET_REGISTRY, TrainingConfig
from data.build_dataset import build_dataset
from engine.feature_cache import FeatureCache, build_cache_for_dataset
from models.encoder_zoo import build_encoder_zoo
from search.greedy_forward import GreedyForwardSearch
from analysis.plotter import plot_incremental_accuracy, plot_fusion_weight_heatmap
from utils.misc import set_seed, resolve_device, setup_logging, save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="贪心前向多视图特征融合搜索（few-shot 打分）"
    )
    # 数据
    p.add_argument("--dataset",     type=str, default="dtd",
                   choices=list(DATASET_REGISTRY))
    p.add_argument("--data_root",   type=str, default="/root/autodl-tmp/data/raw")
    p.add_argument("--cache_dir",   type=str, default="/root/autodl-tmp/data/feature_cache")
    p.add_argument("--results_dir", type=str, default="/root/autodl-tmp/results")
    p.add_argument("--log_dir",     type=str, default="/root/autodl-tmp/logs")

    # 候选模型
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="候选模型子集，默认全部 10 个")

    # 网络架构
    p.add_argument("--fusion_dim", type=int, default=512)

    # 贪心搜索 few-shot 参数
    p.add_argument("--search_k",              type=int, default=5,
                   help="贪心打分时的 K-shot（默认 5）")
    p.add_argument("--search_n_way",          type=int, default=5,
                   help="N-way（默认 5-way）")
    p.add_argument("--search_epochs",         type=int, default=10,
                   help="每个候选组合的训练轮数")
    p.add_argument("--search_train_episodes", type=int, default=100,
                   help="每轮训练情节数")
    p.add_argument("--search_val_episodes",   type=int, default=200,
                   help="打分时的 val 情节数")

    # 优化器
    p.add_argument("--lr",  type=float, default=1e-3)

    # 硬件
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--force_cache", action="store_true",
                   help="强制重新提取特征（忽略缓存）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    cfg = TrainingConfig(
        fusion_dim             = args.fusion_dim,
        learning_rate          = args.lr,
        data_root              = args.data_root,
        cache_dir              = args.cache_dir,
        search_k_shot          = args.search_k,
        search_n_way           = args.search_n_way,
        search_epochs          = args.search_epochs,
        search_train_episodes  = args.search_train_episodes,
        search_val_episodes    = args.search_val_episodes,
        log_dir                = args.log_dir,
        results_dir            = args.results_dir,
        device                 = args.device,
        seed                   = args.seed,
    )

    results_dir = os.path.join(cfg.results_dir, args.dataset, "greedy_search")
    setup_logging(os.path.join(cfg.log_dir, f"greedy_{args.dataset}"),
                  f"greedy_{args.dataset}")
    set_seed(cfg.seed)

    candidate_models = args.models if args.models else ALL_MODELS

    logger.info("=" * 60)
    logger.info(f"数据集      : {args.dataset}")
    logger.info(f"候选模型    : {candidate_models}")
    logger.info(f"打分范式    : {cfg.search_n_way}-way {cfg.search_k_shot}-shot 原型欧氏距离")
    logger.info(f"设备        : {device}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. 加载数据集
    # ------------------------------------------------------------------
    logger.info("加载数据集 …")
    train_ds, val_ds, test_ds = build_dataset(args.dataset, cfg)
    logger.info(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ------------------------------------------------------------------
    # 2. 提取并缓存编码器特征
    # ------------------------------------------------------------------
    cache    = FeatureCache(cfg.cache_dir, args.dataset)
    encoders = build_encoder_zoo(candidate_models)

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

    # ------------------------------------------------------------------
    # 3. 贪心前向搜索（内部全程 few-shot 打分）
    # ------------------------------------------------------------------
    logger.info("开始贪心前向搜索 …")
    searcher = GreedyForwardSearch(
        candidate_models = candidate_models,
        cache            = cache,
        cfg              = cfg,
        device           = device,
    )
    result = searcher.run()

    # ------------------------------------------------------------------
    # 4. 保存结果
    # ------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "greedy_result.json")
    save_json(result.to_dict(), json_path)
    logger.info(f"结果已保存 → {json_path}")

    # 控制台表格
    print("\n贪心搜索结果（few-shot 准确率）")
    print(f"{'步骤':>4}  {'加入模型':<20}  {'Val Acc':>8}  {'Test Acc':>9}")
    print("-" * 50)
    for k, (m, v, t) in enumerate(
        zip(result.selection_order, result.val_accuracies, result.test_accuracies), 1
    ):
        print(f"{k:>4}  {m:<20}  {v:>7.2%}  {t:>8.2%}")
    print()

    # 图像
    plot_incremental_accuracy(
        selection_order = result.selection_order,
        val_accuracies  = result.val_accuracies,
        test_accuracies = result.test_accuracies,
        dataset_name    = args.dataset,
        k_shot          = cfg.search_k_shot,
        save_path       = os.path.join(results_dir, "incremental_accuracy.png"),
    )
    plot_fusion_weight_heatmap(
        fusion_weights  = result.fusion_weights,
        selection_order = result.selection_order,
        dataset_name    = args.dataset,
        save_path       = os.path.join(results_dir, "fusion_weight_heatmap.png"),
    )

    logger.info("贪心搜索完成。")


if __name__ == "__main__":
    main()
