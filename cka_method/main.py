"""
主入口：基于信息论引导的渐进式加权特征融合 —— 完整三阶段流水线。

自动遍历 1/5/10/16-shot，选出最优配置。

用法:
    python main.py --dataset dtd
    python main.py --dataset gtsrb --alpha 1.0 --beta 1.0
    python main.py --dataset eurosat --n_way 5 --n_train_epochs 30
"""
import argparse
import torch
from typing import Dict, List

from src.config import ExpConfig, MODEL_REGISTRY
from src.utils.helpers import (
    set_seed, save_results,
    plot_accuracy_curve, plot_cka_heatmap, plot_weight_evolution,
    plot_shot_comparison, save_summary,
)
from src.models.encoders import get_all_encoders
from src.datasets.loader import build_fewshot_split, build_train_dataset, build_test_dataset
from src.datasets.features import extract_all_features, extract_labels
from src.selection.cka import compute_cka_matrix
from src.selection.relevance import compute_relevance_scores
from src.selection.selector import progressive_model_selection
from src.training.progressive import progressive_fusion_eval

SHOT_LIST = [1, 5, 10, 16]


def parse_args() -> ExpConfig:
    parser = argparse.ArgumentParser(
        description="Information-Theoretic Progressive Weighted Feature Fusion"
    )
    parser.add_argument("--dataset", type=str, default="dtd")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/data/raw")
    parser.add_argument("--n_query", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--d_proj", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Episode 训练参数
    parser.add_argument("--n_way", type=int, default=5,
                        help="Episode N-way，Phase 2 训练与评估使用")
    parser.add_argument("--n_train_episodes", type=int, default=100,
                        help="每 epoch 训练 episode 数")
    parser.add_argument("--n_train_epochs", type=int, default=10,
                        help="训练 epoch 数")
    parser.add_argument("--n_eval_episodes", type=int, default=200,
                        help="测试 episode 数")
    parser.add_argument("--n_eval_query", type=int, default=15,
                        help="每类 query 样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/results/cka_method")
    parser.add_argument("--model_root", type=str, default="/root/autodl-tmp/models")
    args = parser.parse_args()
    return ExpConfig(**vars(args))


def run_single_shot(
    cfg: ExpConfig,
    n_shot: int,
    encoders: Dict,
    device: str,
    train_features: Dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    cka_matrix,
    test_features: Dict[str, torch.Tensor] = None,
    test_labels: torch.Tensor = None,
) -> dict:
    """
    运行单个 shot 设定的完整流水线。

    Phase 1（relevance）使用 few-shot 子集。
    CKA 矩阵由外部传入（所有 shot 共用，避免重复计算）。
    Phase 2（渐进式融合）使用全量训练集 + episode 训练。

    Returns:
        {
            "n_shot", "ordered_models", "accuracies", "weight_logs",
            "ci_95s", "relevance", "cka_matrix", "best_k", "best_acc"
        }
    """
    cfg.n_shot = n_shot

    print(f"\n{'='*60}")
    print(f"  {cfg.dataset} | {n_shot}-shot")
    print(f"{'='*60}")

    # ── Phase 1 数据：few-shot 子集（support + query）用于 relevance ──
    fewshot = build_fewshot_split(
        cfg.dataset, cfg.data_root, n_shot, cfg.n_query, cfg.seed
    )
    support_labels = extract_labels(fewshot.support)
    query_labels = extract_labels(fewshot.query)

    print(f"  support: {len(fewshot.support)} | query: {len(fewshot.query)}")
    print(f"  train:   {len(train_labels)} (全量，用于 Phase 2 episode 训练)")

    # ── 特征切片（从全量训练集特征中按索引切片，避免重复提取）──
    sup_idx = fewshot.support_indices
    qry_idx = fewshot.query_indices
    support_features = {name: train_features[name][sup_idx] for name in cfg.model_names}
    query_features   = {name: train_features[name][qry_idx] for name in cfg.model_names}

    # ── 测试集（缓存复用）──
    if test_features is None or test_labels is None:
        test_dataset = build_test_dataset(cfg.dataset, cfg.data_root)
        test_features = extract_all_features(encoders, test_dataset, device=device)
        test_labels = extract_labels(test_dataset)
        print(f"  test:    {len(test_dataset)}")
    else:
        print(f"  test:    {len(test_labels)} (cached)")

    # ── Phase 1: 模型选择 ──
    print(f"\n  [1.1] Computing task relevance R̂(m)...")
    relevance = compute_relevance_scores(
        support_features=support_features,
        query_features=query_features,
        support_labels=support_labels,
        query_labels=query_labels,
        model_names=cfg.model_names,
        num_classes=cfg.num_classes,
        d_proj=cfg.d_proj,
        device=device,
    )

    print(f"\n  [1.2] CKA matrix (cached, skip recompute)")

    print(f"\n  [1.3] Progressive model selection...")
    ordered_models = progressive_model_selection(
        relevance, cka_matrix, cfg.model_names, cfg.alpha, cfg.beta
    )
    print(f"  Order: {ordered_models}")

    # ── Phase 2: 渐进式融合（episode 训练）──
    print(f"\n  [Phase 2] Progressive Fusion (episode training)...")
    accuracies, weight_logs, ci_95s = progressive_fusion_eval(
        ordered_models=ordered_models,
        all_train_features=train_features,
        train_labels=train_labels,
        all_test_features=test_features,
        test_labels=test_labels,
        cfg=cfg,
    )

    # ── 保存该 shot 的结果 ──
    save_results(
        cfg.output_dir, cfg.dataset, n_shot, "ours",
        ordered_models, accuracies, weight_logs, relevance,
        cka_matrix, cfg.model_names, ci_95s,
    )
    plot_weight_evolution(weight_logs, ordered_models, cfg.dataset,
                          n_shot, cfg.output_dir)
    plot_accuracy_curve(
        {"Ours (CKA + Acc)": accuracies},
        cfg.dataset, n_shot, cfg.output_dir,
        ci_95s={"Ours (CKA + Acc)": ci_95s},
    )

    best_k = int(max(range(len(accuracies)), key=lambda i: accuracies[i])) + 1
    best_acc = accuracies[best_k - 1]

    print(f"\n  ★ {n_shot}-shot | best k*={best_k} | "
          f"Acc={best_acc:.4f} ± {ci_95s[best_k-1]:.4f}")
    print(f"    Optimal set: {ordered_models[:best_k]}")

    return {
        "n_shot": n_shot,
        "ordered_models": ordered_models,
        "accuracies": accuracies,
        "weight_logs": weight_logs,
        "ci_95s": ci_95s,
        "relevance": relevance,
        "cka_matrix": cka_matrix,
        "best_k": best_k,
        "best_acc": best_acc,
    }


def run_experiment(cfg: ExpConfig):
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device

    print(f"\n{'#'*60}")
    print(f"# Dataset: {cfg.dataset} | Shots: {SHOT_LIST} | device: {device}")
    print(f"# α={cfg.alpha}, β={cfg.beta}, d_proj={cfg.d_proj}")
    print(f"# n_way={cfg.n_way}, train_eps={cfg.n_train_episodes}, "
          f"train_epochs={cfg.n_train_epochs}, eval_eps={cfg.n_eval_episodes}")
    print(f"{'#'*60}")

    # ═══════════════════════════════════════════════
    # 加载编码器（一次性，所有 shot 共用）
    # ═══════════════════════════════════════════════
    print("\n[Init] Loading encoders (once for all shots)...")
    encoders = get_all_encoders(cfg.model_names, model_root=cfg.model_root,
                                device=device)

    # ═══════════════════════════════════════════════
    # 提取全量训练集特征（一次性，Phase 2 所有 shot 共用）
    # ═══════════════════════════════════════════════
    print("\n[Init] Extracting full train set features (once for all shots)...")
    train_dataset = build_train_dataset(cfg.dataset, cfg.data_root)
    train_features = extract_all_features(encoders, train_dataset, device=device)
    train_labels = extract_labels(train_dataset)
    print(f"  Train set size: {len(train_dataset)}")

    # ═══════════════════════════════════════════════
    # 提取测试集特征（一次性，所有 shot 共用）
    # ═══════════════════════════════════════════════
    print("\n[Init] Extracting test set features (once for all shots)...")
    test_dataset = build_test_dataset(cfg.dataset, cfg.data_root)
    test_features = extract_all_features(encoders, test_dataset, device=device)
    test_labels = extract_labels(test_dataset)
    print(f"  Test set size: {len(test_dataset)}")

    # ═══════════════════════════════════════════════
    # 计算 CKA 矩阵（一次性，所有 shot 共用）
    # 每类采 50 个样本，Gram 矩阵可控；CKA 衡量冻结编码器的特征相似度，与 shot 无关
    # ═══════════════════════════════════════════════
    print("\n[Init] Computing CKA matrix (once for all shots)...")
    n_cka_per_class = 50
    rng_cka = torch.Generator()
    rng_cka.manual_seed(cfg.seed)
    cka_indices = []
    for c in train_labels.unique().tolist():
        idx = (train_labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(idx), generator=rng_cka)
        cka_indices.append(idx[perm[:n_cka_per_class]])
    cka_sel = torch.cat(cka_indices)
    cka_features = {name: train_features[name][cka_sel] for name in cfg.model_names}
    cka_matrix = compute_cka_matrix(cka_features, cfg.model_names)
    plot_cka_heatmap(cka_matrix, cfg.model_names, cfg.dataset, cfg.output_dir)

    # ═══════════════════════════════════════════════
    # 遍历所有 shot
    # ═══════════════════════════════════════════════
    all_shot_results = {}

    for n_shot in SHOT_LIST:
        result = run_single_shot(
            cfg, n_shot, encoders, device,
            train_features=train_features,
            train_labels=train_labels,
            cka_matrix=cka_matrix,
            test_features=test_features,
            test_labels=test_labels,
        )
        all_shot_results[n_shot] = result

    # ═══════════════════════════════════════════════
    # 汇总 & 选最优
    # ═══════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print(f"# SUMMARY — {cfg.dataset}")
    print(f"{'#'*60}")
    print(f"{'shot':>6} | {'best_k':>6} | {'Acc':>8} | {'CI':>8} | optimal models")
    print(f"{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*30}")

    best_shot = None
    best_overall_acc = -1.0

    for n_shot in SHOT_LIST:
        r = all_shot_results[n_shot]
        ci = r["ci_95s"][r["best_k"] - 1]
        if r["best_acc"] > best_overall_acc:
            best_overall_acc = r["best_acc"]
            best_shot = n_shot
        print(f"{n_shot:>5}  | {r['best_k']:>5}  | {r['best_acc']:>7.4f} | "
              f"±{ci:.4f} | {r['ordered_models'][:r['best_k']]}")

    winner = all_shot_results[best_shot]
    best_ci = winner["ci_95s"][winner["best_k"] - 1]
    print(f"\n{'>'*60}")
    print(f"  BEST: {best_shot}-shot | k*={winner['best_k']} | "
          f"Acc={winner['best_acc']:.4f} ± {best_ci:.4f}")
    print(f"  Models: {winner['ordered_models'][:winner['best_k']]}")
    print(f"{'>'*60}")

    # ── 汇总可视化 & JSON ──
    plot_shot_comparison(all_shot_results, cfg.dataset, cfg.output_dir)
    save_summary(all_shot_results, cfg.dataset, cfg.output_dir)


if __name__ == "__main__":
    cfg = parse_args()
    run_experiment(cfg)
