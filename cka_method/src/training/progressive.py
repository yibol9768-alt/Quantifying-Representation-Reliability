"""
渐进式融合评估（episode 版本）。

按模型选择顺序依次融合 k=1,2,...,N 个模型：
  - 训练：在全量训练集上 episode 采样，防止过拟合
  - 评估：在测试集上 episode 采样，返回 (mean_acc, ci_95)
  - 记录每步的测试准确率和置信区间，绘制性能增量曲线
"""
from typing import Dict, List, Tuple
import torch

from src.config import MODEL_REGISTRY, ExpConfig
from src.fusion.fusion_module import FusionModule
from src.training.trainer import train_fusion, evaluate_fusion


def progressive_fusion_eval(
    ordered_models: List[str],
    all_train_features: Dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    all_test_features: Dict[str, torch.Tensor],
    test_labels: torch.Tensor,
    cfg: ExpConfig,
) -> Tuple[List[float], List[Dict[str, float]], List[float]]:
    """
    渐进式融合评估：依次融合 k=1..N 个模型。

    训练使用全量训练集 + episode 采样（随机化 support/query，防止过拟合）。
    评估使用独立测试集 + episode 采样，返回带置信区间的准确率。

    Args:
        ordered_models:      模型引入的有序序列 [m*_1, ..., m*_N]
        all_train_features:  {model_name: (N_train, feat_dim)}  全量训练特征
        train_labels:        (N_train,)
        all_test_features:   {model_name: (N_test, feat_dim)}   测试集特征
        test_labels:         (N_test,)
        cfg:                 实验配置

    Returns:
        accuracies:  [mean_acc(k=1), ..., mean_acc(k=N)]
        weight_logs: [weights_dict_at_k1, ...]
        ci_95s:      [ci_95(k=1), ..., ci_95(k=N)]
    """
    # n_way 由 cfg.n_way 直接给出（默认 5-way）
    n_way = cfg.n_way

    accuracies: List[float] = []
    weight_logs: List[Dict[str, float]] = []
    ci_95s: List[float] = []

    for k in range(1, len(ordered_models) + 1):
        current_models = ordered_models[:k]
        print(f"\n{'='*60}")
        print(f"Progressive Fusion: k={k}, models={current_models}")
        print(f"  n_way={n_way}, k_shot={cfg.n_shot}, "
              f"epochs={cfg.n_train_epochs}, eps/epoch={cfg.n_train_episodes}")
        print(f"{'='*60}")

        feat_dims = {
            name: MODEL_REGISTRY[name]["feat_dim"]
            for name in current_models
        }

        tr_feats = {name: all_train_features[name] for name in current_models}
        t_feats = {name: all_test_features[name] for name in current_models}

        # 新建融合模块并 episode 训练
        fusion = FusionModule(feat_dims, d_proj=cfg.d_proj)

        fusion, _, _ = train_fusion(
            fusion=fusion,
            train_features=tr_feats,
            train_labels=train_labels,
            n_way=n_way,
            k_shot=cfg.n_shot,
            n_train_episodes=cfg.n_train_episodes,
            n_train_epochs=cfg.n_train_epochs,
            n_eval_query=cfg.n_eval_query,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device=cfg.device,
            seed=cfg.seed,
        )

        # Episode 评估（测试集，带 CI）
        acc, ci = evaluate_fusion(
            fusion=fusion,
            test_features=t_feats,
            test_labels=test_labels,
            n_way=n_way,
            k_shot=cfg.n_shot,
            n_eval_episodes=cfg.n_eval_episodes,
            n_eval_query=cfg.n_eval_query,
            device=cfg.device,
        )

        weights = fusion.get_weight_dict()
        accuracies.append(acc)
        weight_logs.append(weights)
        ci_95s.append(ci)

        print(f"  → k={k} | Acc = {acc:.4f} ± {ci:.4f}")
        print(f"  → Weights: {weights}")

    best_k = int(max(range(len(accuracies)), key=lambda i: accuracies[i])) + 1
    print(f"\n{'='*60}")
    print(f"Best k* = {best_k}, "
          f"Acc = {accuracies[best_k-1]:.4f} ± {ci_95s[best_k-1]:.4f}")
    print(f"Optimal model set: {ordered_models[:best_k]}")
    print(f"{'='*60}")

    return accuracies, weight_logs, ci_95s
