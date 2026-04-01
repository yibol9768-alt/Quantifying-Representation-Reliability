"""
训练循环：基于 episode 的融合模块训练与评估。

梯度仅回传更新：
  - 特征投影层 Linear_i 的参数
  - 融合权重标量 w_i
所有编码器参数保持冻结。

使用 EpisodeLoader（纯 GPU 张量采样），特征一次性移至 GPU，
消除原 DataLoader 的 Python 逐样本循环和 H2D 搬运瓶颈。
"""
import torch
import torch.optim as optim
from typing import Dict, List, Tuple

from src.fusion.fusion_module import FusionModule
from src.fusion.classifier import build_prototypes, prototype_loss
from src.datasets.episode import EpisodeLoader


def train_fusion(
    fusion: FusionModule,
    train_features: Dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    n_way: int,
    k_shot: int,
    n_train_episodes: int = 100,
    n_train_epochs: int = 10,
    n_eval_query: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[FusionModule, List[float], List[float]]:
    """
    Episode 训练融合模块。

    特征在 EpisodeLoader 初始化时一次性移至 GPU，
    之后每个 epoch 直接在 GPU 上采样，无 CPU/GPU 来回传输。

    Args:
        fusion:            FusionModule 实例
        train_features:    {model_name: (N_train, feat_dim)}  全量训练集特征
        train_labels:      (N_train,)                         全量训练集标签
        n_way:             每个 episode 的类别数
        k_shot:            每类 support 样本数
        n_train_episodes:  每 epoch 的 episode 数
        n_train_epochs:    训练 epoch 数
        n_eval_query:      每类 query 样本数
        lr:                学习率
        weight_decay:      L2 正则
        device:            设备
        seed:              随机种子基准（每 epoch 偏移，保证 episode 多样性）
        verbose:           是否打印训练过程

    Returns:
        fusion:        训练后的融合模块
        loss_history:  各 epoch 平均 loss
        acc_history:   各 epoch 平均 episode 准确率
    """
    fusion = fusion.to(device)
    optimizer = optim.Adam(fusion.parameters(), lr=lr, weight_decay=weight_decay)

    # 一次性将特征移至 GPU，构建类别索引
    loader = EpisodeLoader(
        features_dict=train_features,
        labels=train_labels,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_eval_query,
        device=device,
    )

    n_support = n_way * k_shot
    loss_history: List[float] = []
    acc_history: List[float] = []

    for epoch in range(n_train_epochs):
        fusion.train()

        epoch_loss = 0.0
        epoch_acc = 0.0

        for features_batch in loader.iter_episodes(n_train_episodes, seed=seed + epoch * 7):
            # features_batch 已在 GPU 上，无需 .to(device)
            z_all = fusion(features_batch)  # (n_way*(k_shot+n_query), d_proj)

            z_sup = z_all[:n_support]
            z_qry = z_all[n_support:]

            sup_labels = torch.arange(n_way, device=device).repeat_interleave(k_shot)
            qry_labels = torch.arange(n_way, device=device).repeat_interleave(n_eval_query)

            prototypes = build_prototypes(z_sup, sup_labels, n_way)
            loss, acc = prototype_loss(z_qry, qry_labels, prototypes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

        avg_loss = epoch_loss / n_train_episodes
        avg_acc = epoch_acc / n_train_episodes
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)

        if verbose and (epoch + 1) % 5 == 0:
            weights = fusion.get_weight_dict()
            w_str = ", ".join(f"{k}:{v:.3f}" for k, v in weights.items())
            print(f"    Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.4f} | W: [{w_str}]")

    return fusion, loss_history, acc_history


@torch.no_grad()
def evaluate_fusion(
    fusion: FusionModule,
    test_features: Dict[str, torch.Tensor],
    test_labels: torch.Tensor,
    n_way: int,
    k_shot: int,
    n_eval_episodes: int = 200,
    n_eval_query: int = 15,
    device: str = "cuda",
    seed: int = 9999,
) -> Tuple[float, float]:
    """
    Episode 评估融合模块。

    Returns:
        mean_acc: 平均准确率
        ci_95:    95% 置信区间（1.96 × std / sqrt(n)）
    """
    fusion = fusion.to(device).eval()

    loader = EpisodeLoader(
        features_dict=test_features,
        labels=test_labels,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_eval_query,
        device=device,
    )

    n_support = n_way * k_shot
    episode_accs: List[float] = []

    for features_batch in loader.iter_episodes(n_eval_episodes, seed=seed):
        z_all = fusion(features_batch)

        z_sup = z_all[:n_support]
        z_qry = z_all[n_support:]

        sup_labels = torch.arange(n_way, device=device).repeat_interleave(k_shot)
        qry_labels = torch.arange(n_way, device=device).repeat_interleave(n_eval_query)

        prototypes = build_prototypes(z_sup, sup_labels, n_way)
        _, acc = prototype_loss(z_qry, qry_labels, prototypes)
        episode_accs.append(acc)

    accs = torch.tensor(episode_accs)
    mean_acc = accs.mean().item()
    ci_95 = 1.96 * accs.std().item() / (len(accs) ** 0.5)
    return mean_acc, ci_95
