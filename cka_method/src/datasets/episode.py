"""
src/datasets/episode.py
=======================
纯 GPU Episode 加载器。

特征在初始化时一次性移至 GPU，之后每次采样全程为 GPU 张量切片操作，
彻底消除原 DataLoader 方案中的 Python 逐样本 __getitem__ + collate + H2D 搬运瓶颈。

用法：
    loader = EpisodeLoader(features_dict, labels, n_way=5, k_shot=1,
                           n_query=15, device="cuda")
    for feats in loader.iter_episodes(n_episodes=100, seed=42):
        z = fusion(feats)   # feats 已在 GPU 上
"""
from typing import Dict, Iterator, List

import torch


class EpisodeLoader:
    """
    N-way K-shot Episode 采样器（纯 GPU 版）。

    初始化时：
      - 将 features_dict 中所有特征移至 device（一次性 H2D）
      - 构建各类别的全局样本索引（CPU int64 张量，轻量）

    每次采样时：
      - 所有随机操作在 CPU 完成（小张量，极快）
      - 组合好全局索引后执行单次 GPU 切片
      - 返回的特征字典已在 GPU 上，训练循环无需再 .to(device)
    """

    def __init__(
        self,
        features_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        n_way: int,
        k_shot: int,
        n_query: int,
        device: str,
    ) -> None:
        self.n_way   = n_way
        self.k_shot  = k_shot
        self.n_query = n_query
        self.device  = device

        # ── 一次性将特征移至 GPU ──
        self.features_dict: Dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in features_dict.items()
        }

        # ── 按类别构建样本索引（CPU tensor，保持轻量）──
        labels_cpu = labels.cpu()
        self.class_indices: List[torch.Tensor] = []
        for c in labels_cpu.unique().tolist():
            idx = (labels_cpu == c).nonzero(as_tuple=True)[0]  # CPU int64
            if len(idx) >= k_shot + n_query:
                self.class_indices.append(idx)

        self.n_cls = len(self.class_indices)
        if self.n_cls < n_way:
            raise ValueError(
                f"Need {n_way} classes with >= {k_shot + n_query} samples, "
                f"only {self.n_cls} qualify."
            )

    def iter_episodes(
        self, n_episodes: int, seed: int = 42
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        生成 n_episodes 个 episode 的特征字典，特征已在 GPU 上。

        每个字典结构：{model_name: (n_way*(k_shot+n_query), feat_dim)}
        前 n_way*k_shot 行为 support，其余为 query（与原 DataLoader 版本一致）。
        """
        gen = torch.Generator()  # CPU generator，保证跨平台可复现
        gen.manual_seed(seed)
        for _ in range(n_episodes):
            yield self._sample_one(gen)

    def _sample_one(self, gen: torch.Generator) -> Dict[str, torch.Tensor]:
        # 随机选 n_way 个类（CPU 操作）
        cls_perm = torch.randperm(self.n_cls, generator=gen)[: self.n_way].tolist()

        sup_parts: List[torch.Tensor] = []
        qry_parts: List[torch.Tensor] = []
        for ci in cls_perm:
            idx = self.class_indices[ci]                          # CPU int64
            perm = torch.randperm(len(idx), generator=gen)
            chosen = idx[perm[: self.k_shot + self.n_query]]     # CPU int64
            sup_parts.append(chosen[: self.k_shot])
            qry_parts.append(chosen[self.k_shot :])

        # 组合全局索引后一次性移至 GPU，执行单次 GPU 切片
        all_idx = torch.cat(sup_parts + qry_parts).to(self.device)
        return {k: v[all_idx] for k, v in self.features_dict.items()}
