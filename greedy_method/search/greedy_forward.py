"""
search/greedy_forward.py
========================
贪心前向启发式搜索（论文第3节）。

核心改变：打分标准完全基于 few-shot 情节准确率，不使用全量数据或 MLP 分类头。

算法流程
--------
给定候选池 C = {m_1, ..., m_N}：

  步骤 k=1：
    对每个模型单独训练（few-shot 情节），在 val 情节上打分。
    选出最优 m* → S_1 = {m*}。

  步骤 k≥2：
    固定 S_{k-1}，遍历 C∖S_{k-1} 中每个候选 m_i：
      用 S_{k-1}∪{m_i} 的融合特征训练 few-shot 情节，val 情节打分。
    选出使 val 准确率最高的 m_i* → S_k = S_{k-1}∪{m_i*}。

  重复直到所有 N 个模型全部加入。

打分机制：
  训练：在 train 特征上采样情节，更新 LinearProjection + fusion_weights。
  打分：在 val 特征上采样情节，用原型欧氏距离分类，统计平均情节准确率。
  最终测试：在 test 特征上采样情节，记录每步的 few-shot test 准确率。
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from configs.config import TrainingConfig
from engine.feature_cache import FeatureCache
from engine.few_shot_engine import FewShotEngine, build_episode_loader
from models.fusion_network import MultiViewFusion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
class GreedySearchResult:
    """存储贪心搜索的完整输出。"""

    def __init__(self) -> None:
        self.selection_order: List[str]   = []  # 每步加入的模型名
        self.val_accuracies:  List[float] = []  # val 情节平均准确率（打分依据）
        self.test_accuracies: List[float] = []  # test 情节平均准确率（最终报告）
        self.fusion_weights:  List[Dict]  = []  # 每步的 α_i 分布

    def to_dict(self) -> dict:
        return {
            "selection_order": self.selection_order,
            "val_accuracies":  self.val_accuracies,
            "test_accuracies": self.test_accuracies,
            "fusion_weights":  self.fusion_weights,
        }


# ---------------------------------------------------------------------------
# 贪心搜索主体
# ---------------------------------------------------------------------------
class GreedyForwardSearch:
    """
    在单个数据集上执行贪心前向搜索。
    所有编码器特征必须在调用 run() 前通过 FeatureCache 预缓存完毕。
    """

    def __init__(
        self,
        candidate_models: List[str],
        cache:            FeatureCache,
        cfg:              TrainingConfig,
        device:           torch.device,
    ) -> None:
        self.candidate_models = list(candidate_models)
        self.cfg    = cfg
        self.device = device

        logger.info("加载全部模型的缓存特征 …")
        self.train_feats, self.train_labels = cache.load_split("train", candidate_models)
        self.val_feats,   self.val_labels   = cache.load_split("val",   candidate_models)
        self.test_feats,  self.test_labels  = cache.load_split("test",  candidate_models)

    # ------------------------------------------------------------------
    def run(self) -> GreedySearchResult:
        result    = GreedySearchResult()
        remaining = list(self.candidate_models)
        selected: List[str] = []

        for step in range(len(self.candidate_models)):
            logger.info(
                f"\n{'='*60}\n"
                f"贪心步骤 {step+1}/{len(self.candidate_models)}\n"
                f"已选集合 : {selected}\n"
                f"剩余候选 : {remaining}\n"
                f"{'='*60}"
            )

            best_model   = None
            best_val_acc = -1.0
            best_fusion_state = None

            for candidate in remaining:
                trial_set = selected + [candidate]
                logger.info(f"  评估组合: {trial_set}")

                val_acc, fusion_state = self._train_and_eval(trial_set)
                logger.info(f"    few-shot val_acc = {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc       = val_acc
                    best_model         = candidate
                    best_fusion_state  = fusion_state

            selected.append(best_model)
            remaining.remove(best_model)

            test_acc = self._compute_test_acc(selected, best_fusion_state)
            alphas   = self._get_alphas(best_fusion_state)

            result.selection_order.append(best_model)
            result.val_accuracies.append(best_val_acc)
            result.test_accuracies.append(test_acc)
            result.fusion_weights.append(
                {name: float(a) for name, a in zip(selected, alphas)}
            )

            logger.info(
                f"  ✓ 选入: {best_model}  "
                f"val={best_val_acc:.4f}  test={test_acc:.4f}\n"
                f"  融合权重: "
                + "  ".join(
                    f"{n}={a:.3f}"
                    for n, a in result.fusion_weights[-1].items()
                )
            )

        return result

    # ------------------------------------------------------------------
    def _train_and_eval(self, model_subset: List[str]) -> Tuple[float, dict]:
        """
        用 few-shot 情节训练 fusion，在 val 情节上打分。
        返回 (val_acc, fusion_state_dict)。
        """
        cfg = self.cfg
        train_feats = {k: self.train_feats[k] for k in model_subset}
        val_feats   = {k: self.val_feats[k]   for k in model_subset}

        fusion = MultiViewFusion(model_subset, cfg.fusion_dim)
        engine = FewShotEngine(
            fusion       = fusion,
            n_way        = cfg.search_n_way,
            k_shot       = cfg.search_k_shot,
            n_query      = cfg.search_n_query,
            device       = self.device,
            lr           = cfg.learning_rate,
            weight_decay = cfg.weight_decay,
        )

        # 训练：每个 epoch 用不同 seed 生成新情节，防止重复
        for epoch in range(cfg.search_epochs):
            loader = build_episode_loader(
                features_dict = train_feats,
                labels        = self.train_labels,
                n_way         = cfg.search_n_way,
                k_shot        = cfg.search_k_shot,
                n_query       = cfg.search_n_query,
                n_episodes    = cfg.search_train_episodes,
                seed          = cfg.seed + epoch * 7,
            )
            engine.train_episodes(loader, cfg.search_train_episodes)

        # 打分：val 情节
        val_loader = build_episode_loader(
            features_dict = val_feats,
            labels        = self.val_labels,
            n_way         = cfg.search_n_way,
            k_shot        = cfg.search_k_shot,
            n_query       = cfg.search_n_query,
            n_episodes    = cfg.search_val_episodes,
            seed          = cfg.seed + 9999,
        )
        val_acc, _ = engine.evaluate_episodes(val_loader, cfg.search_val_episodes)

        return val_acc, deepcopy(fusion.state_dict())

    # ------------------------------------------------------------------
    def _compute_test_acc(
        self,
        model_subset:  List[str],
        fusion_state:  dict,
    ) -> float:
        """加载训好的 fusion，在 test 情节上评估，返回平均准确率。"""
        cfg = self.cfg
        test_feats = {k: self.test_feats[k] for k in model_subset}

        fusion = MultiViewFusion(model_subset, cfg.fusion_dim)
        fusion.load_state_dict(fusion_state)

        engine = FewShotEngine(
            fusion  = fusion,
            n_way   = cfg.search_n_way,
            k_shot  = cfg.search_k_shot,
            n_query = cfg.search_n_query,
            device  = self.device,
        )
        test_loader = build_episode_loader(
            features_dict = test_feats,
            labels        = self.test_labels,
            n_way         = cfg.search_n_way,
            k_shot        = cfg.search_k_shot,
            n_query       = cfg.search_n_query,
            n_episodes    = cfg.search_val_episodes,
            seed          = cfg.seed + 8888,
        )
        test_acc, _ = engine.evaluate_episodes(test_loader, cfg.search_val_episodes)
        return test_acc

    # ------------------------------------------------------------------
    @staticmethod
    def _get_alphas(fusion_state: dict) -> List[float]:
        w = fusion_state["fusion_weights"]
        return F.softmax(w, dim=0).tolist()
