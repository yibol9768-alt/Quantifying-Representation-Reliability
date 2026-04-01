"""
渐进式模型选择算法。

基于两项排序分数 U(m | S) = R̂(m)^α · (1 - D̂(m, S))^β
渐进式地从候选池中选出有序模型序列。
"""
import numpy as np
from typing import List, Dict, Tuple


def progressive_model_selection(
    relevance: Dict[str, float],
    cka_matrix: np.ndarray,
    model_names: List[str],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> List[str]:
    """
    算法：基于信息论两项分数的渐进式模型选择。

    Args:
        relevance:   {model_name: R̂(m)}  归一化任务相关性
        cka_matrix:  (N, N) CKA 相似度矩阵
        model_names: 模型名称列表（与 cka_matrix 行列对应）
        alpha:       任务相关性权重指数
        beta:        冗余惩罚权重指数

    Returns:
        ordered_models: 按引入顺序排列的模型名称列表
    """
    N = len(model_names)
    name_to_idx = {name: i for i, name in enumerate(model_names)}

    # Step 1: 第一个模型 = 任务相关性最高的
    first = max(model_names, key=lambda m: relevance[m])
    ordered = [first]
    remaining = set(model_names) - {first}

    print(f"  Step 1: Selected {first} (R̂={relevance[first]:.4f})")

    # Step 2-N: 渐进式选择
    for step in range(2, N + 1):
        best_model = None
        best_score = -float("inf")

        selected_indices = [name_to_idx[m] for m in ordered]

        for m in remaining:
            m_idx = name_to_idx[m]

            # 冗余代理：平均 CKA
            d_hat = np.mean([cka_matrix[m_idx, j] for j in selected_indices])

            # 两项排序分数
            r_hat = relevance[m]
            score = (r_hat ** alpha) * ((1.0 - d_hat) ** beta)

            if score > best_score:
                best_score = score
                best_model = m

        ordered.append(best_model)
        remaining.remove(best_model)
        print(f"  Step {step}: Selected {best_model} "
              f"(R̂={relevance[best_model]:.4f}, U={best_score:.4f})")

    return ordered


# ──────── 消融对照方法 ────────

def relevance_only_selection(
    relevance: Dict[str, float],
    model_names: List[str],
) -> List[str]:
    """消融实验：仅按任务相关性排序"""
    return sorted(model_names, key=lambda m: relevance[m], reverse=True)


def diversity_only_selection(
    cka_matrix: np.ndarray,
    model_names: List[str],
) -> List[str]:
    """消融实验：仅按多样性（低冗余）排序"""
    N = len(model_names)
    name_to_idx = {name: i for i, name in enumerate(model_names)}

    # 第一个：平均 CKA 最小的（最独特的）
    avg_cka = [np.mean([cka_matrix[i, j] for j in range(N) if j != i])
               for i in range(N)]
    first_idx = int(np.argmin(avg_cka))
    ordered = [model_names[first_idx]]
    remaining = set(model_names) - {ordered[0]}

    for _ in range(1, N):
        selected_indices = [name_to_idx[m] for m in ordered]
        best_model = min(
            remaining,
            key=lambda m: np.mean([cka_matrix[name_to_idx[m], j]
                                   for j in selected_indices])
        )
        ordered.append(best_model)
        remaining.remove(best_model)

    return ordered


def random_selection(model_names: List[str], seed: int = 42) -> List[str]:
    """对照实验：随机顺序"""
    rng = np.random.RandomState(seed)
    shuffled = list(model_names)
    rng.shuffle(shuffled)
    return shuffled
