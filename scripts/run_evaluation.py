#!/usr/bin/env python3
"""
运行评估，复用原 repreli 代码的逻辑，但使用新模型。

与原代码的区别：
- 原代码：同一模型训练10次（不同seed）
- 本代码：不同模型（CLIP、DINO、MAE等）

Usage:
    python scripts/run_evaluation.py --config configs/heterogeneous.yaml
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import normalize

# 添加原代码路径
sys.path.insert(0, str(Path(__file__).parent.parent / "repreli"))

# 导入原代码的模块
from tasks import get_task, SAVE_DIR as DOWNSTREAM_DIR
from evaluation.evaluate import evaluate_score
from evaluation.evaluator import Evaluator
from utils import df_to_dict, batch_cosine_distances, batch_euclidean_distances, fastSort, npSort, pairwise_jaccard


logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s - %(message)s")


def run_downstream_task(
    train_features: list,  # list of {'emb': ..., 'label': ...}
    test_features: list,
    dataset_name: str,
    task_type: str = "binary",
):
    """
    在多个模型的特征上运行下游任务。

    与原代码类似，但对每个模型分别训练线性头。
    """
    task = get_task(dataset_name, task_type)

    downstream_err_list = []
    pred_prob_list = []
    target_prob_list = []

    for i, (train_data, test_data) in enumerate(zip(train_features, test_features)):
        logging.info(f"Training downstream on model {i+1}/{len(train_features)}")

        rep_train = train_data['emb']
        lab_train = train_data['label']
        rep_test = test_data['emb']
        lab_test = test_data['label']

        # 训练线性头
        task.fit(rep_train, lab_train)
        downstream_err, pred_prob, target_prob = task.run_test(rep_test, lab_test)

        downstream_err_list.append(downstream_err)
        pred_prob_list.append(pred_prob)
        target_prob_list.append(target_prob)

    # 计算 ensemble 性能
    task_idx = downstream_err_list[0]['task_idx']
    p = np.mean(pred_prob_list, axis=0)
    p_target = p[target_prob == 1].reshape(target_prob.shape[:2])

    pred_ent = -np.sum(p * np.log2(p, out=np.zeros_like(p), where=(p != 0)), axis=-1)
    pred_brier = np.sum((p - target_prob)**2, axis=-1)

    global_errs = {
        'task_idx': task_idx,
        'pred_entropy': pred_ent,
        'brier': pred_brier,
        'pred_err': 1 - p_target,
    }

    # 解析为 DataFrame
    results = []
    n_tasks, n_samples = pred_ent.shape

    for i in range(n_samples):
        for j in range(n_tasks):
            if np.isnan(pred_ent[j, i]):
                continue
            for i_model, err_dict in enumerate([global_errs, *downstream_err_list]):
                results.append({
                    'data_idx': i,
                    'model_idx': i_model - 1,
                    **{key: value[j, i] for key, value in err_dict.items()}
                })

    return pd.DataFrame(results)


def compute_nc_scores(
    ref_rep_list: list,  # list of reference representations
    eval_rep_list: list,  # list of evaluation representations
    n_ref: int = 5000,
    k_list: list = [1, 100],
    distance: str = "cosine",
    seed: int = 42,
):
    """
    计算 NC 分数。

    复用原代码的 evaluate_score 逻辑，但简化为只计算 NC。
    """
    n_ensembles = len(ref_rep_list)
    n_test = eval_rep_list[0].shape[0]
    n_total_ref = ref_rep_list[0].shape[0]

    # 选择参考点
    np.random.seed(seed)
    if n_total_ref > n_ref:
        ref_idx = np.random.choice(n_total_ref, size=n_ref, replace=False)
    else:
        ref_idx = np.arange(n_total_ref)

    # 计算距离和邻居
    nb_idx_list = []
    d_sorted_list = []

    for ref_rep, eval_rep in zip(ref_rep_list, eval_rep_list):
        ref_subset = ref_rep[ref_idx]

        if distance == "cosine":
            ref_subset = normalize(ref_subset)
            eval_rep = normalize(eval_rep)
            d_mat = batch_cosine_distances(eval_rep, ref_subset)
        else:
            d_mat = batch_euclidean_distances(eval_rep, ref_subset)

        nb_idx, d_sorted = fastSort(d_mat)
        nb_idx_list.append(nb_idx)
        d_sorted_list.append(d_sorted)

    # 计算 NC 分数
    results = {}

    for k in k_list:
        knn_idx = np.array([nb_idx[:, :k] for nb_idx in nb_idx_list])
        nc_scores = np.zeros(n_test)

        for i_sample in range(n_test):
            jaccard_sims = pairwise_jaccard(knn_idx[:, i_sample])
            nc_scores[i_sample] = np.mean(jaccard_sims)

        results[f"NC_k{k}"] = nc_scores

    # 计算 Feature Variance
    stacked = np.stack(eval_rep_list, axis=0)
    if distance == "cosine":
        stacked = normalize(stacked.reshape(-1, stacked.shape[-1])).reshape(stacked.shape)
    fv_scores = -np.sum(np.var(stacked, axis=0), axis=-1)
    results["FV"] = fv_scores

    # 计算 Avg Dist
    d_sorted_arr = np.array(d_sorted_list)
    for k in k_list:
        avg_dist = np.mean(d_sorted_arr[:, :, :k], axis=-1)
        results[f"AvgDist_k{k}"] = -np.mean(avg_dist, axis=0)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with heterogeneous models")

    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--features_dir", type=str, default="./features", help="Features directory")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate")
    parser.add_argument("--n_ref", type=int, default=5000, help="Number of reference points")
    parser.add_argument("--k_list", nargs="+", type=int, default=[1, 100], help="k values")
    parser.add_argument("--distance", type=str, default="cosine", help="Distance metric")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")

    args = parser.parse_args()

    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        args.models = args.models or config.get("models")
        args.dataset = config.get("dataset", args.dataset)
        args.n_ref = config.get("n_ref", args.n_ref)
        args.k_list = config.get("k_list", args.k_list)
        args.distance = config.get("distance", args.distance)

    if args.models is None:
        print("Error: Please specify models")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(DOWNSTREAM_DIR, exist_ok=True)

    print("=" * 60)
    print("Heterogeneous Model Evaluation")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Dataset: {args.dataset}")
    print(f"n_ref: {args.n_ref}, k_list: {args.k_list}, distance: {args.distance}")
    print()

    # 加载特征
    print("[1/3] Loading features...")
    train_features = []
    test_features = []

    for model_name in args.models:
        train_path = os.path.join(args.features_dir, f"{model_name}_{args.dataset}_train.pkl")
        test_path = os.path.join(args.features_dir, f"{model_name}_{args.dataset}_test.pkl")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Warning: Features not found for {model_name}")
            continue

        with open(train_path, "rb") as f:
            train_features.append(pickle.load(f))
        with open(test_path, "rb") as f:
            test_features.append(pickle.load(f))

        print(f"  Loaded {model_name}: train {train_features[-1]['emb'].shape}, test {test_features[-1]['emb'].shape}")

    if len(train_features) < 2:
        print("Error: Need at least 2 models")
        sys.exit(1)

    # 运行下游任务
    print("\n[2/3] Running downstream tasks (training linear heads)...")
    downstream_df = run_downstream_task(
        train_features, test_features,
        dataset_name=args.dataset,
        task_type="binary"
    )

    # 计算 NC 分数
    print("\n[3/3] Computing NC scores...")
    ref_rep_list = [f['emb'] for f in train_features]
    eval_rep_list = [f['emb'] for f in test_features]

    nc_results = compute_nc_scores(
        ref_rep_list, eval_rep_list,
        n_ref=args.n_ref,
        k_list=args.k_list,
        distance=args.distance,
        seed=args.seed,
    )

    # 计算相关性
    print("\n[4/4] Computing correlations...")
    # 下游性能（ensemble 平均）
    ensemble_df = downstream_df[downstream_df.model_idx == -1].groupby('data_idx').mean()
    downstream_brier = -ensemble_df['brier'].values  # negative Brier score

    correlations = {}
    from scipy.stats import kendalltau

    for method, scores in nc_results.items():
        valid = ~(np.isnan(scores) | np.isnan(downstream_brier))
        if valid.sum() > 10:
            tau, p_val = kendalltau(scores[valid], downstream_brier[valid])
            correlations[method] = {"kendall_tau": tau, "p_value": p_val}
            print(f"  {method}: τ = {tau:.4f} (p = {p_val:.4f})")

    # 保存结果
    results = {
        "config": {
            "models": args.models,
            "dataset": args.dataset,
            "n_ref": args.n_ref,
            "k_list": args.k_list,
            "distance": args.distance,
            "seed": args.seed,
        },
        "nc_scores": {k: v.tolist() for k, v in nc_results.items()},
        "correlations": correlations,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = os.path.join(args.output_dir, f"results_{args.dataset}.yaml")
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"\nResults saved to: {output_path}")

    # 保存为 DataFrame 便于分析
    df_path = os.path.join(args.output_dir, f"nc_scores_{args.dataset}.csv")
    pd.DataFrame(nc_results).to_csv(df_path, index=False)
    print(f"NC scores saved to: {df_path}")


if __name__ == "__main__":
    main()
