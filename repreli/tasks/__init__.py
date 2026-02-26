
import os
import logging
import pickle

import numpy as np
import pandas as pd

from .downstream import DownstreamTask
from .binary import BinaryDownstreamTask
from .multi import MultiDownstreamTask

SAVE_DIR = './downstream'


def get_task(task_name: str, task_type: str) -> DownstreamTask:
    task_name = task_name.lower()
    task_type = task_type.lower()

    if task_type == 'binary':
        task_cls = BinaryDownstreamTask
    elif task_type == 'multi':
        task_cls = MultiDownstreamTask
    else:
        raise NotImplementedError

    if task_name in {'cifar10', 'stl10'}:
        task = task_cls(task_name, task_name, n_classes=10, use_coarse_label=False)
    elif task_name == 'cifar100':
        task = task_cls(task_name, 'cifar100', n_classes=20, use_coarse_label=True)
    else:
        raise NotImplementedError
    return task


def run_downstreams(args, load: bool = True) -> pd.DataFrame:

    # Load
    save_path = f'{SAVE_DIR}/{args.ssl}_{args.arch}_{args.pretrain}_{args.downstream}_{args.task_type}.csv'
    if load and os.path.exists(save_path):
        logging.info(f'Downstream results are loaded from {save_path}.')
        return pd.read_csv(save_path, index_col=False)

    # Run Downstream Tasks
    task = get_task(args.downstream, args.task_type)

    logging.info('Start running downstream task ....')
    downstream_err_list = []
    pred_prob_list = []
    target_prob_list = []
    for train_pth, test_pth in zip(args.downstream_train_paths, args.downstream_test_paths):
        with open(train_pth, 'rb') as f:
            data = pickle.load(f)
            rep_ref = data['emb']
            lab_ref = data['label']

        with open(test_pth, 'rb') as f:
            data = pickle.load(f)
            rep_test = data['emb']
            lab_test = data['label']

        task.fit(rep_ref, lab_ref)
        downstream_err, pred_prob, target_prob = task.run_test(rep_test, lab_test)

        downstream_err_list.append(downstream_err)
        pred_prob_list.append(pred_prob)
        target_prob_list.append(target_prob)

    task_idx = downstream_err_list[0]['task_idx']
    assert all(np.allclose(arr, target_prob) for arr in target_prob_list)
    assert all(np.allclose(err['task_idx'], task_idx) for err in downstream_err_list)
    logging.info('Complete performing downstream task.')

    # Compute global uncertainty
    p = np.mean(pred_prob_list, axis=0)  # (n_tasks, n_samples, n_classes)
    p_target = p[target_prob == 1].reshape(target_prob.shape[:2])
    pred_var = np.sum(np.var(pred_prob_list, axis=0), axis=-1)  # (n_tasks, n_samples)
    pred_ent = -np.sum(p * np.log2(p, out=np.zeros_like(p), where=(p != 0)), axis=-1)  # (n_tasks, n_samples)
    pred_brier = np.sum((p - target_prob)**2, axis=-1)  # (n_tasks, n_samples)

    global_errs = {
        'task_idx': task_idx,
        'pred_entropy': pred_ent,
        'brier': pred_brier,
        'pred_err': 1 - p_target,
        'log_pred_prob': np.log(p_target)
    }

    # Parse as DataFrame
    results = []
    n_tasks, n_samples = pred_var.shape
    for i in range(n_samples):
        for j in range(n_tasks):
            if np.isnan(pred_var[j, i]):
                continue

            for i_model, err_dict in enumerate([global_errs, *downstream_err_list]):
                results.append({
                    'data_idx': i,
                    'model_idx': i_model - 1,
                    **{key: value[j, i] for key, value in err_dict.items()}
                })

    # Save
    results = pd.DataFrame(results)
    results.to_csv(save_path, index=False)
    return results
