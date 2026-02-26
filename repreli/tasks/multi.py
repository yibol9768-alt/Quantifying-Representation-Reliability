
from typing import Tuple, Dict
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from .downstream import DownstreamTask
from .cifar100_utils import sparse2coarse


class MultiDownstreamTask(DownstreamTask):

    def __init__(
            self,
            task_name: str,
            dataset: str,
            n_classes: int,
            use_coarse_label: bool = False,
            **kwargs
    ):
        if use_coarse_label and dataset == 'cifar100':
            assert n_classes == 20
            logging.info('Using coarse labels for CIFAR100')
        elif use_coarse_label and dataset == 'cifar10':
            raise ValueError('Coarse label is not supported for CIFAR10')

        self.name = task_name
        self.dataset = dataset
        self.n_classes = n_classes
        self.clf = None
        self.use_coarse_lable = use_coarse_label

    def fit(self, emb: np.ndarray, ds_label: np.ndarray):
        if self.use_coarse_lable:
            ds_label = sparse2coarse(ds_label)

        logging.info(f'Training classifier')
        self.clf = LogisticRegression(random_state=1234, max_iter=20)
        self.clf.fit(emb, ds_label)

    def run_test(self, emb: np.ndarray, ds_label: np.ndarray) \
            -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:

        if self.use_coarse_lable:
            ds_label = sparse2coarse(ds_label)

        num_data = len(emb)

        target_prob = np.zeros((1, num_data, self.n_classes))
        target_prob[0, np.arange(num_data), ds_label] = 1.0

        pred_prob = self.clf.predict_proba(emb)[None]

        task_idx = np.zeros((self.n_classes, len(emb))).astype(int)
        brier_score = np.sum((pred_prob - target_prob)**2, axis=2)
        pred_prob_errs = 1 - pred_prob[:, np.arange(num_data), ds_label]
        log_pred_prob = -np.log(1 - pred_prob_errs + 1e-10)

        downstream_err = {
            'task_idx': task_idx,
            'brier': brier_score,
            'pred_err': pred_prob_errs,
            'log_pred_prob': log_pred_prob,
        }
        return downstream_err, pred_prob, target_prob
