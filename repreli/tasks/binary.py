
from typing import Tuple, Dict
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from .downstream import DownstreamTask
from .cifar100_utils import sparse2coarse


class BinaryDownstreamTask(DownstreamTask):

    def __init__(
            self,
            task_name: str,
            dataset: str,
            n_classes: int,
            use_coarse_label: bool = True,
            **kwargs
    ):
        if use_coarse_label and dataset == 'cifar100':
            assert n_classes == 20
            logging.info('Using coarse labels for CIFAR100')

        self.name = task_name
        self.dataset = dataset
        self.n_classes = n_classes
        self.clfs = {}
        self.use_coarse_label = use_coarse_label

    @staticmethod
    def get_data(emb, label, class0, class1):
        idx0 = (label == class0)
        idx1 = (label == class1)
        return emb[idx0], emb[idx1], idx0, idx1

    def fit(self, emb: np.ndarray, ds_label: np.ndarray):
        if self.use_coarse_label:
            ds_label = sparse2coarse(ds_label)

        for class0 in range(self.n_classes):
            for class1 in range(class0 + 1, self.n_classes):
                logging.info(f'Training {class0}/{class1} classifier')
                emb_train_class0, emb_train_class1, _, _ = self.get_data(emb, ds_label, class0, class1)
                clf = LogisticRegression(random_state=1234, max_iter=20)
                clf.fit(
                    X=np.concatenate((emb_train_class0, emb_train_class1), 0),
                    y=[0] * len(emb_train_class0) + [1] * len(emb_train_class1)
                )
                self.clfs[(class0, class1)] = clf
                self.clfs[(class1, class0)] = clf

    def run_test(self, emb: np.ndarray, ds_label: np.ndarray) \
            -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:

        if self.use_coarse_label:
            ds_label = sparse2coarse(ds_label)

        brier_score = np.empty((self.n_classes, len(emb))) * np.nan
        pred_prob_errs = np.empty((self.n_classes, len(emb))) * np.nan
        task_idx = np.full((self.n_classes, len(emb)), -1)

        task_brier_score = []
        task_pred_prob_errs = []
        task_log_pred_prob = []
        t_cnt = 0
        for class0 in range(self.n_classes):
            for class1 in range(class0 + 1, self.n_classes):
                clf = self.clfs[(class0, class1)]
                emb_test_class0, emb_test_class1, test_idx0, test_idx1 = self.get_data(emb, ds_label, class0, class1)

                task_idx[class1, test_idx0] = t_cnt
                task_idx[class0, test_idx1] = t_cnt
                t_cnt += 1

                probs0 = clf.predict_proba(emb_test_class0)[:, 0]
                probs1 = clf.predict_proba(emb_test_class1)[:, 1]

                b0 = 2*(1 - probs0)**2
                b1 = 2*(1 - probs1)**2
                brier_score[class1, test_idx0] = b0
                brier_score[class0, test_idx1] = b1

                pred_prob_errs[class1, test_idx0] = (1 - probs0)
                pred_prob_errs[class0, test_idx1] = (1 - probs1)

                task_brier_score.append(np.mean(np.concatenate((b0, b1))))
                p = np.concatenate((probs0, probs1))
                task_pred_prob_errs.append(1 - np.mean(p))
                task_log_pred_prob.append(-np.mean(np.log(p + 1e-10)))

        pred_prob = np.stack([1 - pred_prob_errs, pred_prob_errs], axis=-1)
        target_prob = np.zeros_like(pred_prob)
        target_prob[:, :, 0] = 1
        log_pred_prob = -np.log(1 - pred_prob_errs + 1e-10)

        downstream_err = {
            'task_idx': task_idx,
            'brier': brier_score,
            'pred_err': pred_prob_errs,
            'log_pred_prob': log_pred_prob,
        }
        return downstream_err, pred_prob, target_prob
