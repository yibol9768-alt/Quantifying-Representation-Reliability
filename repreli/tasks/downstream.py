
from typing import Dict, Tuple
from abc import ABC, abstractmethod

import numpy as np


class DownstreamTask(ABC):

    @abstractmethod
    def fit(self, emb: np.ndarray, ds_label: np.ndarray):
        pass

    @abstractmethod
    def run_test(self, emb: np.ndarray, ds_label: np.ndarray) \
            -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        pass
