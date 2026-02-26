
from typing import Dict, Union
import os
import uuid

import numpy as np
import pandas as pd
from scipy import stats


class ResultStorage:
    def __init__(self):
        self.scores = {}
        self.data = {}

    def add_score(self, score: np.ndarray, data: Dict[str, Union[str, int, float]]):
        # generate uuid name
        name = str(uuid.uuid4())
        while name in self.data:
            name = str(uuid.uuid4())

        self.scores[name] = score
        self.data[name] = data

    def update_configs(self, configs: Dict[str, Union[str, int, float]]):
        for name, data in self.data.items():
            data.update(configs)
    

class Evaluator:
    def __init__(self):
        self.scores = {}
        self.data = {}
        self.correlations = {}

    def merge(self, other: ResultStorage):
        self.scores.update(other.scores)
        self.data.update(other.data)

    def eval(self, downstream_errs: Dict[str, np.ndarray], model_id: int):
        corr = {}
        for name, score in self.scores.items():
            corr[name] = {}
            for key, errs in downstream_errs.items():
                # sanity check
                invalid = (np.isnan(score) | np.isnan(errs) | np.isinf(score) | np.isinf(errs))
                if invalid.any():
                    # raise ValueError('NaN or Inf found')
                    print(self.data[name])
                    print(name, key, np.isnan(score), np.isnan(errs), np.isinf(score), np.isinf(errs))
                    # continue

                # correlations
                tau = stats.kendalltau(score, -errs)[0]
                spearman = stats.spearmanr(score, -errs)[0]

                corr[name].update({
                    f'{key}-tau': tau,
                    f'{key}-spearman': spearman,
                })
        self.correlations[model_id] = corr

    def save_score_model(self, path: str) -> pd.DataFrame:
        self.save_score(path + '_score')
        return self.save_model(path + '_model')

    def save_model(self, path: str) -> pd.DataFrame:
        df = pd.DataFrame(self.data).T
        return self.safe_save(df, path)

    def load_model(self) -> pd.DataFrame:
        return pd.DataFrame(self.data).T

    def save_score(self, path: str):
        for name, score in self.scores.items():
            df = pd.DataFrame({'score': score})
            df['data_idx'] = np.arange(len(df))
            df['method_uuid'] = name
            df.to_csv(path + '.csv', mode='a', header=not os.path.exists(path + '.csv'), index=False)

    def save_corr(self, df_model: pd.DataFrame, path: str):
        dfs = []
        for id_, corr in self.correlations.items():
            df = pd.DataFrame(corr).T
            df['i_model'] = id_
            df = pd.merge(df, df_model, left_index=True, right_index=True, how='left')
            df = df[(df['i_model'] == -1) | pd.isna(df['i_ens']) | (df['i_model'] == df['i_ens'])]
            dfs.append(df)
        self.safe_save(pd.concat(dfs, axis=0), path)

    @ staticmethod
    def safe_save(df: pd.DataFrame, path: str, append=False) -> pd.DataFrame:
        if os.path.exists(path + '.csv') and append:
            df_ref = pd.read_csv(path + '.csv')
            df.reindex(columns=df_ref.columns, fill_value=np.nan).to_csv(path + '.csv', mode='a', header=False)
        else:
            df.to_csv(path + '.csv', mode='w')
        return df
