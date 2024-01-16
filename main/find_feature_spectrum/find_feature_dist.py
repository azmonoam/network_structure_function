from typing import Tuple, List

import joblib
import torch

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import multivariate_normal


class FindFeaturesDist:
    def __init__(
            self,
            num_features: int,
            samples_path: str,
    ):
        self.num_features = num_features
        self.samples = self._get_all_archs_data(
            samples_path=samples_path,
        )
        self.samples_to_model = self._get_samples_df()
        self.means, self.covs, self.inv_covs = self._calc_mean_cov_cov_inv_of_gaussian()
        self.target_mean_features = self.means.to_list()

    def _calc_mean_cov_cov_inv_of_gaussian(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        means = self.samples_to_model.mean(axis=0)
        covs = self.samples_to_model.cov()
        inv_covs = np.linalg.inv(covs.to_numpy())
        return means, covs, inv_covs

    def _get_gaussian(self):
        return multivariate_normal(
            mean=self.means,
            cov=self.covs,
        )

    @staticmethod
    def _get_all_archs_data(
            samples_path: str,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        with open(samples_path, 'rb') as fp:
            samples = joblib.load(fp)
        return samples

    def _get_samples_df(
            self,
    ) -> pd.DataFrame:
        samples_list = [
            s
            for s, _ in self.samples
        ]
        return pd.DataFrame(samples_list).astype(float)


    def calc_distance_from_mean(
            self,
            sample: np.ndarray,
    ):
        return distance.mahalanobis(
            u=sample,
            v=self.means,
            VI=self.inv_covs
        )
