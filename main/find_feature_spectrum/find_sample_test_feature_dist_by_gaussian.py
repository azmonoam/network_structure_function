from datetime import datetime as dt
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from matplotlib.patches import Patch
from scipy.stats import multivariate_normal
from tqdm import tqdm

from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDistByPerformance


class FindSampleTestFeaturesDistByGaussian(FindFeaturesDistByPerformance):
    def __init__(
            self,
            num_features: int,
            samples_path: str,
            predictive_model_path: str,
            plots_path: str,
            num_stds: float = 1,
            num_experiments: int = 1500,
            min_range_ind: int = -3,
            max_range_ind: int = -1,
            target_label_ranges: Optional[List[float]] = None,
    ):
        super().__init__(num_features, samples_path, min_range_ind, max_range_ind, target_label_ranges)
        self.time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.num_stds = num_stds
        self.num_experiments = num_experiments
        self.plots_path = plots_path
        with open(predictive_model_path, 'rb') as fp:
            self.predictive_model = joblib.load(fp)
        self.required_performance_min = self.labels_range[0] / 1000
        self.required_performance_max = self.labels_range[1] / 1000

    def sample_from_gaussian_and_predict(
            self,
            normalize: bool = False,
    ) -> pd.DataFrame:
        multivariate_gaussian = self._get_gaussian()
        res_df = pd.DataFrame()
        for _ in tqdm(range(self.num_experiments)):
            random_sample = multivariate_gaussian.rvs()
            while self.calc_distance_from_mean(
                    sample=random_sample
            ) > self.num_stds:
                random_sample = multivariate_gaussian.rvs()
            random_sample = torch.tensor(random_sample).to(torch.float32)
            if normalize:
                random_sample = self.normalize_generated_sample(
                    sample=random_sample,
                )
            prediction = self.predictive_model(random_sample).item()
            prediction = prediction / 1000
            res_dict = {
                'prediction': prediction,
                'required_performance_min': self.required_performance_min,
                'required_performance_max': self.required_performance_max,
                'is_within_required_performance_range':
                    (self.required_performance_min <= prediction <= self.required_performance_max),
            }
            res_df = pd.concat([res_df, pd.DataFrame(res_dict, index=[0], )], ignore_index=True)

        return res_df

    @staticmethod
    def remove_outliers(
            res_df: pd.DataFrame,
            min_percentile: float = 0.01,
            max_percentile: float = 0.99,
    ) -> pd.DataFrame:
        return res_df['prediction'][
            res_df['prediction'].between(
                res_df['prediction'].quantile(min_percentile),
                res_df['prediction'].quantile(max_percentile)
            )
        ].sort_values()

    def plot_prediction_vs_target(
            self,
            predictions: pd.DataFrame,
    ):
        required_performance_diff = self.required_performance_max - self.required_performance_min

        fig = plt.figure()
        ax = fig.add_subplot(111, )
        n, bins, patches = ax.hist(
            x=predictions,
            bins=np.arange(
                self.required_performance_min - (7 * required_performance_diff),
                self.required_performance_max + (8 * required_performance_diff),
                required_performance_diff,
            ),
            color='#4F6272',
        )
        patches[7].set_facecolor('#B7C3F3')
        h = [Patch(facecolor='#B7C3F3', label='Color Patch'), patches]
        ax.legend(h, ['target bin', 'predictions', ])
        plt.xlabel('predicted mean performance')
        plt.title(
            f'Predicted mean performance of architectures with structural features drown from a multivariate gaussian '
            f'distribution (distance from mean < {self.num_stds} stds) of the target performance data (target {round(self.required_performance_min, 4)}-'
            f'{round(self.required_performance_max, 4)})',
            wrap=True,
        )
        plt.savefig(
            f'{self.plots_path}/{self.time_str}_predicted_mean_performance_of_arch_from_multi_gaussian_'
            f'{self.num_features}_features_{self.num_stds}_stds.png')
        plt.show()

    def get_negative_samples(self):
        negative_samples = []
        for sample in self.other_samples:
            if self.calc_distance_from_mean(
                    sample=sample
            ) <= self.num_stds:
                negative_samples.append(sample)
        return negative_samples

    # TODO only relevant to 20 features
    @staticmethod
    def normalize_generated_sample(
            sample: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(6, 13):
            sample[i] = torch.round(sample[i] * 2) / 2
        for i in range(13, 20):
            sample[i] = torch.round(sample[i])
        return sample
