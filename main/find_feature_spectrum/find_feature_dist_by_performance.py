from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from find_feature_spectrum.find_feature_dist import FindFeaturesDist

feature_normelization_mapping = {
    'connectivity_ratio': lambda x: round(x, 2),
    'distances_between_input_neuron': lambda x: round(x * 2) / 2,
    'entropy': lambda x: x,
    'in_connections_per_layer': lambda x: round(x),
    'layer_connectivity_rank': lambda x: round(x),
    'max_connectivity_between_layers_per_layer': lambda x: round(x, 2),
    'max_possible_connections': lambda x: round(x),
    'modularity': lambda x: x,
    'normed_entropy': lambda x: x,
    'num_connections': lambda x: round(x),
    'num_involved_neurons_in_paths_per_input_neuron': lambda x: round(x),
    'num_paths_to_output_per_input_neuron': lambda x: round(x),
    'out_connections_per_layer': lambda x: round(x),
    'total_connectivity_ratio_between_layers': lambda x: round(x, 2),
}


class FindFeaturesDistByPerformance(FindFeaturesDist):
    def __init__(self,
                 num_features: int,
                 samples_path: str,
                 min_range_ind: int = -3,
                 max_range_ind: int = -1,
                 target_label_ranges: Optional[List[float]] = None,
                 ):
        super().__init__(num_features, samples_path)
        self.labels_range = self._get_target_label_range(
            target_label_ranges=target_label_ranges,
            min_range_ind=min_range_ind,
            max_range_ind=max_range_ind,
        )
        self.samples_to_model, self.other_samples = self._select_samples_in_range()
        self.means, self.covs, self.inv_covs = self._calc_mean_cov_cov_inv_of_gaussian()
        self.target_mean_features = self.means.to_list()

    def _get_target_label_range(
            self,
            target_label_ranges: Optional[List[float]],
            min_range_ind: int,
            max_range_ind: int,
    ) -> Tuple[float, float]:
        if target_label_ranges is not None:
            labels = (target_label_ranges[min_range_ind], target_label_ranges[max_range_ind])
        elif self.samples is not None:
            performances = pd.DataFrame([p for (s, p) in self.samples])
            ranges = np.linspace(0, 1, 11)
            performances_q = performances[performances[0].between(
                performances[0].quantile(ranges[min_range_ind]),
                performances[0].quantile(ranges[max_range_ind])
            )]
            labels = float(performances_q.min()), float(performances_q.max())
        else:
            raise ValueError("Either 'target_label_path' or 'samples' should be defined")
        print(f'labels ranges: {labels}')
        return labels

    def _select_samples_in_range(
            self,
    ) -> Tuple[pd.DataFrame, List[np.ndarray]]:
        top_samples = []
        other_samples = []
        for s in self.samples:
            if self.labels_range[0] <= s[1] <= self.labels_range[1]:
                top_samples.append(s[0])
            else:
                other_samples.append(s[0].numpy())
        return pd.DataFrame(top_samples).astype(float), other_samples

    @staticmethod
    def normalize_generated_sample(
            sample: np.ndarray,
            selected_feature_names: List[str],
    ) -> np.ndarray:
        for ind, x in enumerate(sample):
            feature_name = selected_feature_names[ind]
            for optional_feature_name, feature_norm_function in feature_normelization_mapping.items():
                if optional_feature_name in ['connectivity_ratio', 'entropy'] and feature_name != optional_feature_name:
                    continue
                if optional_feature_name in feature_name:
                    sample[ind] = feature_normelization_mapping[optional_feature_name](x)
                    break
        return sample

    def get_errors(
            self,
            num_features: int,
            frec: float = 0.1,
    ) -> np.ndarray:
        means = np.zeros((100, num_features))
        for i in range(100):
            means[i] = np.mean(
                self.samples_to_model.sample(frac=frec, axis=0, random_state=i), axis=0)
        return np.std(means, axis=0)
