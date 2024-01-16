from typing import List, Tuple, Optional

import joblib
import pandas as pd
from sklearn.neighbors import KernelDensity

from find_feature_spectrum.find_features_spectrum import FindFeaturesSpectrum
from new_organism import Organism
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from find_feature_dist_utils.tasks_params import TaskParameters
from find_feature_dist_utils.main_utils import softmax


class FindFeaturesSpectrumByKernelDist(FindFeaturesSpectrum):
    def __init__(
            self,
            num_features: int,
            train_path: str,
            test_path: str,
            used_features_csv_name: str,
            results_folder: str,
            base_path: str,
            task_params: TaskParameters,
            target_label: str = "mean performance",
            num_bins: int = 10,
    ):
        super().__init__(num_features, train_path, test_path, used_features_csv_name, results_folder, base_path,
                         task_params, target_label, )
        self.num_bins = num_bins
        self.binned_labels = range(self.num_bins)

    def _get_target_connectivity(self) -> float:
        return self.structural_features_df[self.structural_features_df['connectivity_ratio'] != 1.0].sample(n=1)[
            'connectivity_ratio'].item()

    def find_arch_by_features_train_and_compere(
            self,
            num_exp_per_arch: int,
            kernel_models: Optional[List[KernelDensity.fit]] = None,
            target_label_ranges: Optional[List[Tuple[float, float]]] = None,
    ):
        if kernel_models is None or target_label_ranges is None:
            kernel_models, target_label_ranges = self._get_full_space_distributions()
        try:
            organism, target_label_range, density = self._find_arch(
                kernel_models=kernel_models,
                target_label_ranges=target_label_ranges,
            )
        except TimeoutError:
            raise TimeoutError('could not find arch in time')
        exp_name = self.exp_name_template.format(
            time=self.time_str,
            min_labels_range=target_label_range[0],
            max_labels_range=target_label_range[1],
            density=density,
        )
        self._save_models(
            exp_name=exp_name,
            organism=organism,
        )
        output_path = f"{self.results_path}/{self.results_folder}/results/{exp_name}_teach.csv"
        self.tech_arch_many_times(
            organism=organism,
            exp_name=exp_name,
            output_path=output_path,
            num_exp_per_arch=num_exp_per_arch,
        )
        self.save_aggregated_results(
            exp_name=exp_name,
            output_path=output_path,
            labels_range=target_label_range,
            density=density,
        )

    def _get_full_space_distributions(self) -> Tuple[List[KernelDensity.fit], List[Tuple[float, float]]]:
        binns_mapping = pd.qcut(self.structural_features_df[self.target_label], q=self.num_bins,
                                labels=self.binned_labels)
        data_df_no_label = self.structural_features_df.drop(labels=self.target_label, axis=1)
        kernel_models = []
        target_label_ranges = []
        for i in range(self.num_bins):
            binned_data = data_df_no_label[binns_mapping == i]
            kernel_models.append(KernelDensity().fit(binned_data))
            target_label_col = self.structural_features_df[self.target_label][binns_mapping == i]
            target_label_ranges.append((target_label_col.min(), target_label_col.max()))
        return kernel_models, target_label_ranges

    def _find_archs_parent_distributions(
            self,
            organism: Organism,
            kernel_models: List[KernelDensity.fit],
            target_label_ranges: List[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], float]:
        features = [
            val
            for i, val in enumerate(organism.structural_features.get_class_values())
            if self.mask_tensor.numpy()[i]
        ]
        features_df = pd.DataFrame(
            {
                self.selected_feature_names[i]: val
                for i, val in enumerate(features)
            },
            index=[0]
        )
        probeblities = softmax([
            model.score_samples(features_df)
            for model in kernel_models
        ])
        disterbution = self.binned_labels[probeblities.argmax()]
        target_label_range = target_label_ranges[disterbution]
        prob = max(probeblities).item()
        return target_label_range, prob

    def _find_arch(
            self,
            target_label_ranges: List[Tuple[float, float]],
            kernel_models: List[KernelDensity.fit],
    ) -> Tuple[Organism, Tuple[float, float], float]:
        i = 0
        prob = 0
        while i < 800:
            density = self._get_target_connectivity()
            organism = self._build_organism(
                connectivity_ratio=density,
            )
            structural_features_calculator = CalcStructuralFeatures(
                organism=organism,
            )
            organism = structural_features_calculator.calc_structural_features()
            target_label_range, prob = self._find_archs_parent_distributions(
                organism=organism,
                kernel_models=kernel_models,
                target_label_ranges=target_label_ranges,
            )
            if prob > 0.5:
                return organism, target_label_range, density
            i += 1
        if prob <= 0.5:
            raise TimeoutError()

    def _save_models(
            self,
            exp_name: str,
            organism: Organism,
    ):
        with open(f'{self.results_path}/{self.results_folder}/models/{exp_name}.pkl', 'wb+') as fp:
            joblib.dump(organism, fp)
        structural_features_dict = organism.structural_features.get_features(
            layer_neuron_idx_mapping=organism.layer_neuron_idx_mapping,
        )
        with open(f'{self.results_path}/{self.results_folder}/models/{exp_name}_structural_features.pkl',
                  'wb+') as fp:
            joblib.dump(structural_features_dict, fp)
