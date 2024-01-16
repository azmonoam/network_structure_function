from typing import Tuple, Dict

import joblib
import pandas as pd
import torch

from find_feature_spectrum.find_features_spectrum import FindFeaturesSpectrum
from new_organism import Organism
from prepare_adjacency_data import PrepareAdjacencyData
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from find_feature_dist_utils.tasks_params import TaskParameters


class FindFeaturesSpectrumByValueRange(FindFeaturesSpectrum):
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
    ):
        super().__init__(num_features, train_path, test_path, used_features_csv_name, results_folder, base_path,
                         task_params, target_label, )

    def find_arch_by_features_train_and_compere(
            self,
            num_exp_per_arch: int,
    ):
        selected_line = self._select_sample()
        labels_range = self._get_labels_range(
            selected_line=selected_line,
        )
        selected_arch = self.adj_vecs[selected_line.index.item()]
        density = selected_line['connectivity_ratio'].item()
        exp_name = self.exp_name_template.format(
            time=self.time_str,
            min_labels_range=labels_range[0],
            max_labels_range=labels_range[1],
            density=density,
        )
        ranges_dict = self._get_ranges_dict(
            labels_range=labels_range,
            connectivity_ratio=density,
            selected_line=selected_line,
        )
        try:
            organism = self._find_arch(
                exp_name=exp_name,
                ranges_dict=ranges_dict,
                selected_arch=selected_arch,
                density=density,
            )
        except TimeoutError:
            raise TimeoutError('could not find arch in time')

        self._save_models(
            exp_name=exp_name,
            ranges_dict=ranges_dict,
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
            labels_range=labels_range,
            density=density,
        )

    def _select_sample(self) -> pd.DataFrame:
        return self.structural_features_df[self.structural_features_df['connectivity_ratio'] != 1.0].sample(
            n=1).drop_duplicates()

    def _get_labels_range(
            self,
            selected_line: pd.DataFrame,
    ) -> Tuple[float, float]:
        labels_range = (
            selected_line[self.target_label].item() - self.structural_features_df[self.target_label].std(),
            selected_line[self.target_label].item() + self.structural_features_df[self.target_label].std()
        )
        return labels_range

    def _get_ranges_dict(
            self,
            labels_range: Tuple[float, float],
            connectivity_ratio: float,
            selected_line: pd.DataFrame,
    ) -> Dict[str, Tuple[float, float]]:
        data_df_within_labels_range = self.structural_features_df[
            self.structural_features_df[self.target_label].between(labels_range[0], labels_range[1])
        ]
        data_df_within_labels_range_by_connectivity = data_df_within_labels_range[
            data_df_within_labels_range['connectivity_ratio'] == connectivity_ratio].drop_duplicates()
        ranges_dict = {}
        for col in data_df_within_labels_range.columns:
            if col in [self.target_label, 'connectivity_ratio']:
                continue
            stand_dev = data_df_within_labels_range_by_connectivity[col].std()
            ranges_dict[col] = (
                round(selected_line[col].item() - stand_dev, 3), round(selected_line[col].item() + stand_dev, 3))
        return ranges_dict

    def _test_arch_is_equal_to_target(
            self,
            selected_arch: torch.Tensor,
            organism: Organism,
            exp_name: str,
    ):
        org_adj_vec = torch.reshape(
            torch.from_numpy(
                PrepareAdjacencyData.get_network_aj_matrix(
                    organism=organism,
                    weight=None,
                )
            ),
            (-1,)
        ).float()
        if torch.equal(org_adj_vec, selected_arch):
            with open(f"{self.results_path}/{self.results_folder}/results/{exp_name}_eql_archs.txt", "a") as file:
                file.write("randomly chosen archs is identical to the source arch\n")
            return True
        return False

    def _test_if_org_in_ranges(
            self,
            ranges_dict: Dict[str, Tuple[float, float]],
            organism: Organism,
    ):
        structural_features_dict = {
            k.replace(', ', '_'): v
            for k, v in organism.structural_features.get_features(
                layer_neuron_idx_mapping=organism.layer_neuron_idx_mapping,
            ).items()
        }
        for feature_name, feature_val in structural_features_dict.items():
            feature_val = round(feature_val, 3)
            if feature_name not in self.selected_feature_names or feature_name == 'connectivity_ratio':
                continue
            if feature_val < ranges_dict[feature_name][0] or feature_val > ranges_dict[feature_name][1]:
                return False
        return True

    def _find_arch(
            self,
            exp_name: str,
            ranges_dict: Dict[str, Tuple[float, float]],
            selected_arch: torch.Tensor,
            density: float,
    ) -> Organism:
        good_org = False
        i = 0
        while i < 800:
            organism = self._build_organism(
                connectivity_ratio=density,
            )
            if self._test_arch_is_equal_to_target(
                    selected_arch=selected_arch,
                    organism=organism,
                    exp_name=exp_name,
            ):
                continue
            structural_features_calculator = CalcStructuralFeatures(
                organism=organism,
            )
            organism = structural_features_calculator.calc_structural_features()
            good_org = self._test_if_org_in_ranges(
                organism=organism,
                ranges_dict=ranges_dict,
            )
            i += 1
            if good_org:
                return organism
        if not good_org:
            raise TimeoutError()

    def _save_models(
            self,
            exp_name: str,
            ranges_dict: Dict[str, Tuple[float, float]],
            organism: Organism,
    ):
        with open(f'{self.results_path}/{self.results_folder}/models/{exp_name}.pkl', 'wb+') as fp:
            joblib.dump(organism, fp)
        structural_features_dict = organism.structural_features.get_features(
            layer_neuron_idx_mapping=organism.layer_neuron_idx_mapping,
        )
        structural_features = [ranges_dict, structural_features_dict]
        with open(f'{self.results_path}/{self.results_folder}/models/{exp_name}_structural_features.pkl',
                  'wb+') as fp:
            joblib.dump(structural_features, fp)
