import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from joblib import Parallel, delayed, load, dump
from sklearn import metrics

from networks_teachability_regression.regression_tree_learn import decompose_tensor_list
from new_organism import Organism
from parameters.base_params import BaseParams
from stractural_features_models.calc_structural_features import CalcStructuralFeatures


class RegressionTreeFeatureSelection:

    def __init__(
            self,
            base_path_to_res: str,
            test_path: str,
            train_path: str,
            out_folder: str,
            out_path: str,
            time_str: str,
            task_params: BaseParams,
            n_threads: Optional[int] = None,
            exp_folder: Optional[Path] = None,
            models_folder: Optional[Path] = None,
            feature_names: Optional[List[str]] = None,
            features_to_drop: Optional[List[str]] = None,
            features_list: Optional[List[str]] = None,
            exp_folder_name_addition: str = '',
            ind_to_drop: Optional[List[int]] = None,
    ):
        self.COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']
        self.base_path_to_res = base_path_to_res
        self.test_path = test_path
        self.train_path = train_path
        self.task_params = task_params
        self.feature_names = feature_names
        if feature_names is None:
            self.feature_names = self._get_feature_names_mapping(
                features_list=features_list,
                features_to_drop=features_to_drop,
            )
        self.train_data, self.test_data = self._get_basic_train_test_data(
            base_path_to_res=base_path_to_res,
            test_path=test_path,
            train_path=train_path,
            ind_to_drop=ind_to_drop,
        )
        self.out_folder = out_folder
        self.out_path = out_path
        self.time_str = time_str
        self.task = task_params.task_global_name
        self.exp_name = f'exp_{self.time_str}' + exp_folder_name_addition
        self.exp_folder = exp_folder
        self.models_folder = models_folder
        if self.models_folder is None:
            self.exp_folder, self.models_folder = self._set_up_exp_folder()
        self.model_name = None
        try:
            self.num_cores = int(subprocess.run('nproc', capture_output=True).stdout)
        except FileNotFoundError:
            self.num_cores = n_threads
        print(f"-- using {self.num_cores} cores --")

    def _get_basic_train_test_data(
            self,
            base_path_to_res: str,
            test_path: str,
            train_path: str,
            ind_to_drop: Optional[List[int]],
    ):
        with open(f'{base_path_to_res}/{test_path}', 'rb') as fp:
            test_data = load(fp)
        with open(f'{base_path_to_res}/{train_path}', 'rb') as fp:
            train_data = load(fp)
        base_mask = [True for _ in range(test_data[0][0].shape[0])]
        if ind_to_drop is not None:
            for i in ind_to_drop:
                base_mask[i] = False
            base_mask = torch.tensor(base_mask)
            train_data = self.mask_tensors(
                mask_tensor=base_mask,
                data_tensors=train_data,
            )
            test_data = self.mask_tensors(
                mask_tensor=base_mask,
                data_tensors=test_data,
            )
        return train_data, test_data

    def _set_up_exp_folder(self) -> Tuple[Path, Path]:
        record_folder = Path(f"{self.out_path}")
        record_folder.mkdir(exist_ok=True)
        if self.exp_folder is None:
            current_experiment_folder = record_folder / self.exp_name
            if current_experiment_folder.is_dir():
                raise Exception('Experiment already exists')
        else:
            current_experiment_folder = record_folder / self.exp_folder
        current_experiment_folder.mkdir(exist_ok=True)
        models_folder = current_experiment_folder / 'masked_data_models'
        models_folder.mkdir(exist_ok=True)
        nn_results_folder = current_experiment_folder / 'teach_archs_regression_feature_selection_results'
        nn_results_folder.mkdir(exist_ok=True)
        return current_experiment_folder, models_folder

    def _set_up_plot_folder(
            self,
            local_base_path: str,
            allow_exp_folder_exist=False,
    ) -> Path:
        record_folder = Path(f"{local_base_path}/plots/{self.out_folder}/")
        record_folder.mkdir(exist_ok=True)
        plots_folder = record_folder / self.exp_name
        if not allow_exp_folder_exist and plots_folder.is_dir():
            raise Exception('Experiment already exists')
        plots_folder.mkdir(exist_ok=True)
        return plots_folder

    @staticmethod
    def mask_tensors(
            mask_tensor: torch.Tensor,
            data_tensors: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.masked_select(inputs, mask_tensor), labels)
            for inputs, labels in data_tensors
        ]

    def _get_train_test_data(self) -> Tuple[np.ndarray, List[int], np.ndarray, List[int]]:
        print("loading test data")
        test_inputs, test_labels = decompose_tensor_list(
            tensor_list=self.test_data,
            ind_to_drop=None,
        )
        print("done loading test data")
        print("loading train data")
        train_inputs, train_labels = decompose_tensor_list(
            tensor_list=self.train_data,
            ind_to_drop=None,
        )
        print("done loading train data")
        return test_inputs, test_labels, train_inputs, train_labels

    def _get_model_results(
            self,
            model,
            train_inputs: np.ndarray,
            test_inputs: np.ndarray,
            train_labels: List[int],
            test_labels: List[int],
            num_features: int,
            original: bool = False,
    ):
        predicted_train_inputs = model.predict(train_inputs)
        predicted_test_inputs = model.predict(test_inputs)
        train_r2 = metrics.r2_score(predicted_train_inputs, train_labels)
        test_r2 = metrics.r2_score(predicted_test_inputs, test_labels)
        print(f"{num_features} features train_r2: {train_r2}")
        print(f"{num_features} features test_r2: {test_r2}")
        train_mse = metrics.mean_squared_error(predicted_train_inputs, train_labels)
        test_mse = metrics.mean_squared_error(predicted_test_inputs, test_labels)
        print(f"{num_features} features train_mse: {train_mse}")
        print(f"{num_features} features test_mse: {test_mse}")
        train_mae = metrics.mean_absolute_error(predicted_train_inputs, train_labels)
        test_mae = metrics.mean_absolute_error(predicted_test_inputs, test_labels)
        print(f"{num_features} features train_mae: {train_mae}")
        print(f"{num_features} features test_mae: {test_mae}")
        train_mape = metrics.mean_absolute_percentage_error(predicted_train_inputs, train_labels)
        test_mape = metrics.mean_absolute_percentage_error(predicted_test_inputs, test_labels)
        print(f"{num_features} features train_mape: {train_mape}")
        print(f"{num_features} features test_mape: {test_mape}")
        if original:
            mask = [True] * len(self.feature_names)
        else:
            mask = model.support_
        model_name = f'{self.time_str}_masked_data_{num_features}_features.pkl'
        res_dict = {
            'num_features': num_features,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'model_name': model_name,
        }
        return mask, res_dict

    def _log_and_save_results(
            self,
            mask: List[bool],
            num_features: int,
            res_dict: Dict[str, Any],
            model,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mask_tens = torch.tensor(mask)
        masked_data = {
            "num_features": num_features,
            "mask": mask_tens,
            "selected_train_data": self.mask_tensors(
                mask_tensor=mask_tens,
                data_tensors=self.train_data,
            ),
            "selected_test_data": self.mask_tensors(
                mask_tensor=mask_tens,
                data_tensors=self.test_data,
            ),
            "selected_feature_names": np.array(self.feature_names)[mask].tolist(),
            "model": model,
        }
        model_path = f'{self.models_folder}/{res_dict["model_name"]}'
        with open(model_path, 'wb+') as fp:
            dump(masked_data, fp)
        res_df = pd.DataFrame(res_dict, index=[0], )
        models_df = pd.DataFrame([np.array(mask) * 1], columns=self.feature_names)
        return res_df, models_df,

    def regression_tree_feature_selection(
            self,
            features_numbers: Optional[List[int]] = None,
            step: int = 1,
            calc_original: bool = True,
            force_col_wise: bool = True,
    ):
        pass

    def train_model(
            self,
            train_inputs,
            train_labels,
            test_inputs,
            test_labels,
            num_features,
            step: int,
            force_col_wise: bool,
            reg_alpha: float,
            learning_rate: float,
            **kwargs: Any,
    ):
        pass

    def _build_models_for_different_number_of_features(
            self,
            train_inputs,
            train_labels,
            test_inputs,
            test_labels,
            features_numbers: List[int],
            step: int,
            reg_alpha: float,
            learning_rate: float,
            force_col_wise: bool,
            **kwargs: Any,
    ) -> List[Tuple[pd.DataFrame, int]]:
        features_numbers = [
            i
            for i in features_numbers
            if i < len(self.feature_names)
        ]
        print("starting Parallel")
        return Parallel(
            n_jobs=self.num_cores,
            timeout=99999,
        )(
            delayed
            (self.train_model)(
                train_inputs,
                train_labels,
                test_inputs,
                test_labels,
                num_features,
                step=step,
                force_col_wise=force_col_wise,
                reg_alpha=reg_alpha,
                learning_rate=learning_rate,
                **kwargs,
            )
            for num_features in features_numbers
        )

    def train_single_num_features(
            self,
            num_features: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_inputs, test_labels, train_inputs, train_labels = self._get_train_test_data()
        rfe, num_features = self.train_model(
            train_inputs=train_inputs,
            train_labels=train_labels,
            test_inputs=test_inputs,
            test_labels=test_labels,
            num_features=num_features,
        )
        mask, res_dict = self._get_model_results(
            model=rfe,
            train_inputs=train_inputs,
            test_inputs=test_inputs,
            train_labels=train_labels,
            test_labels=test_labels,
            num_features=num_features,
        )
        res_df_for_number_of_features, models_df_for_number_of_features = self._log_and_save_results(
            mask=mask,
            num_features=num_features,
            res_dict=res_dict,
        )
        return res_df_for_number_of_features, models_df_for_number_of_features

    def plot_r2_vs_num_features(
            self,
            res_df: pd.DataFrame,
            plots_folder: Path,
            metric_name: str = 'r2',
            name_add: str = '',
            xtick_dix: bool = False,
    ):
        plt.plot(res_df['num_features'], res_df[f'train_{metric_name}'], label='train', c=self.COLORS[0])
        plt.plot(res_df['num_features'], res_df[f'test_{metric_name}'], label='test', c=self.COLORS[1])
        plt.xlabel('num features')
        plt.ylabel(f'{metric_name}')
        plt.title(
            f"The model's {metric_name} for predicting architectures mean performance as a function of the number of features used.{self.model_name}, {self.task}",
            wrap=True,
        )
        plt.legend()
        if xtick_dix:
            ticks = np.linspace(min(res_df['num_features']) - 1, max(res_df['num_features']), 11)
            plt.xticks(ticks)
        plt.savefig(
            f"{plots_folder}/{self.time_str}_{self.task}_{metric_name}_vs_num_features_{self.model_name}_regression_feature_selection{name_add}.png")
        plt.show()

    def _get_feature_names_mapping(
            self,
            features_list: Optional[List[str]] = None,
            features_to_drop: Optional[List[str]] = None,
    ) -> List[str]:
        organism = Organism(
            dimensions=self.task_params.start_dimensions,
            num_layers=self.task_params.num_layers,
            allowed_weights_values=self.task_params.allowed_weights_values,
            allowed_bias_values=self.task_params.allowed_bias_values,
            communities_input_symmetry=self.task_params.communities_input_symmetry,
        )
        organism.build_organism_by_connectivity(
            max_allowed_connections=self.task_params.max_possible_connections,
        )
        structural_features_calculator = CalcStructuralFeatures(
            organism=organism,
        )
        organism = structural_features_calculator.calc_structural_features()

        feature_names = [
            f.replace(',', '_') for f in organism.structural_features.get_features(
                layer_neuron_idx_mapping=organism.layer_neuron_idx_mapping,
                features_list=features_list,
            ).keys()
        ]
        if features_to_drop is None:
            return feature_names
        for feature_to_drop in features_to_drop:
            modularity_index = feature_names.index(feature_to_drop)
            feature_names.pop(modularity_index)
        return feature_names
