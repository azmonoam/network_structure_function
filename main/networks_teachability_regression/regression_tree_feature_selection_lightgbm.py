from pathlib import Path
from typing import List, Tuple, Optional

import lightgbm
import pandas as pd
from sklearn.feature_selection import RFE

from networks_teachability_regression.regression_tree_feature_selection import RegressionTreeFeatureSelection
from parameters.base_params import BaseParams


class LightGBMRegressionTreeFeatureSelection(RegressionTreeFeatureSelection):

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
            reg_alpha: float = 0,
            n_estimators: int = 100,
            ind_to_drop: Optional[List[int]] = None,
    ):
        super().__init__(base_path_to_res, test_path, train_path, out_folder, out_path,
                         time_str, task_params, n_threads, exp_folder, models_folder, feature_names, features_to_drop,
                         features_list, exp_folder_name_addition, ind_to_drop)

        self.model_name = 'LightGBM'
        self.reg_alpha = reg_alpha
        self.n_estimators = n_estimators
        self.eval_metric = 'mean_squared_error'
        self.eval_names = ['test', 'train']

    def regression_tree_feature_selection(
            self,
            features_numbers: Optional[List[int]] = None,
            step: int = 1,
            calc_original: bool = True,
            force_col_wise: bool = True,
            learning_rate: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_inputs, test_labels, train_inputs, train_labels = self._get_train_test_data()
        if calc_original:
            print("calc original...")
            original_model = lightgbm.LGBMRegressor(
                reg_alpha=self.reg_alpha,
                learning_rate=learning_rate,
                n_estimators=self.n_estimators,
                force_col_wise=force_col_wise,
            )
            original_model.fit(
                train_inputs,
                train_labels,
                eval_set=[(test_inputs, test_labels), (train_inputs, train_labels)],
                eval_metric=self.eval_metric,
                eval_names=self.eval_names,
                feature_name=self.feature_names,
            )
            original_num_features = original_model._n_features
            mask, res_dict = self._get_model_results(
                model=original_model,
                train_inputs=train_inputs,
                test_inputs=test_inputs,
                train_labels=train_labels,
                test_labels=test_labels,
                num_features=original_num_features,
                original=True,
            )
            res_df, models_df = self._log_and_save_results(
                mask=mask,
                num_features=original_num_features,
                res_dict=res_dict,
                model=original_model,
            )
        else:
            original_num_features = len(self.feature_names)
            res_df = pd.DataFrame()
            models_df = pd.DataFrame()
        if features_numbers is None:
            features_numbers = [1] + list(range(5, original_num_features, 5))
        print(f"num features_numbers: {len(features_numbers)}")
        all_number_of_features_results = self._build_models_for_different_number_of_features(
            original_num_features=original_num_features,
            train_inputs=train_inputs,
            train_labels=train_labels,
            test_inputs=test_inputs,
            test_labels=test_labels,
            features_numbers=features_numbers,
            step=step,
            reg_alpha=self.reg_alpha,
            learning_rate=learning_rate,
            force_col_wise=force_col_wise,
        )
        for rfe, num_features in all_number_of_features_results:
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
                model=rfe,
            )
            res_df = pd.concat([res_df, res_df_for_number_of_features], ignore_index=True)
            models_df = pd.concat([models_df, models_df_for_number_of_features], ignore_index=True)
        res_df = res_df.sort_values('num_features')
        return res_df, models_df

    def regression_tree_feature_selection_non_parllel(
            self,
            features_numbers: Optional[List[int]] = None,
            step: int = 1,
            calc_original: bool = True,
            force_col_wise: bool = True,
            learning_rate: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_inputs, test_labels, train_inputs, train_labels = self._get_train_test_data()
        if calc_original:
            print("calc original...")
            original_model = lightgbm.LGBMRegressor(
                reg_alpha=self.reg_alpha,
                learning_rate=learning_rate,
                n_estimators=self.n_estimators,
                force_col_wise=force_col_wise,
            )
            original_model.fit(
                train_inputs,
                train_labels,
                eval_set=[(test_inputs, test_labels), (train_inputs, train_labels)],
                eval_metric=self.eval_metric,
                eval_names=self.eval_names,
                feature_name=self.feature_names,
            )
            original_num_features = original_model._n_features
            mask, res_dict = self._get_model_results(
                model=original_model,
                train_inputs=train_inputs,
                test_inputs=test_inputs,
                train_labels=train_labels,
                test_labels=test_labels,
                num_features=original_num_features,
                original=True,
            )
            res_df, models_df = self._log_and_save_results(
                mask=mask,
                num_features=original_num_features,
                res_dict=res_dict,
                model=original_model,
            )
        else:
            original_num_features = len(self.feature_names)
            res_df = pd.DataFrame()
            models_df = pd.DataFrame()
        if features_numbers is None:
            features_numbers = [1] + list(range(5, original_num_features, 5))
        print(f"num features_numbers: {len(features_numbers)}")
        features_numbers = [
            i
            for i in features_numbers
            if i <= len(self.feature_names)
        ]
        all_number_of_features_results = []
        for num_features in features_numbers:
            all_number_of_features_results.append(
                self.train_model(
                    train_inputs,
                    train_labels,
                    test_inputs,
                    test_labels,
                    num_features,
                    step=step,
                    force_col_wise=force_col_wise,
                    reg_alpha=self.reg_alpha,
                    learning_rate=learning_rate,
                ))
        for rfe, num_features in all_number_of_features_results:
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
                model=rfe,
            )
            res_df = pd.concat([res_df, res_df_for_number_of_features], ignore_index=True)
            models_df = pd.concat([models_df, models_df_for_number_of_features], ignore_index=True)
        res_df = res_df.sort_values('num_features')
        return res_df, models_df

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
            **kwargs,
    ) -> Tuple[RFE, int]:
        print(f"num_features: {num_features}")
        model = lightgbm.LGBMRegressor(
            reg_alpha=reg_alpha,
            n_estimators=self.n_estimators,
            force_col_wise=force_col_wise,
            learning_rate=learning_rate
        )
        rfe = RFE(
            estimator=model,
            n_features_to_select=num_features,
            step=step
        )
        rfe = rfe.fit(
            train_inputs,
            train_labels,
        )
        return rfe, num_features
