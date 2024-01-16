from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import xgboost
from sklearn.feature_selection import RFE

from networks_teachability_regression.regression_tree_feature_selection import RegressionTreeFeatureSelection
from parameters.base_params import BaseParams


class XGBoostRegressionTreeFeatureSelection(RegressionTreeFeatureSelection):

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
            ind_to_drop: Optional[List[int]] = None,
    ):
        super().__init__(base_path_to_res, test_path, train_path, out_folder, out_path,
                         time_str, task_params, n_threads, exp_folder, models_folder, feature_names, features_to_drop,
                         features_list, exp_folder_name_addition, ind_to_drop)
        self.reg_alpha = reg_alpha
        self.eval_metric = 'rmse'
        self.model_name = 'xgboost'

    def regression_tree_feature_selection(
            self,
            features_numbers: Optional[List[int]] = None,
            step: int = 1,
            calc_original: bool = True,
            ind_to_drop: Optional[List[int]] = None,
            **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_inputs, test_labels, train_inputs, train_labels = self._get_train_test_data(
            ind_to_drop=ind_to_drop,
        )
        if calc_original:
            original_model = xgboost.XGBRegressor()
            original_model.fit(
                train_inputs,
                train_labels,
                eval_metric=self.eval_metric,
                eval_set=[(test_inputs, test_labels), (train_inputs, train_labels)],
            )
            original_num_features = original_model.n_features_in_
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
        all_number_of_features_results = self._build_models_for_different_number_of_features(
            original_num_features=original_num_features,
            train_inputs=train_inputs,
            train_labels=train_labels,
            test_inputs=test_inputs,
            test_labels=test_labels,
            features_numbers=features_numbers,
            step=step,
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

    def train_model(
            self,
            train_inputs,
            train_labels,
            test_inputs,
            test_labels,
            num_features,
            step: int,
            **kwargs,
    ) -> Tuple[RFE, int]:
        model = xgboost.XGBRegressor()
        model.fit(
            train_inputs,
            train_labels,
            eval_metric=self.eval_metric,
            eval_set=[(test_inputs, test_labels), (train_inputs, train_labels)],
        )
        rfe = RFE(
            estimator=model,
            n_features_to_select=num_features,
        )
        rfe = rfe.fit(
            train_inputs,
            train_labels,
        )
        return rfe, num_features
