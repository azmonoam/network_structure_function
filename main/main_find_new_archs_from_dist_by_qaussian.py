
from teach_arch_multiple_times import TeachArchMultiTime
from find_feature_spectrum.find_sample_test_feature_dist_by_gaussian import FindSampleTestFeaturesDistByGaussian
from utils.set_up_population_utils import get_organism_by_connectivity_ratio
from utils.tasks_params import RetinaParameters
import random
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
import pandas as pd
import torch
import numpy as np
import joblib

base_path = '/home/labs/schneidmann/noamaz/modularity'
# base_path = '/Volumes/noamaz/modularity'
local_base_path = '/'
plots_path = f'{local_base_path}/plots/retina_from_label_to_arch_to_label/gausiian_dist'
ret_path = f'{base_path}/teach_archs/retina'
predictive_model_path = f'{ret_path}/retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/teach_archs_regression_feature_selection_results/2023-05-22-11-35_500_eph/retina_2023-05-22-11-45-00_lr_0.001_bs_512_output_meta_only_20_features_model_cpu.pkl'
samples_path = f'{ret_path}/retina_train_test_data/all_data_20_features_with_preformance.pkl'
target_label_path = f"{ret_path}/retina_teach_archs_requiered_features_kernel_dist/20_features/2023-06-01-14-24-51_target_label_ranges.pkl"
results_path = f'{ret_path}/retina_teach_archs_requiered_features_gaussian_dist/20_features/sigma_0_5'
if __name__ == '__main__':
    task_params = RetinaParameters
    num_features = 20
    num_stds = 0.5
    used_features_csv_name = f'{base_path}/teach_archs/retina/retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/2023-05-04-12-30-24_used_features.csv'
    selected_features_df = pd.read_csv(used_features_csv_name).drop("Unnamed: 0",
                                                                    axis=1)
    selected_features = selected_features_df[selected_features_df.sum(axis=1) == num_features]
    selected_feature_names = selected_features[selected_features == 1].dropna(axis=1).columns
    mask_tensor = torch.tensor(selected_features.iloc[0]).to(torch.bool)
    find_feature_dist_by_qaussian = FindSampleTestFeaturesDistByGaussian(
        num_features=num_features,
        samples_path=samples_path,
        target_label_path=target_label_path,
        predictive_model_path=predictive_model_path,
        plots_path=plots_path,
        num_stds=num_stds,
        num_experiments=1000,
    )
    i = 0
    connectivity_ratio = round(random.uniform(.3, 1.), 2)
    num_exp_per_arch = 300
    while i < 800:
        organism = get_organism_by_connectivity_ratio(
            task_params=task_params,
            connectivity_ratio=connectivity_ratio,
            allowed_weights_values=task_params.allowed_weights_values,
            allowed_bias_values=task_params.allowed_bias_values,
        )

        structural_features_calculator = CalcStructuralFeatures(
            organism=organism,
        )
        organism = structural_features_calculator.calc_structural_features()
        organisms_features = np.array([
            val
            for i, val in enumerate(organism.structural_features.get_class_values())
            if mask_tensor.numpy()[i]
        ])
        if find_feature_dist_by_qaussian._calc_distance(
                sample=organisms_features
        ) <= num_stds:
            min_trs = round(find_feature_dist_by_qaussian.required_performance_min, 4)
            max_trs = round(find_feature_dist_by_qaussian.required_performance_max, 4)
            exp_name = f'{find_feature_dist_by_qaussian.time_str}_required_pref_{min_trs}_{max_trs}_density_{connectivity_ratio}'
            with open(f'{results_path}/models/{exp_name}.pkl', 'wb+') as fp:
                joblib.dump(organism, fp)
            output_path = f"{results_path}/results/{exp_name}_teach.csv"
            teach_arch = TeachArchMultiTime(
                exp_name=exp_name,
                output_path=output_path,
                model_cls=task_params.model_cls,
                learning_rate=task_params.learning_rate,
                num_epochs=task_params.num_epochs,
                num_exp_per_arch=num_exp_per_arch,
                task=task_params.task,
                activate=task_params.activate,
            )

            teach_arch.teach_arch_many_times_parallel(
                organism=organism,
            )
            results = pd.read_csv(output_path).drop("Unnamed: 0", axis=1)
            final_epoch_res = results[results['iterations'] == results['iterations'].max()]
            first_analysis = {
                'exp_name': exp_name,
                'median_performance': final_epoch_res['performance'].median(),
                'mean_performance': final_epoch_res['performance'].mean(),
                'performance_std': final_epoch_res['performance'].std(),
                'max_performance': final_epoch_res['performance'].max(),
                'required_performance_min': find_feature_dist_by_qaussian.required_performance_min,
                'required_performance_max': find_feature_dist_by_qaussian.required_performance_max,
                'is_within_required_performance_range':
                    (find_feature_dist_by_qaussian.required_performance_min <= final_epoch_res[
                        'performance'].mean() <= find_feature_dist_by_qaussian.required_performance_max),
                'density': connectivity_ratio,
            }
            pd.DataFrame.from_dict(first_analysis, orient='index').to_csv(
                f"{results_path}/results/{exp_name}_res_analysis.csv",
            )
            print(f'done {exp_name}')
            break
        i += 1
    if i == 800:
        raise "couldn't find arch"
