from datetime import datetime as dt

from analsis.analsis_utils.plot_utils import plot_loss_and_r2s_for_selected_feature_numbers, \
    plot_mean_performance_bars_for_num_features

from analsis.analsis_utils.utils import get_all_num_features_results_and_desired_external_exp
import os

base_path = '/Volumes/noamaz/modularity'
res_folders = [
    f"{base_path}/teach_archs/retina/retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/teach_archs_regression_feature_selection_results/2023-05-22-11-35_500_eph",
    f"{base_path}/teach_archs/retina/retina_xgboost_feature_selection/exp_2023-04-29-16-49-26/teach_archs_regression_feature_selection_results/2023-05-22-11-35_500_eph/",
]
regressors = [
    'lightgbm',
    'xgboost',
]

local_base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
fig_out_folder = f"retina_feature_selection/"
lr = 0.001
task = "retina"
start_idx = 200
end_idx = 1000
results_path = f"{base_path}/teach_archs/retina/retina_feature_selection/teach_archs_regression_feature_selection_results"

for csv_name in os.listdir(results_path):
    if ".csv" not in csv_name:
        continue
    csv_to_compere = f"{results_path}/{csv_name}"
    num_features = csv_name.split('_features')[0].split('_')[-1]
    rounded_num_features = int(round(int(num_features) / 5.0) * 5.0)
    model_name = csv_name.split('features_')[1].split('.csv')[0]
    comparison_name = f"{model_name}({num_features})"
    desiered_num_features = ['85_lightgbm', '85_xgboost', f'{rounded_num_features}_lightgbm',
                             f'{rounded_num_features}_xgboost', comparison_name]
    print(comparison_name     )
    all_results_dict, res_df = get_all_num_features_results_and_desired_external_exp(
        comparison_name=comparison_name,
        csv_to_compere_path=csv_to_compere,
        res_folders=res_folders,
        regressors=regressors,
        start_idx=start_idx,
        end_idx=end_idx
    )
    plot_loss_and_r2s_for_selected_feature_numbers(
        desiered_num_features=desiered_num_features,
        all_results_dict=all_results_dict,
        cut=15,
        on_end=False,
        lr=lr,
        task=task,
        time_str=time_str,
        local_base_path=local_base_path,
        fig_out_folder=fig_out_folder,
        regressor='',
        sufix=model_name,
    )
    plot_mean_performance_bars_for_num_features(
        desiered_num_features=desiered_num_features,
        all_results_dict=all_results_dict,
        start_ind=800,
        end_ind=1000,
        task=task,
        time_str=time_str,
        local_base_path=local_base_path,
        fig_out_folder=fig_out_folder,
        sufix=model_name,
    )