import argparse
import shutil
from datetime import datetime

import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--first_dim', default='3')
    parser.add_argument('--second_dim', default='4')
    args = parser.parse_args()
    first_dim = int(args.first_dim)
    second_dim = int(args.second_dim)

    base_path = '/Volumes/noamaz/modularity/retina/'
    base_path = '/home/labs/schneidmann/noamaz/modularity/retina'
    original_folder = f'dynamic_retina_3_layers'
    target_folder = f"retina_3_layers_{first_dim}_{second_dim}"
    target_all_res_csv_name = None
    original_csv_name = f'2023-08-15-11-09-19_all_results_from_teach_archs_results_with_motifs_no_duplicates.csv'
    original_csv = pd.read_csv(f"{base_path}/{original_folder}/{original_csv_name}").drop("Unnamed: 0", axis=1)
    relevant_original_csv = original_csv[
        original_csv['neurons_in_layer_1'] == first_dim][
        original_csv['neurons_in_layer_2'] == second_dim
        ]
    for exp in relevant_original_csv['exp_name']:
        shutil.copyfile(f"{base_path}/{original_folder}/teach_archs_models/{exp}.pkl",
                        f"{base_path}/{target_folder}/teach_archs_models/{exp}.pkl")
        shutil.copyfile(f"{base_path}/{original_folder}/teach_archs_results/{exp}_teach.csv",
                        f"{base_path}/{target_folder}/teach_archs_results/{exp}_teach.csv")

    if target_all_res_csv_name:
        target_csv = pd.read_csv(f"{base_path}/{target_folder}/{target_all_res_csv_name}").drop("Unnamed: 0", axis=1)
        first_analysis_df = pd.concat([target_csv, relevant_original_csv],
                                      ignore_index=True)
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        csv_name = f"{time_str}_all_results_from_teach_archs_results_with_motifs"
        first_analysis_df.to_csv(f'{base_path}/{target_folder}/{csv_name}.csv')

        all_res_no_duplicates = first_analysis_df.drop_duplicates(subset=[
            'modularity',
            'num_connections',
            'entropy',
            'normed_entropy',
            'connectivity_ratio',
            'num_neurons',
            'max_possible_connections',
            'motifs_count_0',
            'motifs_count_1',
            'motifs_count_2',
            'neurons_in_layer_0',
            'neurons_in_layer_1',
            'neurons_in_layer_2',
            'neurons_in_layer_3',
        ])
        all_res_no_duplicates.to_csv(
            f'{base_path}/{target_folder}/{csv_name}_no_duplicates.csv')
