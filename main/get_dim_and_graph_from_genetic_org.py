import argparse
import os

import joblib
import networkx as nx

from parameters.selected_exp_names import selected_exp_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='xor')

    args = parser.parse_args()
    task_name = args.task

    base_path = '/home/labs/schneidmann/noamaz/modularity'
    #base_path = "/Volumes/noamaz/modularity"
    selected_params = selected_exp_names[task_name]['random']
    num_features = selected_params.num_selected_features
    folder_name = f'{num_features}_features'
    folder_path = f'{base_path}/{task_name}/{selected_params.source_folder}/requiered_features_genetic_models/{folder_name}'
    organisms_folder = f"{folder_path}/good_archs/teach_archs_models"
    saved_data_dict = None
    for data_dict_file_name in os.listdir(folder_path):
        if f"_data_for_{num_features}_features_good_archs.pkl" in data_dict_file_name and '._' not in data_dict_file_name:
            with open(f"{folder_path}/{data_dict_file_name}", 'rb') as fp:
                saved_data_dict = joblib.load(fp)
            break
    if saved_data_dict is None:
        raise ValueError("couldn't find data dict pkl")
    graphs_and_dims = []
    for org_pkl_name in os.listdir(organisms_folder):
        if '._' in org_pkl_name:
            continue
        try:
            with open(f"{organisms_folder}/{org_pkl_name}", 'rb+') as fp:
                original_org = joblib.load(fp)
            graphs_and_dims.append(
                [
                    nx.to_numpy_array(original_org.network, weight=None, dtype=int),
                    original_org.dimensions,
                ]
            )
        except:
            print(f'couldnt open {org_pkl_name}')
    if len(graphs_and_dims) > 0:
        saved_data_dict['graphs'] = graphs_and_dims
        with open(f"{folder_path}/{data_dict_file_name}", 'wb+') as fp:
            joblib.dump(saved_data_dict, fp)
        print(f'saved {len(graphs_and_dims)} graphs')
    else:
        print(f'couldnt open any org')

