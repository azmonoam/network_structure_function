import argparse
import os
import random
import shutil
from _datetime import datetime as dt
from pathlib import Path

import joblib

from parameters.general_paramters import RANDOM_SEED
from parameters.selected_exp_names import selected_exp_names

if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='retina_xor')

    args = parser.parse_args()
    task_name = args.task_name
    if task_name == 'xor':
        folder_names = {
            (6, 5, 5, 4, 2): '6_5_5_4_2',
            (6, 6, 4, 4, 2): '6_6_4_4_2',
            (6, 5, 4, 5, 2): '6_5_4_5_2',
            (6, 4, 6, 4, 2): '6_4_6_4_2',
            (6, 6, 5, 3, 2): '6_6_5_3_2',
            (6, 4, 5, 5, 2): '6_4_5_5_2',
            (6, 5, 6, 3, 2): '6_5_6_3_2',
            (6, 6, 3, 5, 2): '6_6_3_5_2',
            (6, 4, 4, 6, 2): '6_4_4_6_2',
            (6, 5, 3, 6, 2): '6_5_3_6_2',
            (6, 3, 6, 5, 2): '6_3_6_5_2',
            (6, 3, 5, 6, 2): '6_3_5_6_2',
            (6, 6, 6, 2, 2): None,
            (6, 6, 2, 6, 2): None,
            (6, 2, 6, 6, 2): None,
        }
        total_num_graphs_to_sample = 5000
    elif task_name == 'retina_xor':
        folder_names = {
            (6, 5, 2, 2): "6_5_2_2",
            (6, 2, 5, 2): "6_2_5_2",
            (6, 3, 4, 2): "6_3_4_2",
            (6, 4, 3, 2): "6_4_3_2",
        }
        total_num_graphs_to_sample = 2500
    elif task_name == 'digits':
        folder_names = {
            (64, 5, 7, 10): "64_5_7_10",
            (64, 8, 4, 10): "64_8_4_10",
            (64, 6, 6, 10): "64_6_6_10",
            (64, 7, 5, 10): "64_7_5_10",
        }
        total_num_graphs_to_sample = 5000
    else:
        raise ValueError
    exp_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    out_folder_name = 'teach_archs_models_1s'
    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    num_layers = selected_params.num_layers
    folder_name = f'{num_features}_features'
    data_pkl = selected_params.ergm_init_graphs_pkl_file_name
    base_ergm_fold = f'{base_path}/{folder_path}/ergm/{folder_name}'
    base_gen_fold = f'{base_path}/{folder_path}//requiered_features_genetic_models/{folder_name}/good_archs'
    prepered_data_path = f'{base_ergm_fold}/{data_pkl}'
    with open(prepered_data_path, 'rb') as fp:
        prepered_data = joblib.load(fp)
    base_all_archs_fold = f'{base_gen_fold}/per_dim_results'
    data_prep = {}
    sum_num_graphs = 0
    all_probs = []
    for d, v in prepered_data["data_per_dim"].items():
        if folder_names.get(d) is None:
            continue
        sum_num_graphs += v['num_graphs']
    for d, v in prepered_data["data_per_dim"].items():
        prob = v['num_graphs'] / sum_num_graphs
        all_probs.append(prob)
        data_prep[d] = {
            'num_graphs_to_sample': 0
        }
    for dim in random.choices(list(data_prep.keys()), weights=all_probs, cum_weights=None,
                              k=total_num_graphs_to_sample):
        data_prep[dim]["num_graphs_to_sample"] += 1
    all_graphs = []
    out_folder = Path(f"{base_all_archs_fold}/{out_folder_name}/")
    out_folder.mkdir(exist_ok=True)
    didnt_save = {}
    for i, (dim, data) in enumerate(data_prep.items()):
        if folder_names.get(dim) is None:
            continue
        num_graphs_to_sample = data["num_graphs_to_sample"]
        models_path = f"{base_all_archs_fold}/{folder_names[dim]}/{out_folder_name}"
        chosen_graphs = []
        didnt_save_list = []
        for org_name in random.sample(os.listdir(models_path), k=num_graphs_to_sample):
            chosen_graphs.append(org_name)
            org_name = org_name.split('.pkl')[0]
            out_org_name = f"{org_name}_{i}.pkl"
            if out_org_name in os.listdir(out_folder):
                print(f"{out_org_name} already exist - didnt save it {dim}")
                didnt_save_list.append(f"{org_name}.pkl")
            else:
                shutil.copyfile(f"{models_path}/{org_name}.pkl",
                                f"{out_folder}/{out_org_name}")
        data["chosen_orgs"] = chosen_graphs
        didnt_save[dim] = didnt_save_list
    with open(
            f'{base_all_archs_fold}/{exp_name}_good_archs_combined_gen_samples_names_1s.pkl',
            'wb+') as fp:
        joblib.dump(data_prep, fp)
