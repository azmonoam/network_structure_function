import argparse
import os
import random
from _datetime import datetime as dt
from pathlib import Path

import joblib
import pandas as pd

from ergm_model.data_halper import get_list_of_functions_from_features_names
from ergm_model.nx_methods import NxMethods
from parameters.general_paramters import RANDOM_SEED
from parameters.selected_exp_names import selected_exp_names

if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    #base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='xor')
    parser.add_argument('--type', default='4')

    args = parser.parse_args()
    task_name = args.task_name
    type = int(args.type)
    types_mapping = {
        1: 'full_db',
        2: 'ergm',
        3: 'ergm_random_init',
        4: 'genet',
        5: 'genet_1s',
    }

    exp_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    num_layers = selected_params.num_layers
    folder_name = f'{num_features}_features'
    db_data_pkl = selected_params.ergm_init_graphs_pkl_file_name
    ergm_data_pkl = selected_params.ergm_res_pkl_file_name
    ergm_random_init_data_pkl = selected_params.ergm_res_pkl_file_name_random_init
    gen_data_pkl = selected_params.gen_graphs_pkl_file_name
    base_ergm_fold = f'{base_path}/{folder_path}/ergm/{folder_name}'
    db_prepered_data_path = f'{base_ergm_fold}/{db_data_pkl}'

    record_folder = Path(f'{base_path}/{folder_path}/generated_archs')
    record_folder.mkdir(exist_ok=True)

    with open(db_prepered_data_path, 'rb') as fp:
        db_data = joblib.load(fp)
    features_names = db_data['selected_feature_names']
    all_gen_orgs_stats = []
    if type == 1:
        max_possible_connections_ind = None
        if "max_possible_connections" in features_names:
            max_possible_connections_ind = features_names.index("max_possible_connections")
        dim1_ind = None
        if "dimensions_1" in features_names:
            dim1_ind = features_names.index("dimensions_1")
        for dim, v in db_data['data_per_dim'].items():
            max_possible_conn = sum(
                dim[i] * dim[i + 1]
                for i in range(len(dim) - 1)
            )
            dim_1 = dim[1]
            for stat in v['all_features_values']:
                if max_possible_connections_ind is not None:
                    stat.insert(max_possible_connections_ind, max_possible_conn)
                if dim1_ind is not None:
                    stat.insert(dim1_ind, dim_1)
                all_gen_orgs_stats.append(stat)

    elif type == 2:
        prepered_data_path = f'{base_ergm_fold}/per_dim_results/{ergm_data_pkl}'
        with open(prepered_data_path, 'rb') as fp:
            ergm_data = joblib.load(fp)
        all_gen_orgs_stats = ergm_data['all_sampled_graphs_stats']

    elif type == 3:
        prepered_data_random_init_path = f'{base_ergm_fold}/per_dim_results/{ergm_random_init_data_pkl}'
        with open(prepered_data_random_init_path, 'rb') as fp:
            ergm_random_init_data = joblib.load(fp)
        all_gen_orgs_stats = ergm_random_init_data['all_sampled_graphs_stats']

    elif type in (4, 5):
        if type==4:
            models_fold = 'teach_archs_models'
        else:
            models_fold = 'teach_archs_models_1s'
        base_gen_fold = f'{base_path}/{folder_path}/requiered_features_genetic_models/{folder_name}/good_archs/per_dim_results'
        prepered_data_gen_path = f'{base_gen_fold}/{gen_data_pkl}'
        with open(prepered_data_gen_path, 'rb') as fp:
            gen_data = joblib.load(fp)

        methods = NxMethods(
            connectivity_in_perc=True
        )
        features = get_list_of_functions_from_features_names(
            method_class=methods,
            features_names_list=features_names,
        )
        all_gen_orgs_stats = []
        gen_orgs = [p for p in os.listdir(f'{base_gen_fold}/{models_fold}/') if '._' not in p]
        for org_file_name in gen_orgs:
            with open(f'{base_gen_fold}/{models_fold}/{org_file_name}', 'rb') as fp:
                org = joblib.load(fp)
            features_values = [
                f(org, f_name)
                for f, f_name in zip(features, features_names)
            ]
            all_gen_orgs_stats.append(features_values)
    else:
        raise ValueError()

    all_gen_orgs_stats_df = pd.DataFrame(all_gen_orgs_stats, columns=features_names)
    all_gen_orgs_stats_df.to_csv(f"{record_folder}/{exp_name}_feature_values_{types_mapping[type]}.csv", index=False)

    print('a')
