import argparse
import copy
import random
from _datetime import datetime as dt
from typing import List, Tuple

import joblib
import numpy as np
from joblib import Parallel, delayed

from parameters.general_paramters import RANDOM_SEED
from parameters.selected_exp_names import selected_exp_names
from ergm_model.ergm_from_reference_graph_multi_g import ErgmReferenceGraph


def initialise_graph(
        ergm_model: ErgmReferenceGraph,
):
    num_connections = random.randint(1, len(ergm_model.allowed_connections))
    g = np.zeros((ergm_model.n_nodes, ergm_model.n_nodes))
    for i, j in random.choices(ergm_model.allowed_connections, k=num_connections):
        g[i, j] = 1
    return g


def sample_new_graph(
        ergm_model: ErgmReferenceGraph,
        coefs: np.ndarray,
        seperation: int = 5,
) -> Tuple[np.ndarray, List[float]]:
    current_graph = initialise_graph(
        ergm_model
    )
    current_stats = ergm_model.calc_stats(
        graph=current_graph,
    )
    sum_delta_stats = 0
    for itter in range(seperation):
        for i, j in ergm_model.allowed_connections:
            current_graph, current_stats, _ = ergm_model._update_ergm(
                coefs=coefs,
                current_graph=current_graph,
                i=i,
                j=j,
                current_stats=current_stats,
                sum_delta_stats=sum_delta_stats,
            )

    return current_graph, current_stats


def sample_multiple_parallel_wrapper(
        ergm_model: ErgmReferenceGraph,
        coefs: np.ndarray,
        num_graphs_to_sample: int = 200,
        n_threads: int = 4,
        seperation: int = 5,
):
    return Parallel(
        n_jobs=n_threads,
        timeout=99999,
    )(
        delayed
        (sample_new_graph)(
            seperation=seperation,
            coefs=coefs,
            ergm_model=ergm_model,
        )
        for _ in range(num_graphs_to_sample)
    )


def sample_multiple(
        ergm_model: ErgmReferenceGraph,
        coefs: np.ndarray,
        num_graphs_to_sample: int = 200,
        n_threads: int = 4,
        seperation: int = 5,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    res = sample_multiple_parallel_wrapper(
        ergm_model=ergm_model,
        coefs=coefs,
        num_graphs_to_sample=num_graphs_to_sample,
        n_threads=n_threads,
        seperation=seperation,
    )
    g_stats = []
    graphs = []
    for current_graph, current_stats in res:
        g_stats.append(current_stats)
        graphs.append(current_graph)
    return graphs, g_stats


if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='digits')
    args = parser.parse_args()
    task_name = args.task_name

    if task_name == 'xor':
        pkls_name = {
            (6, 5, 5, 4, 2): "2023-11-25-10-12-01_J11_good_archs_dim_6_5_5_4_2_b.pkl",
            (6, 6, 4, 4, 2): "2023-11-25-11-04-12_J11_good_archs_dim_6_6_4_4_2.pkl",
            (6, 5, 4, 5, 2): "2023-11-25-11-15-45_J9_good_archs_dim_6_5_4_5_2.pkl",
            (6, 4, 6, 4, 2): "2023-11-25-11-18-01_J9_good_archs_dim_6_4_6_4_2.pkl",
            (6, 6, 5, 3, 2): "2023-11-25-11-26-41_J9_good_archs_dim_6_6_5_3_2.pkl",
            (6, 4, 5, 5, 2): "2023-11-25-15-05-36_J12_good_archs_dim_6_4_5_5_2.pkl",
            (6, 5, 6, 3, 2): "2023-11-25-11-44-46_J9_good_archs_dim_6_5_6_3_2.pkl",
            (6, 6, 3, 5, 2): "2023-11-25-11-50-32_J11_good_archs_dim_6_6_3_5_2.pkl",
            (6, 4, 4, 6, 2): "2023-11-25-15-07-06_J10_good_archs_dim_6_4_4_6_2.pkl",
            (6, 5, 3, 6, 2): "2023-11-25-17-41-39_J12_good_archs_dim_6_5_3_6_2.pkl",
            (6, 3, 6, 5, 2): "2023-11-25-15-02-08_J9_good_archs_dim_6_3_6_5_2.pkl",
            (6, 3, 5, 6, 2): "2023-11-25-12-51-33_J11_good_archs_dim_6_3_5_6_2.pkl",
            (6, 6, 6, 2, 2): None,
            (6, 6, 2, 6, 2): None,
            (6, 2, 6, 6, 2): None,
        }
        total_num_graphs_to_sample = 5000
    elif task_name == 'retina_xor':
        pkls_name = {
            (6, 5, 2, 2): "2023-11-27-16-39-46_J12_good_archs_dim_6_5_2_2.pkl",
            (6, 2, 5, 2): "2023-11-27-16-43-07_J10_good_archs_dim_6_2_5_2.pkl",
            (6, 3, 4, 2): "2023-11-27-19-30-02_J3_good_archs_dim_6_3_4_2.pkl",
            (6, 4, 3, 2): "2023-11-27-19-46-05_J3_good_archs_dim_6_4_3_2.pkl",
        }
        total_num_graphs_to_sample = 2500
    elif task_name == 'digits':
        pkls_name = {
            (64, 8, 4, 10): "2023-11-28-16-21-42_J12_good_archs_dim_64_8_4_10.pkl",
            (64, 5, 7, 10): "2023-11-28-16-22-55_J11_good_archs_dim_64_5_7_10.pkl",
            (64, 6, 6, 10): "2023-11-28-16-06-39_J12_good_archs_dim_64_6_6_10.pkl",
            (64, 7, 5, 10): "2023-11-28-16-35-24_J12_good_archs_dim_64_7_5_10.pkl",
        }
        total_num_graphs_to_sample = 5000
    else:
        raise ValueError()

    exp_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    num_layers = selected_params.num_layers
    folder_name = f'{num_features}_features'
    data_pkl = selected_params.ergm_init_graphs_pkl_file_name
    base_ergm_fold = f'{base_path}/{folder_path}/ergm/{folder_name}'
    prepered_data_path = f'{base_ergm_fold}/{data_pkl}'
    with open(prepered_data_path, 'rb') as fp:
        prepered_data = joblib.load(fp)
    base_all_archs_fold = f'{base_ergm_fold}/per_dim_results'
    max_possible_connections_ind = None
    if "max_possible_connections" in prepered_data["selected_feature_names"]:
        max_possible_connections_ind = prepered_data["selected_feature_names"].index("max_possible_connections")
    dim1_ind = None
    if "dimensions_1" in prepered_data["selected_feature_names"]:
        dim1_ind = prepered_data["selected_feature_names"].index("dimensions_1")
    features_to_multipl = [
        'total_connectivity_ratio_between_layers_0',
        'total_connectivity_ratio_between_layers_1',
        'total_connectivity_ratio_between_layers_2',
        'total_connectivity_ratio_between_layers_3',
        'max_connectivity_between_layers_per_layer_0',
        'max_connectivity_between_layers_per_layer_1',
        'max_connectivity_between_layers_per_layer_2',
        'max_connectivity_between_layers_per_layer_3',
        'connectivity_ratio'
    ]
    dims_to_mul = []
    for f in features_to_multipl:
        if f in prepered_data["selected_feature_names"]:
            dims_to_mul.append(prepered_data["selected_feature_names"].index(f))
    data_prep = {}
    sum_num_graphs = 0
    all_probs = []
    for d, v in prepered_data["data_per_dim"].items():
        if pkls_name.get(d) is None:
            continue
        sum_num_graphs += v['num_graphs']
    for d, v in prepered_data["data_per_dim"].items():
        prob = v['num_graphs'] / sum_num_graphs
        dim_st = ''
        for l in d:
            dim_st += f"_{l}"
        dim_record_folder = f'{base_all_archs_fold}/{dim_st[1:]}'
        if pkls_name.get(d) is None:
            continue
        with open(
                f'{dim_record_folder}/{pkls_name[d]}', 'rb+') as fp:
            res = joblib.load(fp)
        ergm = res["model"]
        data_prep[d] = {
            "sample_prob": prob,
            "num_graphs_to_sample": 0,
            "ergm": ergm,
            "graphs": res["graphs"],
            "graphs_stats": res["graphs_stats"],
            "coefs": res["mean_recorded_coeff"],
        }
        all_probs.append(prob)
    for dim in random.choices(list(data_prep.keys()), weights=all_probs, cum_weights=None,
                              k=total_num_graphs_to_sample):
        data_prep[dim]["num_graphs_to_sample"] += 1
    all_graphs = []
    all_graphs_stats = []
    for dim, data in data_prep.items():
        max_possible_conn = sum(
            dim[i] * dim[i + 1]
            for i in range(len(dim) - 1)
        )
        dim_1 = dim[1]
        ergm = data["ergm"]
        num_graphs_to_sample = data["num_graphs_to_sample"]
        dim_graphs, dim_graphs_stats = sample_multiple(
            ergm_model=ergm,
            coefs=data["coefs"],
            num_graphs_to_sample=int(num_graphs_to_sample * 1.5),
        )
        sample_means = np.mean(dim_graphs_stats, axis=0)
        print(f"\n{dim} sampled mean: {sample_means}")
        chosen_graphs = []
        chosen_graphs_stats = []
        graphs_inds = random.sample(range(len(dim_graphs)), k=num_graphs_to_sample)
        used_graphs_inds = copy.deepcopy(graphs_inds)
        for j, i in enumerate(graphs_inds):
            while len(set(np.where(np.sum(dim_graphs[i], axis=0) == 0)[0].tolist()) & set(
                    np.where(np.sum(dim_graphs[i], axis=1) == 0)[0].tolist())) > 0:
                if len(set(range(len(dim_graphs))) - set(used_graphs_inds)) > 0:
                    i = random.sample(set(range(len(dim_graphs))) - set(used_graphs_inds), k=1)[0]
                else:
                    g, s = sample_new_graph(
                        ergm_model=ergm,
                        coefs=data["coefs"],
                    )
                    dim_graphs.append(g)
                    dim_graphs_stats.append(s)
                    i = len(dim_graphs)-1
                used_graphs_inds.append(i)
                graphs_inds[j] = i
            chosen_graphs.append(dim_graphs[i])
            stat = dim_graphs_stats[i]
            if max_possible_connections_ind is not None:
                stat.insert(max_possible_connections_ind, max_possible_conn)
            if dim1_ind is not None:
                stat.insert(dim1_ind, dim_1)
            chosen_graphs_stats.append(stat)
        data["chosen_graphs"] = chosen_graphs
        data["chosen_graphs_stats"] = chosen_graphs_stats
        all_graphs += chosen_graphs
        all_graphs_stats += chosen_graphs_stats
        sample_means = np.mean(chosen_graphs_stats, axis=0)
        sample_std = np.std(chosen_graphs_stats, axis=0)
        print(f"\n{dim}: {num_graphs_to_sample} samples")
        print(f"sample mean {sample_means}")
        print(f"sample std {sample_std}")
        original_mean = prepered_data["data_per_dim"][dim]["target_feature_values"].tolist()
        original_std = np.std(prepered_data["data_per_dim"][dim]["all_features_values"], axis=0).tolist()
        errors = prepered_data["data_per_dim"][dim]["errors"].tolist()
        if max_possible_connections_ind is not None:
            original_mean.insert(max_possible_connections_ind, max_possible_conn)
            original_std.insert(max_possible_connections_ind, 0)
            errors.insert(max_possible_connections_ind, 0.00001)
        if dim1_ind is not None:
            original_mean.insert(dim1_ind, dim_1)
            original_std.insert(dim1_ind, 0)
            errors.insert(dim1_ind, 0.00001)
        print(f'original mean: {original_mean}')
        print(f'original std: {original_std}')
        sample_error = [
            abs(original_mean[i] - sample_means[i]) / errors[i]
            for i in range(len(sample_means))
        ]
        print(f"errors: {sample_error}")
    total_mean = np.mean(all_graphs_stats, axis=0)
    print(f"\ntotal mean {total_mean}")
    total_std = np.std(all_graphs_stats, axis=0)
    print(f"total std {total_std}")
    original_mean = prepered_data["target_feature_values"]
    errors = prepered_data["errors"]
    for ind in dims_to_mul:
        original_mean[ind] = original_mean[ind] * 100
        errors[ind] = errors[ind] * 100
    print(f'original mean: {original_mean}')
    total_error = [
        abs(original_mean[i] - total_mean[i]) / errors[i]
        for i in range(len(total_mean))
    ]
    print(f"errors: {total_error}")
    data_prep["all_sampled_graphs"] = all_graphs
    data_prep["all_sampled_graphs_stats"] = all_graphs_stats
    data_prep["all_sampled_graphs_mean_stats"] = total_mean
    data_prep["target_mean"] = original_mean
    data_prep["target_error"] = errors
    data_prep["all_sampled_graphs_error"] = total_error
    with open(
            f'{base_all_archs_fold}/{exp_name}_good_archs_combined_ergm_samples_random_init.pkl',
            'wb+') as fp:
        joblib.dump(data_prep, fp)
    print(f'{base_all_archs_fold}/{exp_name}_good_archs_combined_ergm_samples_random_init.pkl')
