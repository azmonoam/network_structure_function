import argparse
import re
from datetime import datetime
from itertools import chain
from pathlib import Path

import joblib
import networkx as nx
from tqdm import tqdm

from logical_gates import LogicalGates
from new_organism import Organism
from parameters.digits.digits_by_dim import DigitsByDim
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.retina_parameters import RETINA_EDGE_MAPPING
from parameters.selected_exp_names import selected_exp_names
from parameters.xor.xor_by_dim import XoraByDim
from stractural_features_models.calc_structural_features import CalcStructuralFeatures


def _find_possible_edges_from_list():
    possible_edges = []
    for e in RETINA_EDGE_MAPPING:
        t1, t2, t3, t4 = (int(num) for num in re.findall(r'\d+', e))
        possible_edges.append(((t1, t2), (t3, t4)))
    nodes = grt_nodes_from_allowed_connections(
        allowed_connections=possible_edges,
    )
    return possible_edges, nodes


def _find_possible_edges_from_task(task_params):
    organism = Organism(
        dimensions=task_params.start_dimensions,
        num_layers=task_params.num_layers,
        allowed_weights_values=task_params.allowed_weights_values,
        allowed_bias_values=task_params.allowed_bias_values,
        communities_input_symmetry=task_params.communities_input_symmetry,
    )
    organism.build_organism_by_connectivity(
        max_allowed_connections=task_params.max_possible_connections,
    )
    possible_edges = []
    for i in chain(*organism.layer_neuron_idx_mapping):
        for j in chain(*organism.layer_neuron_idx_mapping):
            possible_edges.append((i, j))
    return possible_edges, list(organism.network.nodes)


def grt_nodes_from_allowed_connections(
        allowed_connections,
):
    nodes = []
    for i, j in allowed_connections:
        if i not in nodes:
            nodes.append(i)
    return nodes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=1)
    parser.add_argument('--task_name', default='digits')
    parser.add_argument('--random_init', default='0')

    args = parser.parse_args()
    job_num = int(args.job_num)
    task_name = args.task_name
    random_init = bool(int(args.random_init))

    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    selected_params = selected_exp_names[task_name]['random']
    folder_path = f'/{task_name}/{selected_params.source_folder}'
    num_features = selected_params.num_selected_features
    num_layers = selected_params.num_layers
    folder_name = f'{num_features}_features'
    ergm_results_path = f'{base_path}/{folder_path}/ergm/{folder_name}/per_dim_results/'
    if random_init:
        data_pkl = selected_params.ergm_res_pkl_file_name_random_init
        models_folder = Path(f"{ergm_results_path}/teach_archs_models_random_init")
        results_folder = Path(f"{ergm_results_path}/teach_archs_results_random_init")
        models_folder.mkdir(exist_ok=True)
        results_folder.mkdir(exist_ok=True)
    else:
        data_pkl = selected_params.ergm_res_pkl_file_name
        models_folder = Path(f"{ergm_results_path}/teach_archs_models")
        results_folder = Path(f"{ergm_results_path}/teach_archs_results")
        models_folder.mkdir(exist_ok=True)
        results_folder.mkdir(exist_ok=True)
    prepered_data_path = f'{ergm_results_path}/{data_pkl}'
    with open(prepered_data_path, 'rb') as fp:
        prepered_data = joblib.load(fp)
    all_dims = [d for d in prepered_data.keys() if isinstance(d, tuple)]
    if task_name == 'retina_xor':
        dims = all_dims[job_num - 1]
        num_layers = len(dims) - 1
        task_params = RetinaByDim(
            start_dimensions=list(dims),
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
    elif task_name == 'xor':
        dims = all_dims[job_num - 1]
        num_layers = len(dims) - 1
        task_params = XoraByDim(
            start_dimensions=list(dims),
            num_layers=num_layers,
            by_epochs=False,
        )
    elif task_name == 'digits':
        dims = all_dims[job_num - 1]
        num_layers = len(dims) - 1
        task_params = DigitsByDim(
            start_dimensions=list(dims),
            num_layers=num_layers,
            by_epochs=False,
        )
    else:
        raise ValueError

    possible_edges, nodes = _find_possible_edges_from_task(task_params)
    res = prepered_data[dims]

    used_gs = []
    used_g_stats = []
    missing_g = 0
    for p, g in tqdm(enumerate(res['chosen_graphs'])):
        edges = []
        k = 0
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if g[i, j] == 1:
                    edges.append(possible_edges[k])
                k += 1
        net = nx.DiGraph()
        net.add_nodes_from(nodes)
        net.add_edges_from(edges)
        if len(list(nx.isolates(net))) > 0:
            print(f'graph {p}: invalid graph, unconnected nodes')
            missing_g += 0
            continue
        used_gs.append(g)
        used_g_stats.append(res['chosen_graphs_stats'][p])
        org = Organism(
            dimensions=task_params.start_dimensions,
            num_layers=task_params.num_layers,
            allowed_weights_values=task_params.allowed_weights_values,
            network=net,
            communities_input_symmetry=task_params.communities_input_symmetry,
        )
        org.network = net
        structural_features_calculator = CalcStructuralFeatures(
            organism=org,
        )
        try:
            org = structural_features_calculator.calc_structural_features()
            with open(f"{models_folder}/{time_str}_{job_num}_{p}.pkl",
                      'wb+') as fp:
                joblib.dump(org, fp)
        except Exception as e:
            print(f'graph {p}: there was a problem calculating the structural features, probably too sparse'
                  f' ({sum(sum(g))} connections).\n{e}')
            missing_g += 1
    print(f"saved {len(res['chosen_graphs']) - missing_g}/{len(res['chosen_graphs'])} graphs")
print('a')
