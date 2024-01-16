import argparse

import joblib

from learn_task_evolutionarily import LearnTaskEvolutionarily
from logical_gates import LogicalGates
from neuroevolution import Neuroevolution
from parameters.general_paramters import general_allowed_bias_values, general_allowed_weights_values
from tasks import RetinaTask
from utils.set_up_population_utils import get_population_fixed_size_clone_possible

# ToDo add locations for nodes and length cost


if __name__ == '__main__':
    rule = LogicalGates.AND
    connection_cost_method = 'num_connections'
    start_dimensions = [8, 8, 4, 2, 2]
    population_size = 1000
    generations = 15000
    max_connections_per_neuron_in_layer_dict = {
        '1': [4, 4, 2, 2],
        '2': [5, 2, 2, 2],
        '3': [5, 3, 2, 2],
        '4': [5, 2, 2, 2],
        '5': [6, 2, 2, 2],
        '6': [6, 3, 2, 2],
        '7': [6, 4, 2, 2],
        '8': [3, 3, 2, 2],
        '9': [3, 2, 2, 2],
        '10': [3, 4, 2, 2],
        '11': [4, 2, 2, 2],
        '12': [4, 3, 2, 2],
        '13': [7, 2, 2, 2],
        '14': [7, 3, 2, 2],
        '15': [7, 4, 2, 2],

    }
    potential_parents_percent = 15
    mutation_probabilities = {
        'connection_addition_mutation_probability': 0.2,
        'connection_removal_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
        'neuron_addition_mutation_probability': None,
        'neuron_removal_mutation_probability': None,
    }
    clones_only = True
    # optimization_method = 'performance_only'
    optimization_method = 'performance_and_connection_cost'
    optimize_by_connection_in_prob = 0.25
    num_layers = 4
    input_dim = 8
    task = RetinaTask(
        input_dim=input_dim,
        rule=rule,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num')
    parser.add_argument('--ind', default='11')
    args = parser.parse_args()
    job_num = args.job_num
    ind = args.ind
    if not ind:
        max_connections_per_neuron_in_layer = [6, 2, 2, 2]
        pkl_path = None
        experiment_name = None

    else:
        max_connections_per_neuron_in_layer = max_connections_per_neuron_in_layer_dict[ind]
        pkl_path = f"populations/population{ind}.pkl"
        experiment_base_name = f"26_12_22_pop{ind}"
        experiment_name = f'{experiment_base_name}_{job_num}'

    neuroevolution = Neuroevolution(
        rules=rule,
        input_dim=input_dim,
        allowed_weights_values=general_allowed_weights_values,
        allowed_bias_values=general_allowed_bias_values,
        population_size=population_size,
        generations=generations,
        potential_parents_percent=potential_parents_percent,
        mutation_probabilities=mutation_probabilities,
        connection_cost_method=connection_cost_method,
        optimization_method=optimization_method,
        optimize_by_connection_in_prob=optimize_by_connection_in_prob,
        num_layers=num_layers,
        task=task,
    )
    base_path = '/home/labs/schneidmann/noamaz/modularity/'
    learn_task_evolutionarily = LearnTaskEvolutionarily(
        neuroevolution=neuroevolution,
        base_path=base_path,
        experiment_name=experiment_name,
        save_every=500,
    )
    if pkl_path:
        with open(pkl_path, 'rb') as fp:
            population = joblib.load(fp)
    else:
        population = get_population_fixed_size_clone_possible(
            neuroevolution=neuroevolution,
            start_dimensions=start_dimensions,
            max_connections_per_neuron_in_layer=max_connections_per_neuron_in_layer,
            clones_only=clones_only,
        )
    learn_task_evolutionarily.learn_evolutionarily(
        population=population,

    )
