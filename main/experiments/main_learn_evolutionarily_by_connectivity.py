import argparse
import random
from datetime import datetime as dt

from learn_task_evolutionarily import LearnTaskEvolutionarily
from logical_gates import LogicalGates
from neuroevolution import Neuroevolution
from parameters.general_paramters import general_allowed_bias_values, general_allowed_weights_values
from parameters.xor_parameters import xors_6bit_network_dims
from tasks import XorsTask
from utils.set_up_population_utils import get_population_by_connectivity_ratio

if __name__ == '__main__':
    rule = LogicalGates.AND
    connection_cost_method = 'num_connections'
    population_size = 1000
    generations = 500
    potential_parents_percent = 15
    mutation_probabilities = {
        'connection_switch_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
    }
    optimization_method = 'performance_only'
    optimize_by_connection_in_prob = 0.25
    num_layers = 4
    input_dim = 6
    start_dimensions = xors_6bit_network_dims
    task = XorsTask(
        input_dim=input_dim,
        rule=rule,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default='1')
    parser.add_argument('--num_exp_per_arch', default=300)
    parser.add_argument('--learning_rate', default=0.1)
    parser.add_argument('--num_epochs', default=2000)
    parser.add_argument('--connectivity_ratio', default=None)

    args = parser.parse_args()
    job_num = int(args.job_num)
    num_exp_per_arch = int(args.num_exp_per_arch)
    learning_rate = float(args.learning_rate)
    num_epochs = int(args.num_epochs)
    if args.connectivity_ratio is not None:
        connectivity_ratio = float(args.connectivity_ratio)
    else:
        if 0 <= job_num < 1000:
            connectivity_ratio = 0.3
        elif 1000 <= job_num < 2000:
            connectivity_ratio = 0.4
        elif 2000 <= job_num < 3000:
            connectivity_ratio = 0.5
        elif 3000 <= job_num < 4000:
            connectivity_ratio = 0.6
        elif 4000 <= job_num < 5000:
            connectivity_ratio = 0.7
        elif 5000 <= job_num < 6000:
            connectivity_ratio = 0.8
        elif 6000 <= job_num < 7000:
            connectivity_ratio = 0.9
        elif 7000 <= job_num < 8000:
            connectivity_ratio = 1.0
        else:
            connectivity_ratio = random.randint(3, 10) / 10
    experiment_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M")}_{job_num}_connectivity_ratio_{connectivity_ratio}'

    base_path = '/'
    # base_path = '/home/labs/schneidmann/noamaz/modularity'
    neuroevolution = Neuroevolution(
        rules=rule,
        input_dim=input_dim,
        num_layers=num_layers,
        allowed_weights_values=general_allowed_weights_values,
        allowed_bias_values=general_allowed_bias_values,
        population_size=population_size,
        generations=generations,
        potential_parents_percent=potential_parents_percent,
        mutation_probabilities=mutation_probabilities,
        connection_cost_method=connection_cost_method,
        optimization_method=optimization_method,
        optimize_by_connection_in_prob=optimize_by_connection_in_prob,
        task=task
    )
    learn_task_evolutionarily = LearnTaskEvolutionarily(
        neuroevolution=neuroevolution,
        base_path=base_path,
        experiment_name=experiment_name,
        save_every=50,
    )
    population = get_population_by_connectivity_ratio(
        start_dimensions=start_dimensions,
        connectivity_ratio=connectivity_ratio,
        neuroevolution=neuroevolution
    )
    learn_task_evolutionarily.learn_evolutionarily_for_teachability(
        population=population,
    )
