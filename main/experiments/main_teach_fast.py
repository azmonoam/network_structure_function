import argparse
from datetime import datetime as dt

from jobs_params import main_teach_fast
from logical_gates import LogicalGates
from neuroevolution import Neuroevolution
from nn_models.nn_model_five_layers import FeedforwardNeuralNetModelFiveLayers
from parameters.general_paramters import general_allowed_bias_values, general_allowed_weights_values
from parameters.xor_parameters import xors_6bit_network_dims
from tasks import XorsTask
from teach_population_fast import teach_fast

if __name__ == '__main__':
    rule = LogicalGates.XOR
    connection_cost_method = 'num_connections'
    potential_parents_percent = 15
    mutation_probabilities = {
        'connection_switch_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
    }
    start_dimensions = xors_6bit_network_dims

    optimization_method = 'performance_only'
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=8)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--num_epochs', default=10)
    parser.add_argument('--population_size', default=1000)
    parser.add_argument('--generations', default=300)
    parser.add_argument('--reinitiate', default=False)
    parser.add_argument('--connectivity_ratio', default=1.0)

    args = parser.parse_args()
    job_num = int(args.job_num)
    population_size = int(args.population_size)
    generations = int(args.generations)
    if job_num and main_teach_fast.get(job_num) is not None:
            learning_rate, connectivity_ratio, num_epochs, reinitiate = main_teach_fast[job_num]
    else:
        learning_rate = float(args.learning_rate)
        connectivity_ratio = float(args.connectivity_ratio)
        num_epochs = int(args.num_epochs)
        reinitiate = bool(int(args.reinitiate))
    experiment_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M")}_lr_{learning_rate}_ep_{num_epochs}_pop_' \
                      f'{population_size}_reinitiate_{reinitiate}_connectivity_ratio_{connectivity_ratio}'
    base_path = '/home/labs/schneidmann/noamaz/modularity/'
    #base_path = '/Users/noamazmon/PycharmProjects/network_modularity/'
    num_layers = 5
    input_dim = 6
    out_path = f"{base_path}/xor_teach_fast_res"
    activate = 'relu'
    task = XorsTask(
        input_dim=input_dim,
        rule=rule,
    )
    model_cls = FeedforwardNeuralNetModelFiveLayers
    neuroevolution = Neuroevolution(
        input_dim=input_dim,
        rules=rule,
        num_layers=num_layers,
        allowed_weights_values=general_allowed_weights_values,
        allowed_bias_values=general_allowed_bias_values,
        population_size=population_size,
        generations=generations,
        potential_parents_percent=potential_parents_percent,
        mutation_probabilities=mutation_probabilities,
        connection_cost_method=connection_cost_method,
        optimization_method=optimization_method,
        task=task,
    )
    teach_fast(
        experiment_name=experiment_name,
        base_path=base_path,
        neuroevolution=neuroevolution,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        reinitiate=reinitiate,
        out_path=out_path,
        start_dimensions=start_dimensions,
        connectivity_ratio=connectivity_ratio,
        save_every=5,
        activate=activate,
        task=task,
        model_cls=model_cls,
    )
