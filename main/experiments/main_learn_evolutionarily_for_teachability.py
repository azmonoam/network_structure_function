import argparse
import random
from datetime import datetime as dt

from learn_task_evolutionarily import LearnTaskEvolutionarily
from logical_gates import LogicalGates
from neuroevolution import Neuroevolution
from nn_models.nn_model_five_layers import FeedforwardNeuralNetModelFourLayers
from parameters.general_paramters import general_allowed_bias_values, general_allowed_weights_values
from parameters.retina_parameters import retina_6bit_network_dims
from tasks import RetinaTask
from utils.set_up_population_utils import get_population_by_connectivity_ratio

# TODO: update results csv to fit new org

if __name__ == '__main__':
    start_dimensions = retina_6bit_network_dims
    rule = LogicalGates.AND
    input_dim = 6
    connection_cost_method = 'num_connections'
    task = RetinaTask(
        input_dim=input_dim,
        rule=rule,
    )
    activate = 'relu'
    model_cls = FeedforwardNeuralNetModelFourLayers
    population_size = 500
    generations = 200
    mutation_probabilities = {
        'connection_switch_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
    }
    num_layers = 4
    connectivity_ratio = round(random.uniform(.3, 1.), 2)
    optimization_method = 'performance_only'
    potential_parents_percent = 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=225)
    parser.add_argument('--num_exp_per_arch', default=300)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--num_epochs', default=6000)

    args = parser.parse_args()
    job_num = int(args.job_num)
    num_exp_per_arch = int(args.num_exp_per_arch)
    learning_rate = float(args.learning_rate)
    num_epochs = int(args.num_epochs)
    if 0 <= job_num < 2000:
        performance_save_threshold = 1.0
    elif 2000 <= job_num < 4000:
        performance_save_threshold = 0.9
    elif 4000 <= job_num < 6000:
        performance_save_threshold = 0.8
    elif 6000 <= job_num < 8000:
        performance_save_threshold = 0.7
    elif 8000 <= job_num < 10000:
        performance_save_threshold = 0.6
    elif 10000 <= job_num < 12000:
        performance_save_threshold = 0.6
    else:
        performance_save_threshold = None
    pkl_path = None
    experiment_name = f'{dt.now().strftime("%Y-%m-%d-%H-%M")}_{job_num}_tr_{performance_save_threshold}'

    base_path = '//'
    # base_path = '/home/labs/schneidmann/noamaz/modularity/'
    neuroevolution = Neuroevolution(
        task=task,
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
    )
    learn_task_evolutionarily = LearnTaskEvolutionarily(
        neuroevolution=neuroevolution,
        base_path=base_path,
        experiment_name=experiment_name,
        save_every=50,
    )
    population = get_population_by_connectivity_ratio(
        neuroevolution=neuroevolution,
        connectivity_ratio=connectivity_ratio,
        start_dimensions=start_dimensions,
    )
    learn_task_evolutionarily.learn_evolutionarily_for_teachability_treshold_with_teaching(
        population=population,
        num_exp_per_arch=num_exp_per_arch,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        performance_save_threshold=performance_save_threshold,
        teach=True,
        model_folder_name="xor_teach_archs_models",
        teaching_csv_folder_name="xor_teach_archs_results",
        activate=activate,
        model_cls=model_cls,
    )
