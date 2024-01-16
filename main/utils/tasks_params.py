import torch

from logical_gates import LogicalGates
from nn_models.nn_model_five_layers import FeedforwardNeuralNetModelFiveLayers
from nn_models.nn_model_four_layers import FeedforwardNeuralNetModelFourLayers
from nn_models.nn_model_three_layers import FeedforwardNeuralNetModelThreeLayers
from nn_models.nn_model_two_layers import FeedforwardNeuralNetModelTwoLayers
from parameters.digits_parameters import digits_network_dims
from parameters.general_paramters import (
    mnist_network_dims,
    general_allowed_bias_values,
    general_allowed_weights_values,
)
from parameters.retina_parameters import retina_6bit_network_dims
from parameters.xor_parameters import xors_6bit_network_dims
from tasks import XorsTask, RetinaTask, MnistTask, DigitsTask


class XorParameters:
    task_name = 'xors'
    connection_cost_method = 'num_connections'
    population_size = None
    generations = None
    use_bias = True
    mutation_probabilities = {
        'connection_switch_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
    }
    allowed_bias_values = general_allowed_bias_values
    allowed_weights_values = general_allowed_weights_values
    input_dim = 6
    learning_rate = 0.001
    rule = LogicalGates.XOR
    num_epochs = 6000
    num_layers = 5
    communities_input_symmetry = 3
    start_dimensions = xors_6bit_network_dims
    task = XorsTask(
        input_dim=input_dim,
        rule=rule,
    )
    flatten = False
    activate = 'relu'
    optimizer = torch.optim.Adam
    model_cls = FeedforwardNeuralNetModelFiveLayers
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    teach_arch_base_path = f"{base_path}/teach_archs/{task_name}"
    output_folder = f"{teach_arch_base_path}/xor_teach_archs_results"
    pkls_folder = f"{teach_arch_base_path}/xor_teach_archs_new_models"


class RetinaParameters:
    task_name = 'retina'
    connection_cost_method = 'num_connections'
    population_size = None
    generations = None
    use_bias = True
    mutation_probabilities = {
        'connection_switch_mutation_probability': 0.2,
        'bias_mutation_probability': 1 / 24,
        'weight_mutation_probability_factor': 2,
    }
    allowed_bias_values = general_allowed_bias_values
    allowed_weights_values = general_allowed_weights_values
    rule = LogicalGates.AND
    communities_input_symmetry = 2
    input_dim = 6
    task = RetinaTask(
        input_dim=input_dim,
        rule=rule,
    )
    flatten = False
    start_dimensions = retina_6bit_network_dims
    learning_rate = 0.1
    num_layers = 4
    num_epochs = 2000
    activate = 'tanh'
    optimizer = torch.optim.SGD
    model_cls = FeedforwardNeuralNetModelFourLayers
    allowed_bias_values = allowed_bias_values
    allowed_weights_values = allowed_weights_values
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    teach_arch_base_path = f"{base_path}/teach_archs/{task_name}"
    output_folder = f"{teach_arch_base_path}/retina_teach_archs_results"
    pkls_folder = f"{teach_arch_base_path}/retina_teach_archs_new_models"


class MnistParameters:
    task_name = 'mnist'
    input_dim = 784
    start_dimensions = mnist_network_dims
    learning_rate = 0.005
    num_layers = 2
    num_epochs = 25
    activate = 'relu'
    task = MnistTask(
        input_dim=input_dim,
        rule=None
    )
    allowed_bias_values = general_allowed_bias_values
    allowed_weights_values = general_allowed_weights_values
    flatten = True
    optimizer = torch.optim.Adam
    model_cls = FeedforwardNeuralNetModelTwoLayers
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    teach_arch_base_path = f"{base_path}/teach_archs/{task_name}"
    output_folder = f"{teach_arch_base_path}/mnist_teach_archs_results"
    pkls_folder = f"{teach_arch_base_path}/mnist_teach_archs_new_models"


class DigitsParameters:
    task_name = 'digits'
    input_dim = 64
    start_dimensions = digits_network_dims
    learning_rate = 0.005
    num_layers = 3
    num_epochs = 600
    activate = 'relu'
    task = DigitsTask(
        input_dim=input_dim,
        rule=None
    )
    communities_input_symmetry = None
    allowed_bias_values = general_allowed_bias_values
    allowed_weights_values = general_allowed_weights_values
    flatten = False
    optimizer = torch.optim.Adam
    model_cls = FeedforwardNeuralNetModelThreeLayers
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    teach_arch_base_path = f"{base_path}/teach_archs/{task_name}"
    output_folder = f"{teach_arch_base_path}/digits_teach_archs_results"
    pkls_folder = f"{teach_arch_base_path}/digits_teach_archs_new_models"


TaskParameters = (XorParameters, MnistParameters, RetinaParameters, DigitsParameters)
