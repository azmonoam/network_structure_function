from typing import List, Tuple, Union, Dict, Optional

NodeType = Tuple[int, int]
EdgeType = Tuple[NodeType, NodeType, Optional[Dict[str, Union[int, Tuple]]]]


def get_layer_neuron_idx_mapping(
        start_dimensions: List[int],
) -> List[List[int]]:
    start_neuron_num = 0
    layer_neuron_idx_mapping = []
    for num_neurons in start_dimensions:
        layer_neuron_idx_mapping.append(
            [
                neuron_number
                for neuron_number in range(start_neuron_num, start_neuron_num + num_neurons)
            ]
        )
        start_neuron_num += num_neurons
    return layer_neuron_idx_mapping


def get_tuple_layer_neuron_idx_mapping(
        start_dimensions: List[int],
) -> List[List[NodeType]]:
    layer_neuron_idx_mapping = []
    for layer, num_neurons in enumerate(start_dimensions):
        layer_neuron_idx_mapping.append(
            [
                (layer, neuron_number)
                for neuron_number in range(num_neurons)
            ]
        )
    return layer_neuron_idx_mapping


def get_number_of_neurons_per_layer(
        layer_neuron_idx_mapping: List[List[Union[int, Tuple[int, int]]]],
) -> List[Tuple[List[int], List[int]]]:
    layer_neuron_indexes = []
    start_dimensions = 0
    for layer_idx in range(len(layer_neuron_idx_mapping[:-1])):
        count_in_neuron = sum(1 for _ in layer_neuron_idx_mapping[layer_idx])
        count_out_neuron = sum(1 for _ in layer_neuron_idx_mapping[layer_idx + 1])
        rows = [start_dimensions, start_dimensions + count_in_neuron]
        columns = [start_dimensions + count_in_neuron, start_dimensions + count_in_neuron + count_out_neuron]
        layer_neuron_indexes.append((rows, columns))
        start_dimensions += count_in_neuron
    return layer_neuron_indexes


def get_allowed_edges(
        layer_neuron_idx_mapping: List[List[Union[int, NodeType]]],
) -> List[List[Union[Tuple[int, int], EdgeType]]]:
    allowed_edges = []
    for i in range(len(layer_neuron_idx_mapping) - 1):
        layer_allowed_edges = []
        current_layer_neurons, next_layer_neurons = (layer_neuron_idx_mapping[i], layer_neuron_idx_mapping[i + 1])
        for current_layer_neuron in current_layer_neurons:
            for next_layer_neuron in next_layer_neurons:
                layer_allowed_edges.append((current_layer_neuron, next_layer_neuron))
        allowed_edges.append(layer_allowed_edges)
    return allowed_edges
