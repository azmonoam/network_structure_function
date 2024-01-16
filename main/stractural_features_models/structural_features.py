from typing import List, Optional, Dict, Union, Tuple, FrozenSet
from utils.set_up_organism_utils import NodeType

import numpy as np


def flatten_lists(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten_lists(i)]
    else:
        return [x]


countable_params = [
    "total_connectivity_ratio_between_layers",
    "layer_connectivity_rank",
    "max_connectivity_between_layers_per_layer",
    "distances_between_input_neuron",
]
per_layer_per_neuron_params = [
    'out_connections_per_layer',
    'in_connections_per_layer',
]
per_neuron_params = [
    "num_paths_to_output_per_input_neuron",
    "num_involved_neurons_in_paths_per_input_neuron",
]

GLOBAL_FEATURES = [
    'modularity',
    'entropy',
    'normed_entropy',
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'motifs_count',
    'dimensions',
    'num_layers',
    'num_neurons'
]
CONSTANT_FEATURES = [
    'entropy',
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'motifs_count',
    'dimensions',
    'num_layers',
    'num_neurons',
    "num_paths_to_output_per_input_neuron",
    "num_involved_neurons_in_paths_per_input_neuron",
]

CONSTANT_FEATURES_PER_ARCH = [
    'entropy',
    'connectivity_ratio',
    'num_connections',
    'motifs_count',
    "num_paths_to_output_per_input_neuron",
    "num_involved_neurons_in_paths_per_input_neuron",
]

NO_DIST_FEATURES = [
    'entropy',
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'motifs_count',
    'dimensions',
    'num_layers',
    'num_neurons'
    "num_paths_to_output_per_input_neuron",
    "num_involved_neurons_in_paths_per_input_neuron",
    "total_connectivity_ratio_between_layers",
    "layer_connectivity_rank",
    "max_connectivity_between_layers_per_layer",
    "out_connections_per_layer",
    "in_connections_per_layer"
]

NO_MOD_FEATURES = [
    'entropy',
    'normed_entropy',
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'out_connections_per_layer',
    'in_connections_per_layer ',
    'total_connectivity_ratio_between_layers',
    'max_connectivity_between_layers_per_layer',
    'layer_connectivity_rank',
    'distances_between_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
    'motifs_count',
    'dimensions',
    'num_layers',
    'num_neurons'
]

NICE_FEATURES = [
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'out_connections_per_layer',
    'in_connections_per_layer',
    'total_connectivity_ratio_between_layers',
    'max_connectivity_between_layers_per_layer',
    'layer_connectivity_rank',
    'distances_between_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
    'motifs_count',
    'dimensions',
    'num_layers',
    'num_neurons'
]
NICE_FEATURES_NO_INV = [
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'out_connections_per_layer',
    'in_connections_per_layer',
    'total_connectivity_ratio_between_layers',
    'max_connectivity_between_layers_per_layer',
    'layer_connectivity_rank',
    'distances_between_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'motifs_count',
    'dimensions',
    'num_layers',
    'num_neurons'
]
NICE_FEATURES_NO_DIST = [
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'out_connections_per_layer',
    'in_connections_per_layer ',
    'total_connectivity_ratio_between_layers',
    'max_connectivity_between_layers_per_layer',
    'layer_connectivity_rank',
    'distances_between_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
    'motifs_count',
    'dimensions',
    'num_layers',
    'num_neurons'
]

ONLY_TOP = [
    'connectivity_ratio',
    'motifs_count_1',
    'motifs_count_2',

]
class Feature:
    def get_class_values(
            self,
            features_list: Optional[List[str]] = None,
    ):
        all_vals = []
        for key, val in list(self.__dict__.items()):
            if features_list is not None and key not in features_list:
                continue
            if isinstance(val, list):
                all_vals += flatten_lists(val)
            elif isinstance(val, np.ndarray):
                all_vals += val.tolist()
            else:
                all_vals.append(val)
        return all_vals

    def get_features(
            self,
            layer_neuron_idx_mapping: List[List[Tuple[int, int]]],
            features_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        features_dict = {}
        for key, val in self.__dict__.items():
            if features_list is not None and key not in features_list:
                continue
            if isinstance(val, list):
                for ind in range(len(val)):
                    features_dict[f'{key}_{ind}'] = val[ind]
            else:
                features_dict[key] = val
        return features_dict


class ModularityFeatures(Feature):
    def __init__(
            self,
            modularity: float,
            communities: List[FrozenSet[NodeType]],
    ):
        self.modularity: float = modularity
        self.communities: List[FrozenSet[NodeType]] = communities

    def get_class_values(
            self,
            features_list: Optional[List[str]] = None,
    ) -> List[Union[Tuple, float]]:
        if features_list is None:
            features_list = ['modularity']
        all_vals = []
        for key, val in list(self.__dict__.items()):
            if features_list is not None and key not in features_list:
                continue
            if isinstance(val, list):
                all_vals += flatten_lists(val)
            elif isinstance(val, np.ndarray):
                all_vals += val.tolist()
            else:
                all_vals.append(val)
        return all_vals

    def get_features(
            self,
            layer_neuron_idx_mapping: List[List[Tuple[int, int]]],
            features_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        if features_list is None:
            features_list = ['modularity']
        features_dict = {}
        for key, val in self.__dict__.items():
            if features_list is not None and key not in features_list:
                continue
            features_dict[key] = val
        return features_dict


class EntropyFeatures(Feature):
    def __init__(
            self,
            entropy: float,
            normed_entropy: float
    ):
        self.entropy: float = entropy
        self.normed_entropy: float = normed_entropy


class ConnectivityFeatures(Feature):
    def __init__(
            self,
            connectivity_ratio: float,
            num_connections: int,
            max_possible_connections: int,
            connectivity_data_dict_per_layer: Dict[str, List[Union[float, int]]],
            distances_between_input_neuron: List[float],
            num_paths_to_output_per_input_neuron: List[int],
            num_involved_neurons_in_paths_per_input_neuron: List[int],
    ):
        self.connectivity_ratio: float = connectivity_ratio
        self.num_connections: int = num_connections
        self.max_possible_connections: int = max_possible_connections
        self.out_connections_per_layer: Optional[List[List[int]]] = connectivity_data_dict_per_layer.get(
            'out_connections_per_layer')
        self.in_connections_per_layer: Optional[List[List[int]]] = connectivity_data_dict_per_layer.get(
            'in_connections_per_layer')
        self.total_connectivity_ratio_between_layers: Optional[List[float]] = connectivity_data_dict_per_layer.get(
            'total_connectivity_ratio_between_layers')
        self.max_connectivity_between_layers_per_layer: Optional[List[float]] = connectivity_data_dict_per_layer.get(
            'max_connectivity_between_layers_per_layer')
        self.layer_connectivity_rank: Optional[np.ndarray] = connectivity_data_dict_per_layer.get(
            'layer_connectivity_rank')
        self.distances_between_input_neuron: List[float] = distances_between_input_neuron
        self.num_paths_to_output_per_input_neuron: List[int] = num_paths_to_output_per_input_neuron
        self.num_involved_neurons_in_paths_per_input_neuron: List[int] = num_involved_neurons_in_paths_per_input_neuron

    def get_features(
            self,
            layer_neuron_idx_mapping: List[List[Tuple[int, int]]],
            features_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        features_dict = {}
        for key, val in self.__dict__.items():
            if features_list is not None and key not in features_list:
                continue
            if isinstance(val, (int, float)):
                features_dict[key] = val
            elif key in per_layer_per_neuron_params:
                for ind in range(len(val)):
                    features_dict = {
                        **features_dict,
                        **{
                            f"{key}_{neuron} ": val[ind][i]
                            for i, neuron in enumerate(layer_neuron_idx_mapping[ind])
                        }
                    }
            elif key in countable_params:
                features_dict = {
                    **features_dict,
                    **{
                        f"{key}_{i}": val[i]
                        for i in range(len(val))
                    }
                }
            elif key in per_neuron_params:
                features_dict = {
                    **features_dict,
                    **{
                        f"{key}_{neuron}": val[ind]
                        for ind, neuron in enumerate(layer_neuron_idx_mapping[0])
                    }
                }
        return features_dict


class MotifsCount(Feature):
    def __init__(
            self,
            motifs_count: List[int],
    ):
        self.motifs_count: List[int] = motifs_count


class Structure(Feature):
    def __init__(
            self,
            dimensions: List[int],
            num_layers: int,
    ):
        self.dimensions: List[int] = dimensions
        self.num_layers: int = num_layers
        self.num_neurons: int = sum(dimensions)


class StructuralFeatures:
    def __init__(
            self,
            modularity: Optional[ModularityFeatures] = None,
            entropy: Optional[EntropyFeatures] = None,
            connectivity: Optional[ConnectivityFeatures] = None,
            motifs: Optional[MotifsCount] = None,
            structure: Optional[Structure] = None,

    ):
        self.modularity = modularity
        self.entropy = entropy
        self.connectivity = connectivity
        self.motifs = motifs
        self.structure = structure

    def get_class_values(
            self,
            features_list: Optional[List[str]] = None,
    ) -> List[Union[Tuple, float]]:
        all_values = []
        for val in self.__dict__.values():
            if val is None:
                continue
            all_values += val.get_class_values(
                features_list=features_list,
            )
        return all_values

    def get_features(
            self,
            layer_neuron_idx_mapping: List[List[Tuple[int, int]]],
            features_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        features_dict = {}
        for features_class in self.__dict__.values():
            if features_class is None:
                continue
            features_dict = {
                **features_dict,
                **features_class.get_features(
                    layer_neuron_idx_mapping=layer_neuron_idx_mapping,
                    features_list=features_list,
                )
            }
        return features_dict

    def get_class_global_values(self):
        return self.get_class_values(
            features_list=GLOBAL_FEATURES,
        )

    def get_class_global_features(
            self,
            layer_neuron_idx_mapping: List[List[Tuple[int, int]]],
    ) -> Dict[str, float]:
        return self.get_features(
            layer_neuron_idx_mapping=layer_neuron_idx_mapping,
            features_list=GLOBAL_FEATURES,
        )

    def get_class_constant_values(self):
        return self.get_class_values(
            features_list=CONSTANT_FEATURES,
        )

    def get_class_constant_features(
            self,
            layer_neuron_idx_mapping: List[List[Tuple[int, int]]],
    ) -> Dict[str, float]:
        return self.get_features(
            layer_neuron_idx_mapping=layer_neuron_idx_mapping,
            features_list=CONSTANT_FEATURES,
        )
