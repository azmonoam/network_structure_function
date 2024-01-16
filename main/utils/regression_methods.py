from typing import List, Optional

from new_organism import Organism
from parameters.base_params import BaseParams
from stractural_features_models.calc_structural_features import CalcStructuralFeatures


def get_list_features_names(
        task_params: BaseParams,
        features_list: Optional[List[str]] = None,
        features_to_drop: Optional[List[str]] = None,
) -> List[str]:
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
    structural_features_calculator = CalcStructuralFeatures(
        organism=organism,
    )
    organism = structural_features_calculator.calc_structural_features()

    feature_names = [
        f.replace(',', '_') for f in organism.structural_features.get_features(
            layer_neuron_idx_mapping=organism.layer_neuron_idx_mapping,
            features_list=features_list,
        ).keys()
    ]
    if features_to_drop is None:
        return feature_names
    for feature_to_drop in features_to_drop:
        f_index = feature_names.index(feature_to_drop)
        feature_names.pop(f_index)
    return feature_names
