import copy
import random
from typing import List, Optional
from utils.tasks_params import TaskParameters

import networkx as nx
import numpy as np

from neuroevolution import Neuroevolution
from new_organism import Organism
from organism import Organism as OldOrg
from stractural_features_models.structural_features import StructuralFeatures


def old_organism_creator(
        neuroevolution: Neuroevolution,
        dimensions: List[int],
) -> OldOrg:
    return OldOrg(
        neuroevolution=neuroevolution,
        dimensions=dimensions,
    )


def connectivity_based_old_organism_creator(
        neuroevolution: Neuroevolution,
        dimensions: List[int],
        connectivity_ratio: float,
        max_allowed_connections: int,
) -> OldOrg:
    return OldOrg(
        neuroevolution=neuroevolution,
        dimensions=dimensions,
        connectivity_ratio=connectivity_ratio,
        max_allowed_connections=max_allowed_connections,
    )


def get_start_dims(
        num_layers: int,
        input_size: int,
        output_size: int = 2,
):
    start_dimensions = [input_size, ]
    for i in range(num_layers - 1):
        start_dimensions.append(random.randint(2, start_dimensions[-1]))
    start_dimensions.append(output_size)
    return start_dimensions


def get_population_fixed_size_clone_possible(
        clones_only: bool,
        start_dimensions: List[int],
        max_connections_per_neuron_in_layer: Optional[List[int]],
        neuroevolution: Neuroevolution,
) -> List[OldOrg]:
    if clones_only:
        organism = old_organism_creator(
            neuroevolution=neuroevolution,
            dimensions=start_dimensions,
        )
        organism.build_organism(
            max_connections_per_neuron_in_layer=max_connections_per_neuron_in_layer,
        )
        organism.generation = 0
        organism.predictions = organism.predict(
            input_array=neuroevolution.inputs,
        )
        population = [
            copy.deepcopy(organism) for _ in range(neuroevolution.population_size)
        ]
    else:
        population = []
        for _ in range(neuroevolution.population_size):
            organism = old_organism_creator(
                neuroevolution=neuroevolution,
                dimensions=start_dimensions,

            )
            organism.build_organism()
            organism.predictions = organism.predict(
                input_array=neuroevolution.inputs,
            )
            population.append(organism)
    return population


def get_population_with_random_dims(
        neuroevolution: Neuroevolution,
) -> List[OldOrg]:
    population = []
    start_dimensions = get_start_dims(
        num_layers=neuroevolution.num_layers,
        input_size=neuroevolution.input_dim
    )
    max_connections_per_neuron_in_layer = [
        random.randint(2, start_dimensions[i + 1])
        for i in range(len(start_dimensions) - 1)
    ]
    for i in range(neuroevolution.population_size):
        organism = old_organism_creator(
            neuroevolution=neuroevolution,
            dimensions=start_dimensions,
        )
        organism.build_organism(
            max_connections_per_neuron_in_layer=max_connections_per_neuron_in_layer,
        )
        organism.generation = 0
        population.append(organism)
    return population


def get_population_with_random_dims_different_size_each(
        neuroevolution: Neuroevolution,
) -> List[OldOrg]:
    population = []
    for i in range(neuroevolution.population_size):
        start_dimensions = get_start_dims(
            num_layers=neuroevolution.num_layers,
            input_size=neuroevolution.input_dim
        )
        max_connections_per_neuron_in_layer = [
            random.randint(2, start_dimensions[i + 1])
            for i in range(len(start_dimensions) - 1)
        ]
        organism = old_organism_creator(
            neuroevolution=neuroevolution,
            dimensions=start_dimensions,
        )
        organism.build_organism(
            max_connections_per_neuron_in_layer=max_connections_per_neuron_in_layer,
        )
        organism.generation = 0
        organism.predictions = organism.predict(
            input_array=neuroevolution.inputs,
        )
        population.append(organism)
    return population


def get_population_by_connectivity_ratio(
        neuroevolution: Neuroevolution,
        start_dimensions: List[int],
        connectivity_ratio: float,
) -> List[OldOrg]:
    population = []
    fully_connected_num_connection = sum(
        start_dimensions[i] * start_dimensions[i + 1]
        for i in range(neuroevolution.num_layers)
    )
    max_allowed_connections = int(np.floor(fully_connected_num_connection * connectivity_ratio))
    _test_connection_ratio(
        max_allowed_connections=max_allowed_connections,
        start_dimensions=start_dimensions,
        connectivity_ratio=connectivity_ratio,
        fully_connected_num_connection=fully_connected_num_connection,
    )
    for i in range(neuroevolution.population_size):
        organism = connectivity_based_old_organism_creator(
            neuroevolution=neuroevolution,
            dimensions=start_dimensions,
            connectivity_ratio=connectivity_ratio,
            max_allowed_connections=max_allowed_connections,
        )
        organism.build_organism_by_connectivity(
            max_allowed_connections=max_allowed_connections,
        )
        organism.generation = 0
        organism.predictions = organism.predict(
            input_array=neuroevolution.inputs,
        )
        population.append(organism)
    return population


def get_old_organism_by_connectivity_ratio(
        neuroevolution: Neuroevolution,
        start_dimensions: List[int],
        connectivity_ratio: float,
) -> OldOrg:
    fully_connected_num_connection = sum(
        start_dimensions[i] * start_dimensions[i + 1]
        for i in range(neuroevolution.num_layers)
    )
    max_allowed_connections = int(np.floor(fully_connected_num_connection * connectivity_ratio))
    organism = old_organism_creator(
        neuroevolution=neuroevolution,
        dimensions=start_dimensions,
    )
    organism.build_organism_by_connectivity(
        max_allowed_connections=max_allowed_connections,
    )
    organism.predictions = organism.predict(
        input_array=neuroevolution.inputs,
    )
    return organism


def _test_connection_ratio(
        max_allowed_connections: int,
        start_dimensions: List[int],
        connectivity_ratio: float,
        fully_connected_num_connection: int,
):
    min_connections = sum(start_dimensions[:-1])
    if min_connections > max_allowed_connections:
        min_allowed_connection_rate = min_connections / fully_connected_num_connection
        raise Exception(
            f"The connection rate {connectivity_ratio} is too low the minimum allowed connection rate is"
            f" {min_allowed_connection_rate}")


def get_organism_by_connectivity_ratio(
        task_params: TaskParameters,
        connectivity_ratio: float,
) -> Organism:
    max_allowed_connections = int(np.ceil(task_params.max_possible_connections * connectivity_ratio))
    organism = Organism(
        dimensions=task_params.start_dimensions,
        num_layers=task_params.num_layers,
        allowed_weights_values=task_params.allowed_weights_values,
        allowed_bias_values=task_params.allowed_bias_values,
        communities_input_symmetry=task_params.communities_input_symmetry,
        max_allowed_connections=max_allowed_connections,
        network=nx.DiGraph(),
        structural_features=StructuralFeatures(),
    )
    organism.build_organism_by_connectivity(
        max_allowed_connections=max_allowed_connections,
    )
    return organism
