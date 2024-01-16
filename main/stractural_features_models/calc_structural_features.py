import copy
from typing import List
from new_organism import Organism
from stractural_features_models.structural_features import StructuralFeatures
from stractural_features_models.connectivity_calc import ConnectivityCalc
from stractural_features_models.entropy_calc import EntropyCalc
from stractural_features_models.modularity_calc import ModularityCalc
from stractural_features_models.motifs_calc import MotifsCountCalc
from stractural_features_models.structural_features import (
    ModularityFeatures,
    EntropyFeatures,
    ConnectivityFeatures,
    MotifsCount,
    Structure,
)
from utils.main_utils import clean_graph_from_unused_nodes

from parameters.base_params import BaseParams


class CalcStructuralFeatures:
    def __init__(
            self,
            organism: Organism,
    ):
        self.organism = organism
        if self.organism.clean_network is None:
            self.organism.clean_network = self.clean_network()

    def clean_network(self):
        return clean_graph_from_unused_nodes(
            output_layer=len(self.organism.dimensions) - 1,
            input_layer=0,
            network=copy.deepcopy(self.organism.network),
        )

    def get_modularity(self) -> ModularityFeatures:
        modularity_calculator = ModularityCalc(
            organism=self.organism
        )
        return modularity_calculator.calc_modularity()

    def get_entropy(
            self,
            num_connections: int,
    ) -> EntropyFeatures:
        entropy_calculator = EntropyCalc(
            organism=self.organism,
            num_connections=num_connections,
        )
        return entropy_calculator.calc_entropy_features()

    def get_connectivity(self) -> ConnectivityFeatures:
        connectivity_calculator = ConnectivityCalc(
            num_layers=self.organism.num_layers,
            dimensions=self.organism.dimensions,
        )
        return connectivity_calculator.get_connectivity(
            organism=self.organism,
        )

    def get_motifs(self) -> MotifsCount:
        motifs_count_calculator = MotifsCountCalc()
        return motifs_count_calculator.get_motif_count(
            organism=self.organism,
        )

    def get_structure(self) -> Structure:
        return Structure(
            dimensions=self.organism.dimensions,
            num_layers=self.organism.num_layers,
        )

    def calc_structural_features(self) -> Organism:
        if self.organism.structural_features.modularity is None:
            self.organism.structural_features.modularity = self.get_modularity()
        if self.organism.structural_features.connectivity is None:
            self.organism.structural_features.connectivity = self.get_connectivity()
        if self.organism.structural_features.entropy is None:
            self.organism.structural_features.entropy = self.get_entropy(
                num_connections=self.organism.structural_features.connectivity.num_connections,
            )
        if self.organism.structural_features.motifs is None:
            self.organism.structural_features.motifs = self.get_motifs()
        if self.organism.structural_features.structure is None:
            self.organism.structural_features.structure = self.get_structure()
        return self.organism

    def calc_normed_structural_features(
            self,
            parameters: BaseParams,
    ) -> StructuralFeatures:
        return StructuralFeatures(
            modularity=self.normalize_modularity(),
            entropy=self.normalize_entropy(
                parameters=parameters,
            ),
            connectivity=self.normalize_connectivity(
                parameters=parameters,
            ),
            motifs=self.normalize_motifs(
                parameters=parameters,
            ),
            structure=self.normalize_structure(),
        )

    def normalize_connectivity(
            self,
            parameters: BaseParams,
    ) -> ConnectivityFeatures:
        normed_connectivity = copy.deepcopy(self.organism.structural_features.connectivity)
        normed_connectivity.num_paths_to_output_per_input_neuron = self._normalize_list_feature(
            original_values=self.organism.structural_features.connectivity.num_paths_to_output_per_input_neuron,
            normalizing_values=parameters.max_num_paths_to_output_per_input_neuron,
        )
        normed_connectivity.num_involved_neurons_in_paths_per_input_neuron = self._normalize_list_feature(
            original_values=self.organism.structural_features.connectivity.num_involved_neurons_in_paths_per_input_neuron,
            normalizing_values=parameters.max_num_involved_neurons_in_paths_per_input_neuron,
        )
        normed_connectivity.in_connections_per_layer = self._normalize_list_of_lists_feature(
            original_values=self.organism.structural_features.connectivity.in_connections_per_layer,
            normalizing_values=parameters.max_in_connections_per_layers,
        )
        normed_connectivity.out_connections_per_layer = self._normalize_list_of_lists_feature(
            original_values=self.organism.structural_features.connectivity.out_connections_per_layer,
            normalizing_values=parameters.max_out_connections_per_layers,
        )
        return normed_connectivity

    def normalize_entropy(
            self,
            parameters: BaseParams,
    ) -> EntropyFeatures:
        return EntropyFeatures(
            entropy=self.organism.structural_features.entropy.entropy / parameters.ff_entropy,
            normed_entropy=self.organism.structural_features.entropy.normed_entropy / parameters.ff_normed_entropy,
        )

    def normalize_motifs(
            self,
            parameters: BaseParams,
    ) -> MotifsCount:
        return MotifsCount(
            motifs_count=self._normalize_list_feature(
                original_values=self.organism.structural_features.motifs.motifs_count,
                normalizing_values=parameters.max_motif_count,
            ),
        )

    def normalize_modularity(
            self,
    ) -> ModularityFeatures:
        return copy.deepcopy(self.organism.structural_features.modularity)

    def normalize_structure(
            self,
    ) -> Structure:
        return copy.deepcopy(self.organism.structural_features.structure)

    @staticmethod
    def _normalize_list_feature(
            original_values: List[float],
            normalizing_values: List[float],
    ) -> List[float]:
        return [
            m / n if n != 0
            else 0
            for m, n in zip(original_values, normalizing_values)
        ]

    @staticmethod
    def _normalize_list_of_lists_feature(
            original_values: List[List[float]],
            normalizing_values: List[float],
    ) -> List[List[float]]:
        normalized_list_of_lists = []
        for list_of_vals, n in zip(original_values, normalizing_values):
            normalized_list_of_lists.append(
                [
                    m / n if n != 0
                    else 0
                    for m in list_of_vals
                ]
            )
        return normalized_list_of_lists
