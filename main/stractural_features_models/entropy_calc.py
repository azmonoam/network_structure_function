import networkx as nx
import numpy as np

from new_organism import Organism
from stractural_features_models.structural_features import EntropyFeatures


class EntropyCalc:
    def __init__(
            self,
            organism: Organism,
            num_connections: int,
    ):
        self.organism = organism
        self.num_connections = num_connections

    def calc_entropy_features(self):
        entropy = self.get_entropy()
        normed_entropy = self.get_normed_entropy(
            entropy=entropy,
        )
        return EntropyFeatures(
            entropy=entropy,
            normed_entropy=normed_entropy,
        )

    def get_entropy(self) -> float:
        deg = [d for n, d in self.organism.network.in_degree() if n > self.organism.layer_neuron_idx_mapping[0][-1]]
        prob_list = [i / sum(deg) for i in deg if i != 0]
        return -(prob_list * np.log(np.abs(prob_list))).sum()

    def get_normed_entropy(
            self,
            entropy: float,
    ) -> float:
        return entropy / np.log(
            min(self.num_connections, len(set(self.organism.network.nodes) - set(nx.isolates(self.organism.network)))))
