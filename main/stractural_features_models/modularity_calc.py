from typing import List

import networkx.algorithms.community as nx_comm

from new_organism import Organism
from stractural_features_models.structural_features import ModularityFeatures


class ModularityCalc:
    def __init__(
            self,
            organism: Organism,
    ):
        self.organism = organism

    def calc_modularity(
            self,
    ) -> ModularityFeatures:
        cutoff = self.organism.communities_input_symmetry
        best_n = self.organism.communities_input_symmetry
        if not self.organism.communities_input_symmetry:
            cutoff = 1
            best_n = None
        communities = nx_comm.greedy_modularity_communities(
            G=self.organism.clean_network,
            cutoff=cutoff,
            best_n=best_n,
        )
        self._set_node_community(
            communities=communities,
        )
        modularity = nx_comm.modularity(
            G=self.organism.clean_network,
            communities=communities,
            weight=None,
        )
        return ModularityFeatures(
            modularity=modularity,
            communities=communities,
        )

    def _set_node_community(
            self,
            communities: List[list[int]],
    ):
        for c, v_c in enumerate(communities):
            for v in v_c:
                self.organism.clean_network.nodes[v]['community'] = c
