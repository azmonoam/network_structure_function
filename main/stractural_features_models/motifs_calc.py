import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

from new_organism import Organism
from stractural_features_models.structural_features import MotifsCount


class MotifsCountCalc:
    def __init__(
            self,
            is_ff: bool = True
    ):
        all_possible_motiffs = [
            np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 1, 1], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 1], [0, 0, 0], [1, 1, 0]]),
            np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0]]),
            np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0]]),
            np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]]),
            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        ]
        all_possible_FF_motiffs = [
            np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
        ]
        if is_ff:
            self.motifs_list = all_possible_FF_motiffs
        else:
            self.motifs_list = all_possible_motiffs

    def get_motif_count(
            self,
            organism: Organism,
    ) -> MotifsCount:
        motifs_count = []
        for ind, motif_example in enumerate(self.motifs_list):
            motif_example_graph = nx.from_numpy_array(motif_example, create_using=nx.DiGraph)
            matcher = isomorphism.DiGraphMatcher(organism.network, motif_example_graph)
            motifs = matcher.subgraph_isomorphisms_iter()
            motifs_count.append(
                len(
                    {
                        frozenset(motif_mapping.keys())
                        for motif_mapping in motifs
                    }
                )
            )
        return MotifsCount(
            motifs_count=motifs_count,
        )
