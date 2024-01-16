from typing import Optional

import networkx as nx
import numpy as np

from organism import Organism


def get_entropy(
        organism: Organism,
) -> float:
    deg = [d for n, d in organism.network.in_degree() if n > organism.layer_neuron_idx_mapping[0][-1]]
    prob_list = [i / sum(deg) for i in deg if i != 0]
    return -(prob_list * np.log(np.abs(prob_list))).sum()


def get_clustering_coeff(
        organism: Organism,
) -> float:
    return nx.average_clustering(
        G=organism.network,
        count_zeros=False,
    )


def get_average_shortest_path_length(
        organism: Organism,
) -> Optional[float]:
    try:
        return nx.average_shortest_path_length(
            G=organism.network,
        )
    except nx.NetworkXError:
        return None


def get_normed_entropy(
        entropy: float,
        organism: Organism,
) -> float:
    num_connections = organism.connection_cost
    return entropy / np.log(
        min(num_connections, len(set(organism.network.nodes) - set(nx.isolates(organism.network)))))
