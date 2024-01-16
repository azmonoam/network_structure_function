import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from tqdm import tqdm

from analsis.analsis_utils.utils import get_organism_from_pkl
from stractural_features_models.modularity_calc import ModularityCalc
from organism import Organism


def get_modularity_for_all_organisms(
        exp_name: str,
) -> List[Organism]:
    organisms = []
    get_generation = lambda x: int(x.split('best_')[-1].split('.pkl')[0])
    folder = Path(f"/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{exp_name}/best_network")
    for path in tqdm(sorted(os.listdir(path=folder), key=get_generation)):
        organism = get_organism_from_pkl(
            path=f'{folder}/{path}',
        )
        modularity_calc = ModularityCalc(
            network=organism.layers,
        )
        communities, modularity = modularity_calc.calc_modularity()
        organism.modularity = modularity
        organism.communities = communities
        organism.generation = get_generation(path)
        organisms.append(organism)
    return organisms


def main(
        exp_name: str,
):
    organisms = get_modularity_for_all_organisms(
        exp_name=exp_name,
    )
    generations = [organism.generation for organism in organisms]
    modularities = [organism.modularity for organism in organisms]
    plt.scatter(generations, modularities)
    plt.xlabel('generation')
    plt.ylabel('modularity')
    plt.show()


if __name__ == '__main__':
    experiment_name = '2022-12-02_14:37:35'
    main(experiment_name)
