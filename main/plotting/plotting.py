import os
from typing import List, Optional

import imageio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from organism import Organism
from plot_network_nx import draw_network
from utils.main_utils import COLORS


def draw_network_and_performance(
        best_organisms: List[Organism],
        performances: List[float],
        folder: str,
        optimization_method: str,
        save_photos: bool = False,
        plot_every: int = 100,
):
    files_list = []
    network_temp_file_name = f"{folder}/network_temp.png"
    performance_temp_file_name = f"{folder}/performance_temp.png"
    for i in range(len(best_organisms)):
        if i % plot_every == 0:
            organism = best_organisms[i]
            file_name = f'evolve_{i}_.png'
            files_list.append(file_name)
            draw_network(
                organism=organism,
                generation=i,
                network_file_name=network_temp_file_name,
                title=False,
            )
            plot_performance_by_generation(
                performances=performances[:i],
                max_generations=len(performances),
                file_name=performance_temp_file_name,
                y_lim=[min(performances), max(performances) + 0.02],
                title=False,
            )
            if optimization_method == 'performance_and_connection_cost':
                title = f'Generation: {i}, performance: {organism.performance:.3f},\n' \
                        f'modularity: {organism.modularity:.3f} , connection cost: {organism.connection_cost}'
            else:
                title = f'Generation: {i}, performance: {organism.performance:.3f},\n' \
                        f'modularity: {organism.modularity:.3f}'
            plot_double_from_file(
                network_network_temp_file_namefile_name=network_temp_file_name,
                performance_temp_file_name=performance_temp_file_name,
                title=title,
                save_path=f"{folder}/{file_name}",
            )
    sorted(files_list, key=lambda y: int(y.split('_')[1]))
    _create_gif(
        sorted_files_list=files_list,
        folder=folder,
    )
    if not save_photos:
        for file in files_list:
            os.remove(f'{folder}/{file}')
        os.remove(performance_temp_file_name)
        os.remove(network_temp_file_name)


def plot_performance_by_generation(
        performances: List[float],
        max_generations: int,
        y_lim: List[float],
        file_name: Optional[str] = None,
        title: Optional[bool] = True,
):
    plt.figure(figsize=(5, 6), dpi=80)
    plt.plot(performances, label='performances', c=COLORS[0])
    plt.xlabel('generation')
    plt.ylabel('performance')
    plt.xlim([0, max_generations])
    plt.ylim(y_lim)
    # plt.legend(loc='upper left', )
    if title:
        plt.title(f'performance per generation')
    if file_name:
        plt.savefig(file_name, dpi=80)
    plt.close()


def _create_gif(
        sorted_files_list: List[str],
        folder: str,
        frame_length: float = 0.07,
        end_pause: float = 3,
):
    images = []
    for ii in range(0, len(sorted_files_list)):
        if ii == len(sorted_files_list) - 1:
            for jj in range(0, int(end_pause / frame_length)):
                images.append(imageio.v2.imread(f'{folder}/{sorted_files_list[ii]}'))
        else:
            images.append(imageio.v2.imread(f'{folder}/{sorted_files_list[ii]}'))
    imageio.mimsave(f'{folder}/evolve.gif', images, 'GIF', duration=frame_length)


def plot_double_from_file(
        network_network_temp_file_namefile_name: str,
        performance_temp_file_name: str,
        title: str,
        save_path: str,
):
    fig = plt.figure()
    img1 = mpimg.imread(network_network_temp_file_namefile_name)
    img2 = mpimg.imread(performance_temp_file_name)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img2)
    ax1.axis('off')
    ax2.axis('off')
    fig.suptitle(title)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.5,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
