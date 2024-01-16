from datetime import datetime as dt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

blues = ["#0d88e6", "#a7d5ed", "#0d88e6", ]
rads = ["#b30000", "#e1a692", "#b30000", ]
greens = ["#529471", "#48b5c4", "#529471"]
light_oranges = ["#ef9b20", "#ede15b", "#ef9b20", ]
oranges = ["#fd841f", "#ffb55a", "#fd841f", ]
purples = ["#5e569b", "#beb9db", "#5e569b", ]
pinks = ["#ff80ff", "#e4bcad", "#ff80ff"]
grays = ["#5c5757", "#d7e1ee", "#5c5757", ]
limes = ["#c9e52f", "#d0f400", "#c9e52f", ]
bright_pinks = ["#eb44e8", "#ff80ff", "#eb44e8", ]
turquoise = ["#466964", "#6cd4c5", "#466964", ]
browns = ["#704f4f", "#a77979", "#704f4f", ]
sage = ["#557153", "#a9af7e", "#557153", ]
strong_blue = ["#4636fc", "#aee1fc", "#4636fc"],
maroon = ["#900c3f", "#c70039", '#900c3f']
colors = [blues, rads, purples, oranges, greens, grays, pinks, limes, turquoise, light_oranges,
          browns, sage, strong_blue, maroon, bright_pinks]


def remove_outliers_for_dfs(
        df: pd.DataFrame,
        columns_to_edit: List[str],
        min_quantile: float = 0.01,
        max_quentile: float = 0.99,
) -> pd.DataFrame:
    for col in columns_to_edit:
        df = df[df[col].between(df[col].quantile(min_quantile), df[col].quantile(max_quentile))]
    return df


def distance(
        first_point: List[float],
        second_point: List[float],
):
    x1, y1, z1 = first_point
    x2, y2, z2 = second_point
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def plot_3d_all(
        order: Tuple[str] = ('modularity', 'performances', 'connection_cost')
):
    destination_file_name_3d = f'/Users/noamazmon/PycharmProjects/network_modularity/' \
                               f'/3d_devil_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png'
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i, net_per in enumerate(data['performances_start'].unique()):
        labels = [None, None, None]
        if i == 0:
            labels = ['data', 'mean', 'median']
        singel_net_data = data[data['performances_start'] == net_per]
        singel_net_data = remove_outliers_for_dfs(
            df=singel_net_data,
            columns_to_edit=['modularity_end', 'performances_end', 'connection_cost_end']
        )
        ax.scatter3D(
            singel_net_data[f'{order[0]}_start'],
            singel_net_data[f'{order[1]}_start'],
            singel_net_data[f'{order[2]}_start'],
            s=30,
            color=colors[i][0],
            label=labels[0]
        )
        ax.scatter3D(
            singel_net_data[f'{order[0]}_end'],
            singel_net_data[f'{order[1]}_end'],
            singel_net_data[f'{order[2]}_end'],
            alpha=0.01,
            s=30,
            color=colors[i][1],
        )
        starts = [
            singel_net_data[f'{order[0]}_start'].mean(),
            singel_net_data[f'{order[1]}_start'].mean(),
            singel_net_data[f'{order[2]}_start'].mean(),
        ]
        mean_lis = [
            singel_net_data[f'{order[0]}_end'].mean(),
            singel_net_data[f'{order[1]}_end'].mean(),
            singel_net_data[f'{order[2]}_end'].mean(),
        ]
        median_lis = [
            singel_net_data[f'{order[0]}_end'].median(),
            singel_net_data[f'{order[1]}_end'].median(),
            singel_net_data[f'{order[2]}_end'].median(),
        ]
        ax.scatter3D(
            mean_lis[0], mean_lis[1], mean_lis[2],
            s=40,
            marker='s',
            color=colors[i][2],
            label=labels[1]
        )
        ax.scatter3D(
            median_lis[0], median_lis[1], median_lis[2],
            s=40,
            marker='^',
            color=colors[i][2],
            label=labels[2]
        )
        print(
            f"start: {starts[0]:.3f}, {starts[1]:.3f}, {starts[2]:.3f} "
            f'mean: {mean_lis[0]:.3f}, {mean_lis[1]:.3f}, {mean_lis[2]:.3f} '
            f'median: {median_lis[0]:.3f}, {median_lis[1]:.3f}, {median_lis[2]:.3f}'
        )
    ax.view_init(20, azim=-50)
    ax.set_xlabel(order[0].replace('_', ' '))
    ax.set_ylabel(order[1].replace('_', ' '))
    ax.set_zlabel(order[2].replace('_', ' '))
    ax.set_title('Network Evolution Space')
    plt.legend(loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    fig.savefig(destination_file_name_3d)


def plot_3d_medians(
        order: Tuple[str] = ('modularity', 'performances', 'connection_cost')
):
    destination_file_name_3d = f'/Users/noamazmon/PycharmProjects/network_modularity/' \
                               f'/3d_devil_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png'
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i, net_per in enumerate(data['performances_start'].unique()):
        labels = [None, None, None]
        if i == 0:
            labels = ['data', 'mean', 'median']
        singel_net_data = data[data['performances_start'] == net_per]
        singel_net_data = remove_outliers_for_dfs(
            df=singel_net_data,
            columns_to_edit=['modularity_end', 'performances_end', 'connection_cost_end']
        )
        ax.scatter3D(
            singel_net_data[f'{order[0]}_start'],
            singel_net_data[f'{order[1]}_start'],
            singel_net_data[f'{order[2]}_start'],
            s=30,
            color=colors[i][0],
            label=labels[0]
        )
        median_lis = [
            singel_net_data[f'{order[0]}_end'].median(),
            singel_net_data[f'{order[1]}_end'].median(),
            singel_net_data[f'{order[2]}_end'].median(),
        ]

        ax.scatter3D(
            median_lis[0], median_lis[1], median_lis[2],
            s=40,
            marker='^',
            color=colors[i][2],
            label=labels[2]
        )
    ax.view_init(azim=20)
    ax.set_xlabel(order[0].replace('_', ' '))
    ax.set_ylabel(order[1].replace('_', ' '))
    ax.set_zlabel(order[2].replace('_', ' '))
    ax.set_title('Network Evolution Space')
    plt.legend(loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    fig.savefig(destination_file_name_3d)


def plot_3d_means(
        order: Tuple[str] = ('modularity', 'performances', 'connection_cost')
):
    destination_file_name_3d = f'/Users/noamazmon/PycharmProjects/network_modularity/' \
                               f'/3d_devil_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png'
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i, net_per in enumerate(data['performances_start'].unique()):
        labels = [None, None, None]
        if i == 0:
            labels = ['data', 'mean', 'median']
        singel_net_data = data[data['performances_start'] == net_per]
        singel_net_data = remove_outliers_for_dfs(
            df=singel_net_data,
            columns_to_edit=['modularity_end', 'performances_end', 'connection_cost_end']
        )
        ax.scatter3D(
            singel_net_data[f'{order[0]}_start'],
            singel_net_data[f'{order[1]}_start'],
            singel_net_data[f'{order[2]}_start'],
            s=30,
            color=colors[i][0],
            label=labels[0]
        )
        mean_lis = [
            singel_net_data[f'{order[0]}_end'].mean(),
            singel_net_data[f'{order[1]}_end'].mean(),
            singel_net_data[f'{order[2]}_end'].mean(),
        ]
        ax.scatter3D(
            mean_lis[0], mean_lis[1], mean_lis[2],
            s=40,
            marker='s',
            color=colors[i][2],
            label=labels[1]
        )

    ax.view_init(20, azim=20)
    ax.set_xlabel(order[0].replace('_', ' '))
    ax.set_ylabel(order[1].replace('_', ' '))
    ax.set_zlabel(order[2].replace('_', ' '))
    ax.set_title('Network Evolution Space')
    plt.legend(loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    fig.savefig(destination_file_name_3d)


if __name__ == '__main__':
    order = ('modularity', 'performances', 'connection_cost')
    add_data = False
    csvs_to_add = [
        "starts_and_ends_of_all_networks_26_12_22_2.csv"
    ]
    base_path = '/'
    all_net_csv_name = 'all_networks_starts_and_ends.csv'
    data = pd.read_csv(f'{base_path}/{all_net_csv_name}', index_col=False)
    if add_data:
        for csv in csvs_to_add:
            data = pd.concat([data, pd.read_csv(f'{base_path}/{csv}', index_col=False)], ignore_index=True)
        data.to_csv(f'{base_path}/{all_net_csv_name}', index=False)
    all_starts = []
    all_medians = []
    all_means = []
    distances_to_starts = np.zeros((len(data['performances_start'].unique()), len(data['performances_start'].unique())))
    distances_to_medians = np.zeros(
        (len(data['performances_start'].unique()), len(data['performances_start'].unique())))
    distances_to_means = np.zeros((len(data['performances_start'].unique()), len(data['performances_start'].unique())))

    for i, net_per in enumerate(data['performances_start'].unique()):
        singel_net_data = data[data['performances_start'] == net_per]
        all_starts.append([
            singel_net_data[f'{order[0]}_start'].mean() / data[f'{order[0]}_start'].max(),
            singel_net_data[f'{order[1]}_start'].mean() / data[f'{order[1]}_start'].max(),
            singel_net_data[f'{order[2]}_start'].mean() / data[f'{order[2]}_start'].max(),
        ])
        all_means.append([
            singel_net_data[f'{order[0]}_end'].mean() / data[f'{order[0]}_end'].max(),
            singel_net_data[f'{order[1]}_end'].mean() / data[f'{order[1]}_end'].max(),
            singel_net_data[f'{order[2]}_end'].mean() / data[f'{order[2]}_end'].max(),
        ])
        all_medians.append(
            [
                singel_net_data[f'{order[0]}_end'].median() / data[f'{order[0]}_end'].max(),
                singel_net_data[f'{order[1]}_end'].median() / data[f'{order[1]}_end'].max(),
                singel_net_data[f'{order[2]}_end'].median() / data[f'{order[2]}_end'].max(),
            ])
    for i in range(len(data['performances_start'].unique())):
        for j in range(len(data['performances_start'].unique())):
            distances_to_starts[i][j] = distance(all_starts[i], all_starts[j])
            distances_to_medians[i][j] = distance(all_means[i], all_means[j])
            distances_to_means[i][j] = distance(all_medians[i], all_medians[j])
    plot_3d_all()
    plot_3d_means()
    plot_3d_medians()
