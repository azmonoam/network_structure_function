from datetime import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd

COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']


def plot_hists(first_analysis_df: pd.DataFrame):
    fig = plt.figure()
    plt.hist([first_analysis_df['success_percent_0.9'], first_analysis_df['success_percent_1.0']], bins=15,
             label=['performance > 90%', 'performance = 100%'], alpha=0.7, color=COLORS[:2])
    plt.title('High performance proportion')
    plt.xlabel('percentage of successful network (out of 300 trials)')
    plt.ylabel('number of architectures')
    plt.legend()
    plt.show()
    fig.savefig(f'{base_path}/plots/teachbility_plots/all_prec_{dt.now().strftime("%Y-%m-%d-%H-%M")}.png')
    fig = plt.figure()
    plt.hist(first_analysis_df['success_percent_1.0'], bins=15, alpha=0.7, color=COLORS[0])
    plt.title('percentage of networks with performance 100%')
    plt.xlabel('percentage of successful network (out of 300 trials)')
    plt.ylabel('number of architectures')
    plt.show()
    fig.savefig(f'{base_path}/plots/teachbility_plots/100_prec_{dt.now().strftime("%Y-%m-%d-%H-%M")}.png')
    fig = plt.figure()
    plt.hist(first_analysis_df['success_percent_0.9'], bins=15, alpha=0.7, color=COLORS[1])
    plt.title('percentage of networks with performance higher then 90%')
    plt.xlabel('percentage of successful network (out of 300 trials)')
    plt.ylabel('number of architectures')
    plt.show()
    fig.savefig(f'{base_path}/plots/teachbility_plots/90_prec_{dt.now().strftime("%Y-%m-%d-%H-%M")}.png')


if __name__ == '__main__':
    base_path = '/'
    folder = f"{base_path}"
    first_analysis_df = pd.read_csv(f"{folder}/2023-01-02-12-36-34_first_analysis.csv").drop("Unnamed: 0", axis=1)
    all_res_df = pd.read_csv(f"{folder}/2023-01-02-12-36-34_all_res_df.csv").drop("Unnamed: 0", axis=1)
    plot_hists(
        first_analysis_df=first_analysis_df,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 6))

    ax1.set_ylabel('number of architectures')
    ax1.hist([
        all_res_df['modularity'][all_res_df['Performances'] == 1.0],
        all_res_df['modularity'][all_res_df['Performances'] != 1.0],
    ], bins=15, alpha=0.7, color=COLORS[:2], density=True,
        label=['reached 100% performance', 'didnt reach 100% performance'])
    ax2.hist([
        all_res_df['num_connections'][all_res_df['Performances'] == 1.0],
        all_res_df['num_connections'][all_res_df['Performances'] != 1.0],
    ], bins=15, alpha=0.7, color=COLORS[:2], density=True,
        label=['reached 100% performance', 'didnt reach 100% performance'])

    fig.suptitle('Value frequency', y=0.92)
    ax1.set_xlabel('Modularity')
    ax2.set_xlabel('Number of connections')
    box1 = ax1.get_position()
    box2 = ax2.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width * 0.95, box1.height * 0.9, ])
    ax2.set_position([box2.x0, box2.y0, box2.width * 0.95, box2.height * 0.9, ])

    # Put a legend to the right of the current axis
    ax1.legend(loc='upper center', ncol=2, fancybox=True, bbox_to_anchor=(1.2, 1.1))
    plt.show()
    fig.savefig(f'{base_path}/plots/teachbility_plots/mod_connection_freq_{dt.now().strftime("%Y-%m-%d-%H-%M")}.png')
