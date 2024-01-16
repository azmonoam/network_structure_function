import matplotlib.pyplot as plt
import pandas as pd

from plotting.plotting import plot_performance_by_generation

experiment_name = 'MVG_20221113_rate100'
path = f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{experiment_name}/results.csv'
destination_file_name = f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{experiment_name}/{experiment_name}_performances.png'
res = pd.read_csv(path)
performances = res['performances'].to_list()
plot_performance_by_generation(
    performances=performances,
    max_generations=len(performances),
    file_name=destination_file_name,
    y_lim=[min(performances), max(performances) + 0.02],
    title=False,
)
plt.show()
