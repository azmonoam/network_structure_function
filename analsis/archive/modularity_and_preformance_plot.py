import matplotlib.pyplot as plt
import pandas as pd

from analsis.analsis_utils.utils import COLORS

experiment_name = 'MVG_20221113_rate100'
path = f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{experiment_name}/results.csv'
destination_file_name = f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{experiment_name}/{experiment_name}_performances.png'
res = pd.read_csv(path)
performances = res['performances'].to_list()
modularity = res['modularity'].to_list()


fig, axs = plt.subplots(2, 1)
axs[0].plot(performances, color=COLORS[0])
axs[0].set_xticks([])
axs[0].set_ylabel('Performance')
axs[1].plot(modularity, color=COLORS[0])
axs[1].set_ylabel('Modularity')
axs[1].set_xlabel('Generation')
plt.suptitle('MVG - switch every 100')
fig.tight_layout()
plt.show()
fig.savefig(f'/Users/noamazmon/PycharmProjects/network_modularity/20221211_plots/MVG_switch100_{experiment_name}.png')