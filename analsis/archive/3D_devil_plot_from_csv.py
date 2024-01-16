from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm

experiment_name = '141222_10prec'
path = f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{experiment_name}/results.csv'
destination_file_name_3d = f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{experiment_name}' \
                        f'/3d_devil_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png'
destination_file_name_2d = f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{experiment_name}' \
                        f'/2d_devil_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}.png'
res = pd.read_csv(path)
small_res = res[::100]
performances = small_res['performances'].to_list()
modularity = small_res['modularity'].to_list()
cost = small_res['connection_cost'].to_list()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, )
p3d = ax.scatter(cost, modularity, s=[-4 / np.log(x / 1.01) for x in performances],
                 c=np.linspace(0, res.shape[0], small_res.shape[0]),
                 cmap=cm.coolwarm)
labels = sorted({f'{x:.1f}' for x in performances})
handles, _ = p3d.legend_elements(prop="sizes", num=len(labels) + 1, alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper right", title="performance")
ax.set_xlabel('connection cost')
ax.set_ylabel('modularity')
ax.set_title('Network Evolution')
fig.colorbar(p3d, shrink=0.8, pad=0.15, location='left', label='generation')
plt.savefig(destination_file_name_2d)
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
p3d = ax.scatter(cost, modularity, performances, s=30, c=np.linspace(0,  res.shape[0], small_res.shape[0]),
                 cmap=cm.coolwarm)
ax.set_xlabel('connection cost')
ax.set_ylabel('modularity')
ax.set_zlabel('performances')
ax.set_title('Network Evolution')
fig.colorbar(p3d, shrink=0.5, pad=0.03, location='left', label='generation')
plt.tight_layout()
plt.savefig(destination_file_name_3d)
plt.show()
