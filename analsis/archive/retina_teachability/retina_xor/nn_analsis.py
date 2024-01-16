from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim

COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f', "#c77dff", "#f7d6e0"]
base_path = '/Volumes/noamaz/modularity'
model_name = 'retina_xor_2023-09-12-16-28-39_lr_0.001_bs_512_output_10k_ep_adj_False_meta_True_2k_ephoc_no_mod_model_cpu.pkl'
train_path = 'retina_xor_train_2023-09-04-11-44-55_adj_False_meta_True_10k_ep_no_mod.pkl'
test_path = 'retina_xor_test_2023-09-04-11-44-55_adj_False_meta_True_10k_ep_no_mod.pkl'
task_data_folder_name = 'train_test_data'
time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

dims = [6, 3, 4, 2]
num_layers = len(dims) - 1
task_params = RetinaByDim(
    start_dimensions=dims,
    num_layers=num_layers,
    by_epochs=False,
    task_base_folder_name='retina_xor',
    rule=LogicalGates.XOR,
)
plot_path = f'/Users/noamazmon/PycharmProjects/network_modularity/plots/{task_params.task_base_folder_name}_multi_arch/{task_params.task_version_name}'
task_params.base_path = base_path
base_path_to_res = f"{task_params.teach_arch_base_path}/{task_data_folder_name}"
with open(
        f"{task_params.teach_arch_base_path}/{task_data_folder_name}/{test_path}",
        'rb+') as fp:
    test_data = joblib.load(fp)
with open(
        f"{task_params.teach_arch_base_path}/teach_archs_regression_results/{model_name}",
        'rb+') as fp:
    model = joblib.load(fp)


test_loader = DataLoader(
    dataset=test_data,
    batch_size=10000,
    shuffle=False,
)


for test_input, test_label in test_loader:
    test_outputs = model(test_input)

res = pd.DataFrame()
test_pred = test_outputs.reshape(-1).detach() / 1000
test_label_no_increse = test_label / 1000
res['test_pred'] = test_pred
res['test_label'] = test_label_no_increse
loss = (test_pred - test_label_no_increse) ** 2
res['se'] = loss
res['error'] = loss ** 0.5

plt.scatter(res['test_label'], res['error'], s=4, c=COLORS[0])
plt.xlabel('test data label')
plt.ylabel('absolute error')
plt.title('Prediction absolute error as a function of the predicted point true mean performance', wrap=True)
plt.savefig(f"{plot_path}/{time_str}_nn_model_error_vs_actual.png")
plt.show()

plt.scatter(res['test_label'], res['test_pred'], s=4, c=COLORS[0])
start = min(res['test_label'].min(), res['test_pred'].min())
stop = max(res['test_label'].max(), res['test_pred'].max())
plt.plot(np.linspace(start, stop,  50),np.linspace(start, stop,  50), c=COLORS[4])
plt.xlabel('test data label')
plt.ylabel('test data prediction')
plt.title('Prediction mean performance vs actual mean performance', wrap=True)
plt.savefig(f"{plot_path}/{time_str}_nn_model_pred_vs_actual.png")
plt.show()

plt.hist(res['error'], bins=50, color=COLORS[2])
plt.xlabel('absolute error')
plt.title('Absolute error on prediction', wrap=True)
plt.savefig(f"{plot_path}/{time_str}_nn_model_error_hist.png")
plt.show()

print('a')