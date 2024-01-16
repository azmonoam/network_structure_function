import matplotlib.pyplot as plt
import os
import joblib

base_path = "/Volumes/noamaz/modularity"
ergm_path = f'{base_path}/retina_xor/retina_3_layers_3_4/ergm_results/6_features_nice/'
ergm_path2 = f'{base_path}/retina_xor/retina_3_layers_3_4/ergm_results/3_top_features/'

names_only_good1 = []
all_datas = []
names_only_good2 = []
all_datas2 = []
for pkl_file in os.listdir(ergm_path):
    try:
        with open(f"{ergm_path}/{pkl_file}", 'rb') as fp:
            all_datas.append(joblib.load(fp))
            names_only_good1.append(pkl_file)
    except:
        continue

for pkl_file in os.listdir(ergm_path2):
    try:
        with open(f"{ergm_path2}/{pkl_file}", 'rb') as fp:
            all_datas2.append(joblib.load(fp))
            names_only_good2.append(pkl_file)
    except:
        continue

for data in all_datas:
    if 'all_erors' not in data.keys():
        continue
    data['sum_errors'] = [sum(a) for a in data['all_erors']]

for data in all_datas2:
    if 'all_erors' not in data.keys():
        continue
    data['sum_errors'] = [sum(a) for a in data['all_erors']]

for name, data in zip(names_only_good1, all_datas):
    if 'sum_errors' not in data.keys():
        continue
    plt.plot(range(len(data['sum_errors'])), data['sum_errors'])
    plt.title(name)
    plt.show()

for name, data in zip(names_only_good2, all_datas2):
    if 'sum_errors' not in data.keys():
        continue
    plt.plot(range(len(data['sum_errors'])), data['sum_errors'], c='red')
    plt.title(name)
    plt.show()
