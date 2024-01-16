import os
import joblib
import networkx as nx
import numpy as np
import pandas as pd

model_name = '2023-09-11-16-37-02_J7_P2'
base_path = '/Volumes/noamaz/modularity/'
task_path = f'{base_path}/retina_xor/retina_3_layers_3_4/'
models_path = f'{task_path}/ergm_results//6_features_all_features/{model_name}/teach_archs_models'
original_data_path = f'{task_path}/ergm_results//6_features_all_features/{model_name}.pkl'
csv_name = '2023-09-12-11-25-37_all_results_from_teach_archs_results_with_motifs_6000_ep'
invalid_org_names = []
with open(f"{original_data_path}", 'rb') as fp:
    orig_data = joblib.load(fp)
requiered_feature_names = orig_data['features_names']
feature_vals = {
    f: []
    for f in requiered_feature_names
}
for org_file_name in os.listdir(models_path):
    with open(f"{models_path}/{org_file_name}", 'rb') as fp:
        org = joblib.load(fp)
    if len(list(nx.isolates(org.network))) > 0:
        invalid_org_names.append(org_file_name.split('.pkl')[0])
        continue
    stractural_fetaures = org.structural_features.get_features(org.layer_neuron_idx_mapping)
    for f_name, f_val in stractural_fetaures.items():
        if f_name.replace(',', '_') in requiered_feature_names:
            feature_vals[f_name.replace(',', '_')].append(f_val)

means = [np.mean(v) for v in feature_vals.values()]
for csv_path in [ f'{task_path}/ergm_results//6_features_all_features/{model_name}/first_analysis_results/{csv_name}' + add for add in ['', '_no_duplicates']]:
    res_pd = pd.read_csv( f'{csv_path}.csv')
    res_pd_no_inbaid = res_pd[~res_pd['exp_name'].isin(invalid_org_names)]
    res_pd_no_inbaid.to_csv(f'{csv_path}_no_invalid.csv')

print('a')
