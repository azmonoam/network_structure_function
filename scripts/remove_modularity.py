import joblib
import torch

from parameters.retina_parameters import retina_structural_features_full_name_vec
task = 'digits'
base_path = f'/Volumes/noamaz/modularity/teach_archs/{task}'
train_path = f'{task}_train_test_data/digits_train_2023-08-01-14-45-08_adj_False_meta_True_with_motifs'
test_path = f'{task}_train_test_data/digits_test_2023-08-01-14-45-08_adj_False_meta_True_with_motifs'

modularity_idx = retina_structural_features_full_name_vec.index('modularity')
for data_path in [train_path, test_path]:
    with open(f'{base_path}/{data_path}.pkl', 'rb') as fp:
        data = joblib.load(fp)
    data_no_modularity = [
        (torch.cat([sample[0:modularity_idx], sample[modularity_idx+1:]]), performance)
        for sample, performance in data
    ]
    with open(f'{base_path}/{data_path}_no_modularity.pkl', 'wb+') as fp:
        joblib.dump(data_no_modularity, fp)

