import joblib

import random
base_path = '/Volumes/noamaz/modularity/teach_archs/retina'
five_features =f'retina_lightgbm_feature_selection/no_modularity/exp_2023-07-15-13-12-18/masked_data_models/2023-07-15-13-12-18_masked_data_5_features'

with open(f'{base_path}/{five_features}.pkl', 'rb') as fp:
    data = joblib.load(fp)
all_data = data['selected_train_data'] +  data['selected_test_data']
random.shuffle(all_data)
num_features =  data['num_features']
target_samples_path = f'{base_path}/retina_train_test_data/all_data_{num_features}_features_with_preformance_no_modularity.pkl'

with open(target_samples_path, 'wb+') as fp:
    joblib.dump(all_data, fp)
print('a')