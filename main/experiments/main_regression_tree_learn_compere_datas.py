from datetime import datetime as dt
from typing import List

import lightgbm
from sklearn import metrics

from networks_teachability_regression.regression_tree_learn import _get_data
from parameters.retina.dynamic_retina import DynamicRetinaParams
from stractural_features_models.calc_structural_features import CalcStructuralFeatures
from utils.set_up_population_utils import get_organism_by_connectivity_ratio

COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']
task = 'retina'
num_layers = 3

if task == 'retina':
    task_data_path = f'retina/retina_3_layers_3_4/'
    out_folder = "lightgbm_regression_tree"

if __name__ == '__main__':
    local_base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
    local_out_folder = f'{local_base_path}/plots/retina_multi_archs/{num_layers}_layer/{out_folder}'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = '/Volumes/noamaz/modularity'
    # base_path = '/home/labs/schneidmann/noamaz/modularity'
    base_path_to_res = f"{base_path}/{task_data_path}/train_test_data"
    num_feature = 20

    paths = [
        ('retina_train_2023-08-20-13-08-40_adj_False_meta_True_const_features_normed.pkl',
         'retina_test_2023-08-20-13-08-40_adj_False_meta_True_const_features_normed.pkl',
         ),
        ('retina_train_2023-08-20-13-08-40_adj_False_meta_True_const_features.pkl',
         'retina_test_2023-08-20-13-08-40_adj_False_meta_True_const_features.pkl'),

        ('retina_train_2023-08-20-13-51-12_adj_True_meta_True.pkl',
         'retina_test_2023-08-20-13-51-12_adj_True_meta_True.pkl'),
        ('retina_train_2023-08-20-13-51-03_adj_False_meta_True.pkl',
         'retina_test_2023-08-20-13-51-03_adj_False_meta_True.pkl',),
    ]
    train_test_data = []
    for train_path, test_path in paths:
        train_test_data.append(_get_data(
            train_path=f'{base_path_to_res}/{train_path}',
            test_path=f'{base_path_to_res}/{test_path}',
            train_data=None,
            test_data=None,
            ind_to_drop=None,
        )
        )
    n_estimators = 10000
    learning_rate = 0.001
    subsample_for_bin = 2000
    names = [
        'const features normed',
        'const features',
        'adj and features',
        'features'
    ]
    res = {}
    for (test_inputs, test_labels, train_inputs, train_labels), name in zip(train_test_data, names):
        model = lightgbm.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample_for_bin=subsample_for_bin,
        )
        model.fit(
            train_inputs,
            train_labels,
            eval_metric='mean_squared_error',
            eval_set=[(test_inputs, test_labels), (train_inputs, train_labels)],
            eval_names=['test', 'train'],
        )
        model.score(test_inputs, test_labels)
        predicted_train_inputs = model.predict(train_inputs)
        predicted_test_inputs = model.predict(test_inputs)
        train_r2 = metrics.r2_score(predicted_train_inputs, train_labels)
        test_r2 = metrics.r2_score(predicted_test_inputs, test_labels)
        print('{}: Training r2 score {:.4f}'.format(name, train_r2))
        print('{}: Testing r2 score {:.4f}'.format(name, test_r2))
        res[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            "mse_loss_train": model.evals_result_['train']['l2'],
            "mse_loss_test": model.evals_result_['test']['l2'],
        }
    print('a')
