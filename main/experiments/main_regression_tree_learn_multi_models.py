from datetime import datetime as dt

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from networks_teachability_regression.regression_tree_learn import tree_regression
from parameters.digits_parameters import (
    digits_structural_features_name_vec,
    DIGITS_EDGE_MAPPING
)
from parameters.retina_parameters import (
    retina_structural_features_name_vec,
    RETINA_EDGE_MAPPING,
)
from parameters.xor_parameters import (
    xor_structural_features_name_vec,
    XOR_EDGE_MAPPING,
)

COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f']
task = 'digits'
if task =='xor':
    EDGE_MAPPING = XOR_EDGE_MAPPING
    METRIC_MAPPING = xor_structural_features_name_vec
    task_data_path = 'teach_archs/xors/xor_train_test_data'
    out_folder = "xor_regression_tree"
    train_paths = [
        'xor_train_2023-04-13-14-15-49_adj_True_meta_True.pkl',
        'xor_train_2023-04-13-14-15-54_adj_True_meta_False.pkl',
        'xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl',
    ]
    test_paths = [
        'xor_test_2023-04-13-14-15-49_adj_True_meta_True.pkl',
        'xor_test_2023-04-13-14-15-54_adj_True_meta_False.pkl',
        'xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl',

    ]
elif task == 'retina':
    EDGE_MAPPING = RETINA_EDGE_MAPPING
    METRIC_MAPPING = retina_structural_features_name_vec
    task_data_path = 'teach_archs/retina/retina_train_test_data"'
    out_folder = "retina_regression_tree"
    train_paths = [
        'retina_train_2023-04-16-15-02-58_adj_True_meta_True.pkl',
        'retina_train_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl',
    ]
    test_paths = [
        'retina_test_2023-04-16-15-02-58_adj_True_meta_True.pkl',
        'retina_test_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl',
    ]

elif task =='digits':
    EDGE_MAPPING = DIGITS_EDGE_MAPPING
    METRIC_MAPPING = digits_structural_features_name_vec
    task_data_path = 'teach_archs/digits/digits_train_test_data'
    out_folder = "digits_regression_tree"
    train_paths = [
        'digits_train_2023-06-26-15-36-02_adj_True_meta_True.pkl',
        'digits_train_2023-06-26-14-53-44_adj_True_meta_False.pkl',
        'digits_train_2023-06-26-14-00-11_adj_False_meta_True.pkl',
    ]
    test_paths = [
        'digits_test_2023-06-26-15-36-02_adj_True_meta_True.pkl',
        'digits_test_2023-06-26-14-53-44_adj_True_meta_False.pkl',
        'digits_test_2023-06-26-14-00-11_adj_False_meta_True.pkl',
    ]


if __name__ == '__main__':
    local_base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_path = '/Volumes/noamaz/modularity'
    # base_path = '/home/labs/schneidmann/noamaz/modularity'
    base_path_to_res = f"{base_path}/{task_data_path}"
    ALL_MAPPING = EDGE_MAPPING + METRIC_MAPPING
    data_types = [
        'full input',
        'adjacency matrix input',
        'structure metrics input',
    ]
    input_names = ['Edge or Metric', 'Edge', 'Metric', ]
    num_features = [20, 20, 20, ]
    mappings = [ALL_MAPPING, EDGE_MAPPING, METRIC_MAPPING]
    res = []
    train_r2s = []
    test_r2s = []
    for train_path, test_path, data_type, mapping, input_name, num_feature in zip(train_paths, test_paths,
                                                                                  data_types,
                                                                                  mappings, input_names,
                                                                                  num_features,
                                                                                  ):
        model, train_r2, test_r2 = tree_regression(
            train_path=f'{base_path_to_res}/{train_path}',
            test_path=f'{base_path_to_res}/{test_path}',
        )
        lightgbm.plot_metric(model, title=f'{task} mean squared error during training {data_type}',
                             dataset_names=['train', 'test'],
                             ylabel='Mean squared error')
        plt.savefig(
            f"{local_base_path}/plots/{out_folder}/{time_str}_{task}_mean_squared_error_lightgbm_training_{data_type.replace(' ', '_')}.png")
        plt.show()

        feature_importance = pd.DataFrame(
            sorted(zip(model.feature_importances_, mapping), reverse=True),
            columns=['Value', input_name],
        )
        plt.figure()
        sns.barplot(x="Value", y=input_name, data=feature_importance.iloc[:num_feature],
                    palette=sns.color_palette("flare", n_colors=num_feature)
                    )
        plt.title(f'{task} - LightGBM Features\n{data_type}')
        plt.tight_layout()
        plt.savefig(
            f"{local_base_path}/plots/{out_folder}/{time_str}_{task}_feature_importance_top_20_lightgbm_training_{data_type.replace(' ', '_')}.png")
        plt.show()
        res.append(model.evals_result_)
        train_r2s.append(train_r2)
        test_r2s.append(test_r2)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6))
    bar_size = 0.15
    padding = 0.15
    y_locs = np.arange(2 * (bar_size * 3 + padding))
    for i, data_type in enumerate(data_types):
        ax1.plot(list(range(len(res[i]['test']['l2'])))[-30:], res[i]['test']['l2'][-30:], label=data_type, c=COLORS[i])
        ax2.plot(list(range(len(res[i]['train']['l2'])))[-30:], res[i]['train']['l2'][-30:], label=data_type,
                 c=COLORS[i])
        ax3.barh(y_locs + (i * bar_size), [train_r2s[i], test_r2s[i]], height=bar_size, color=COLORS[i],
                 label=data_types[i])

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE loss- Test')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE loss - Train')
    ax3.set(
        yticks=y_locs + bar_size,
        yticklabels=['Train', 'Test'],
        ylim=[0 - padding, len(y_locs) - padding],
        xlim=[max(max(train_r2s), max(test_r2s)) - 0.09, max(max(train_r2s), max(test_r2s)) + 0.03],
    )
    ax3.set_xlabel('r2 score')

    ax1.set_title(f'{task} - Prediction of teachability mean performance')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.savefig(f"{local_base_path}/plots/{out_folder}/{time_str}_{task}_data_type_comparison_lightgbm_regression.png")
    plt.show()
