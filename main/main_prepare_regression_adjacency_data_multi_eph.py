import argparse

import joblib

from jobs_params import main_prepere_techability_params_retina_multi_arch_multi_ep
from prepare_regression_adjacency_data_multi_label import PrepareRegressionAdjacencyDataMultiLabel

if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=2)
    parser.add_argument('--n_threads', default=1)
    parser.add_argument('--task', default='retina')
    parser.add_argument('--folder_path', default='0')

    args = parser.parse_args()
    job_num = int(args.job_num)
    n_threads = int(args.n_threads)
    folder_path = args.folder_path
    task = args.task
    if task == 'retina':
        main_predict_techability_params = main_prepere_techability_params_retina_multi_arch_multi_ep
        if folder_path == '0':
            folder_path = main_predict_techability_params['folder_path']
    else:
        raise ValueError()

    consider_meta_data = main_predict_techability_params[job_num]['consider_meta_data']
    normalize_features = main_predict_techability_params[job_num]['normalize_features']
    features_list = main_predict_techability_params[job_num]['features_list']
    pkl_name_addition = main_predict_techability_params[job_num].get('pkl_name_addition', '')
    consider_adj_mat = main_predict_techability_params[job_num]['consider_adj_mat']

    out_path_folder = f"{base_path}/{folder_path}/{main_predict_techability_params['train_test_folder']}"
    results_csvs_dir = f"{folder_path}/{main_predict_techability_params['results_csvs_dir']}"
    results_model_path = f"{folder_path}/{main_predict_techability_params['results_model_path']}"

    label_name = "mean_performance"
    increase_label_scale = True

    prepare_adjacency_data = PrepareRegressionAdjacencyDataMultiLabel(
        base_path=base_path,
        results_csvs_dir=results_csvs_dir,
        results_model_path=results_model_path,
        label_name=label_name,
        increase_label_scale=increase_label_scale,
        consider_meta_data=consider_meta_data,
        consider_adj_mat=consider_adj_mat,
        normalize_features=normalize_features,
        n_threads=n_threads,
    )
    train_test_data = prepare_adjacency_data.create_test_and_train_data(
        train_percent=0.9,
        features_list=features_list,
    )

    for epochs, epoch_train_test_data in train_test_data.items():
        for train_or_test in ['train', 'test']:
            data = epoch_train_test_data[train_or_test]
            file_name = f'{task}_{train_or_test}_{prepare_adjacency_data.time_str}_adj_{consider_adj_mat}_meta_{consider_meta_data}{pkl_name_addition}_{epochs}_ep.pkl'
            with open(f"{out_path_folder}/{file_name}", 'wb+') as fp:
                joblib.dump(data, fp)

