import argparse

import joblib

from jobs_params import main_prepere_techability_params_retina_multi_arch, \
    main_prepere_techability_params_retina_xor, main_prepere_techability_params_xor_new, \
    main_prepere_techability_params_digits_new
from prepare_regression_adjacency_data import PrepareRegressionAdjacencyData

if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    # base_path = '/Volumes/noamaz/modularity'
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=2)
    parser.add_argument('--n_threads', default=1)
    parser.add_argument('--task', default='retina')

    args = parser.parse_args()
    job_num = int(args.job_num)
    n_threads = int(args.n_threads)
    task = args.task
    if task == 'retina':
        main_predict_techability_params = main_prepere_techability_params_retina_multi_arch
    elif task == 'retina_xor':
        main_predict_techability_params = main_prepere_techability_params_retina_xor
    elif task == 'digits':
        main_predict_techability_params = main_prepere_techability_params_digits_new
    elif task == 'xor':
        main_predict_techability_params = main_prepere_techability_params_xor_new
    else:
        raise ValueError()
    folder_path = main_predict_techability_params['folder_path']

    consider_meta_data = main_predict_techability_params[job_num]['consider_meta_data']
    normalize_features = main_predict_techability_params[job_num]['normalize_features']
    features_list = main_predict_techability_params[job_num]['features_list']
    pkl_name_addition = main_predict_techability_params[job_num].get('pkl_name_addition', '')
    consider_adj_mat = main_predict_techability_params[job_num]['consider_adj_mat']
    train_test_folder = main_predict_techability_params['train_test_folder']

    results_csv = f"{folder_path}/{main_predict_techability_params['results_csv']}"
    results_model_path = f"{folder_path}/{main_predict_techability_params['results_model_path']}"

    label_name = "mean_performance"
    increase_label_scale = True

    prepare_adjacency_data = PrepareRegressionAdjacencyData(
        base_path=base_path,
        results_csv_name=results_csv,
        results_model_path=results_model_path,
        label_name=label_name,
        increase_label_scale=increase_label_scale,
        consider_meta_data=consider_meta_data,
        consider_adj_mat=consider_adj_mat,
        normalize_features=normalize_features,
        n_threads=n_threads,
    )
    train_data, test_data = prepare_adjacency_data.create_test_and_train_data(
        train_percent=0.9,
        features_list=features_list,
    )
    with open(
            f'{base_path}/{folder_path}/{train_test_folder}/{task}_train_{prepare_adjacency_data.time_str}_adj_{consider_adj_mat}_meta_{consider_meta_data}{pkl_name_addition}.pkl',
            'wb+') as fp:
        joblib.dump(train_data, fp)
    with open(
            f'{base_path}/{folder_path}/{train_test_folder}/{task}_test_{prepare_adjacency_data.time_str}_adj_{consider_adj_mat}_meta_{consider_meta_data}{pkl_name_addition}.pkl',
            'wb+') as fp:
        joblib.dump(test_data, fp)
