import argparse

from find_feature_spectrum.find_features_spectrum_by_value_range import FindFeaturesSpectrumByValueRange
from jobs_params import retina_find_features_spectrom, xor_find_features_spectrom
from utils.tasks_params import XorParameters, RetinaParameters

if __name__ == '__main__':
    base_path = '/Volumes/noamaz/modularity'

    num_exp_per_arch=300

    label = "mean performance"
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=1)
    parser.add_argument('--task', default='retina')

    args = parser.parse_args()
    job_num = int(args.job_num)
    task_name = args.task
    results_path = f"{base_path}/teach_archs/{task_name}"

    if task_name == 'xors':
        task_params = XorParameters
        job_params_dict = xor_find_features_spectrom

    elif task_name == 'retina':
        task_params = RetinaParameters
        job_params_dict = retina_find_features_spectrom

    num_features = job_params_dict['num_features']
    train_path = job_params_dict['train_path']
    test_path =job_params_dict['test_path']
    used_features_csv = job_params_dict['used_features_csv']
    results_folder = job_params_dict['results_folder']

    find_features_spectrum_by_value_range = FindFeaturesSpectrumByValueRange(
        task_params=task_params,
        num_features=num_features,
        train_path=train_path,
        test_path=test_path,
        used_features_csv_name=used_features_csv,
        results_folder=results_folder,
        base_path=base_path,
    )
    find_features_spectrum_by_value_range.find_arch_by_features_train_and_compere(
        num_exp_per_arch=num_exp_per_arch,
    )

