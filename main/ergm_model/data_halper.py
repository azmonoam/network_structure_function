from typing import List, Tuple, Dict, Any

from find_feature_spectrum.find_feature_dist_by_performance import FindFeaturesDist
from find_feature_spectrum.find_feature_dist_utils import get_selected_feature_names, get_features_of_samples


def get_list_of_functions_from_features_names(
        method_class,
        features_names_list: List[str],
):
    feature_functions = []
    for feature_name in features_names_list:
        for optional_feature_name, feature_function in method_class.method_features_mapping.items():
            if optional_feature_name in ['connectivity_ratio', 'entropy'] and feature_name != optional_feature_name:
                continue
            if optional_feature_name in feature_name:
                feature_functions.append(feature_function)
                break
    #  raise ValueError('Could not find feature function')
    return feature_functions


def get_dist_lookup_data(
        params_dict: Dict[str, Any],
        task: str,
        num_features: int,
        base_path: str,
        normalize: bool = True,
) -> Tuple[List[str], FindFeaturesDist]:
    task_path = f'{base_path}/teach_archs/{task}'
    used_features_csv_name = f'{task_path}/{params_dict[task].get("used_features_csv_name")}'
    target_label_ranges = params_dict[task].get("target_label_ranges")
    samples_path = params_dict[task].get(num_features, {}).get('samples_path')
    sample_path_name_addition = params_dict[task].get('sample_path_name_addition')
    min_range_ind, max_range_ind = params_dict[task].get('performance_range_ind')
    if not samples_path:
        samples_path = f'{task_path}/{task}_train_test_data/all_data_{num_features}_features_with_{sample_path_name_addition}.pkl'
        get_features_of_samples(
            task_path=task_path,
            train_path=params_dict[task].get('train_path'),
            test_path=params_dict[task].get('test_path'),
            used_features_csv_name=used_features_csv_name,
            num_features=num_features,
            target_samples_path=samples_path,
        )
    else:
        samples_path = f'{task_path}/{samples_path}'
    print(samples_path)
    selected_feature_names = get_selected_feature_names(
        used_features_csv_name=used_features_csv_name,
        num_features=num_features
    ).to_list()
    find_feature_dist = FindFeaturesDist(
        num_features=num_features,
        samples_path=samples_path,
        target_label_ranges=target_label_ranges,
        min_range_ind=min_range_ind,
        max_range_ind=max_range_ind,
    )
    if normalize:
        find_feature_dist.target_mean_features = find_feature_dist.normalize_generated_sample(
            find_feature_dist.target_mean_features, selected_feature_names)
    return selected_feature_names, find_feature_dist
