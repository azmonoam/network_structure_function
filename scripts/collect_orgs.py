import multiprocessing
import os
from typing import List, Tuple, Optional

import pandas as pd
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()


def _find_exp(
        exp_name: str,
):
    model_path = None
    res_path = None
    for folder in POSSIBLE_MODELS_FOLDERS:
        if os.path.exists(f"{base_path}/{folder}/{exp_name}.pkl"):
            model_path = f"{base_path}/{folder}/{exp_name}.pkl"
            break
    for folder in POSSIBLE_RES_FOLDERS:
        if os.path.exists(f"{base_path}/{folder}/{exp_name}_teach.csv"):
            res_path = f"{base_path}/{folder}/{exp_name}_teach.csv"
            break
    return (exp_name, model_path, res_path)


def _get_exp_paths() -> List[Tuple[str, Optional[str], Optional[str]]]:
    return Parallel(n_jobs=num_cores)(
        delayed
        (_find_exp)(exp_name)
        for exp_name in exp_names
    )


def create_paths_tables():
    found_exps = []
    missing_exps = []
    for exp_name, model_path, res_path in _get_exp_paths():
        if model_path is None or res_path is None:
            missing_exps.append((exp_name, model_path, res_path))
        else:
            found_exps.append((model_path, res_path))
    pd.DataFrame(found_exps, columns=('model_path', 'res_path')).to_csv(f"{base_path}/found_exps.csv")
    pd.DataFrame(missing_exps, columns=('exp_name', 'model_path', 'res_path')).to_csv(f"{base_path}/missing_exps.csv")


if __name__ == '__main__':
    base_path = '/home/labs/schneidmann/noamaz/modularity'

    POSSIBLE_MODELS_FOLDERS = [
        "random_pick_teach_arch_models",
        "teach_arch_5_models",
        "teach_arch_models",
        "successful_results_5_3",
        "successful_results_5",
        "successful_results_4",
        "successful_results_3",
    ]
    POSSIBLE_RES_FOLDERS = [
        "random_pick_teach_arch",
        "retrain_good_arc_results",
        "teach_arch_5_3",
        "teach_arch_5_2",
        "teach_arch_5",
        "teach_arch_4_2",
        "teach_arch_4",
        "teach_arch_3_dynemic",
        "teach_arch_3",
        "teach_arch_2",
        "teach_arch",
    ]
    exp_names = pd.read_csv(f'{base_path}/2023-02-19_all_first_analysis_results.csv')['exp_name'].to_list()
    create_paths_tables()
