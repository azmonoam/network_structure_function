import itertools
from datetime import datetime as dt
from typing import Optional, List, Tuple

import joblib
import networkx as nx
import numpy as np

from find_arch_from_features_genrtic import FindArchGenetic
from find_feature_spectrum.find_feature_dist import FindFeaturesDist
from find_feature_spectrum.find_feature_dist_utils import get_selected_feature_names
from logical_gates import LogicalGates
from parameters.retina.retina_by_dim import RetinaByDim
from parameters.xor.xor_by_dim import XoraByDim


class SampleGaussian:
    def __init__(
            self,
            num_features: int,
            samples_path: str,
            num_samples: int,
            connectivity_pkl_path: Optional[str] = None,
    ):
        self.find_feature_dist = FindFeaturesDist(
            num_features=num_features,
            samples_path=samples_path,
        )
        self.dist = self.find_feature_dist._get_gaussian()
        self.num_samples = num_samples
        self._init_connectivity_df(
            connectivity_pkl_path=connectivity_pkl_path
        )

    def _init_connectivity_df(
            self,
            connectivity_pkl_path: Optional[str],
    ):
        if connectivity_pkl_path is None:
            self.connectivities_df = None
        else:
            with open(connectivity_pkl_path, 'rb+') as fp:
                self.connectivities_df = joblib.load(fp)

    def get_errors(
            self,
            num_features: int,
            frec: float = 0.1,
    ) -> np.ndarray:
        means = np.zeros((100, num_features))
        for i in range(100):
            means[i] = np.mean(
                self.find_feature_dist.samples_to_model.sample(frac=frec, axis=0), axis=0)
        return np.std(means, axis=0)

    def get_samples_form_sorted_arrey(self) -> Tuple[List[List[float]], Optional[List[List[float]]]]:
        std = self.find_feature_dist.samples_to_model.std()
        sorted_df = self.find_feature_dist.samples_to_model.sort_values(std.sort_values().index.to_list())
        jump = (sorted_df.shape[0] - 1) // (num_samples - 1)
        samples = [
            sorted_df.iloc[i * jump].to_list()
            for i in range(self.num_samples)
        ]
        if self.connectivities_df is not None:
            sorted_connectivities_df = self.connectivities_df.reindex(sorted_df.index)
            connectivities = [
                sorted_connectivities_df.iloc[i * jump]
                for i in range(self.num_samples)
            ]
            return samples, connectivities
        return samples, None

    def get_samples_from_quentile(
            self,
            use_only_middle: bool = False,
    ) -> List[List[float]]:
        if use_only_middle:
            quantiles = np.linspace(0, 1, num_samples + 2, endpoint=False)[1:-1]
        else:
            quantiles = np.linspace(0, 1, num_samples, endpoint=False)
        return [
            self.find_feature_dist.samples_to_model.quantile(q).to_list()
            for q in quantiles
        ]

    def get_sample_by_ec_dist(
            self,
            dist_stds,
    ):
        mean_distance = np.mean([
            np.sqrt(sum(
                np.divide(
                    abs(
                        self.find_feature_dist.samples_to_model.sample().to_numpy().flatten() -
                        self.find_feature_dist.samples_to_model.sample().to_numpy().flatten()
                    ),
                    dist_stds
                )
            ))
            for _ in range(200)
        ]
        )
        i = 0
        while i < 1000:
            s = [
                self.find_feature_dist.samples_to_model.sample().to_numpy().flatten()
                for _ in range(self.num_samples)
            ]
            last_good = True
            for a, b in itertools.combinations(s, 2):
                if last_good:
                    dis = np.sqrt(sum(np.divide(abs(a - b), dist_stds)))
                    if dis < mean_distance:
                        last_good = False
            if last_good:
                return s
            i += 1
        raise ValueError("could not found far enough samples")


if __name__ == '__main__':
    all_features = False
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    task = 'xor'
    num_features = 4
    num_samples = 1
    base_path = "/Volumes/noamaz/modularity"
    train_test_folder_name = 'train_test_data'

    folder_name = f'{num_features}_features'
    if task == 'retina':
        folder_path = "retina_xor/retina_3_layers_3_4"
        train_test_dir = f"{base_path}/{folder_path}/{train_test_folder_name}"
        lgb_dir = f"{base_path}/{folder_path}/lightgbm_feature_selection"
        if num_features == 10:
            exp_name = 'exp_2023-09-09-13-55-25'
            connectivity_pkl_path = f"{train_test_dir}/retina_xor_2023-09-09-13-55-25_all_train_test_connectivity_data"
            samples_path = f"{train_test_dir}/retina_xor_2023-09-09-13-55-25_all_train_test_masked_data_10_features.pkl"
            used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-09-13-55-25_1_75_used_features.csv"
        elif num_features == 6:
            if all_features:
                exp_name = 'exp_2023-09-11-09-59-56'
                connectivity_pkl_path = f"{train_test_dir}/retina_xor_2023-09-11-09-59-56_all_train_test_connectivity_data"
                samples_path = f"{train_test_dir}/retina_xor_2023-09-11-09-59-56_all_train_test_masked_data_6_features.pkl"
                used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-11-09-59-56_1_70_used_features.csv"
                folder_name = '6_features_all_features'
            else:
                exp_name = 'exp_2023-09-12-20-38-43_nice_features'
                connectivity_pkl_path = f"{train_test_dir}/retina_xor_2023-09-12-20-38-43_all_train_test_connectivity_data_nice_features.pkl"
                samples_path = f"{train_test_dir}/retina_xor_2023-09-12-20-38-43_all_train_test_masked_data_6_features_nice_features.pkl"
                used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-12-20-38-43_1_60_used_features.csv"
                folder_name = '6_features_nice'
        elif num_features == 3:
            exp_name = 'exp_2023-09-14-16-31-14_nice_features'
            samples_path = f"{train_test_dir}/retina_xor_all_train_test_masked_data_3_top_features.pkl"
            used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-14-16-31-14_3_features_used_features.csv"
            folder_name = '3_top_features'
        dims = [6, 3, 4, 2]
        num_layers = len(dims) - 1
        task_params = RetinaByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
            task_base_folder_name='retina_xor',
            rule=LogicalGates.XOR,
        )
    elif task == 'xor':
        folder_path = "xor/xor_4_layers_6_5_3"
        train_test_dir = f"{base_path}/{folder_path}/{train_test_folder_name}"
        exp_name = 'exp_2023-09-16-13-35-58_nice_features'
        if num_features == 4:
            lgb_dir = f"{base_path}/{folder_path}/lightgbm_feature_selection"
            samples_path = f"{train_test_dir}/2023-09-16-13-35-58_all_data_4_features_nica_features.pkl"
            used_features_csv_path = f"{lgb_dir}/{exp_name}/2023-09-16-13-35-58_1_70_used_features.csv"
        dims = [6, 6, 5, 3, 2]
        num_layers = len(dims) - 1
        task_params = XoraByDim(
            start_dimensions=dims,
            num_layers=num_layers,
            by_epochs=False,
        )
    else:
        raise ValueError

    potential_parents_percent = 15
    population_size = 500
    generations = 1000
    use_distance_fitness = False
    mse_early_stopping_criteria_factor = 0.005
    sampler = SampleGaussian(
        num_features=num_features,
        samples_path=samples_path,
        num_samples=num_samples,
    )
    selected_feature_names = get_selected_feature_names(
        used_features_csv_name=used_features_csv_path,
        num_features=num_features
    ).to_list()

    dist_stds = sampler.get_errors(
        num_features=num_features,
        frec=0.1
    )
    # target_values = sampler.get_sample_by_ec_dist(
    #   dist_stds=dist_stds
    # )
    target_values = [sampler.find_feature_dist.samples_to_model.mean()]
    for i, sample in enumerate(target_values):
        sample = sample.tolist()
        print(f"target_value: {sample}")
        find_arches_genetic = FindArchGenetic(
            generations=generations,
            task_params=task_params,
            population_size=population_size,
            potential_parents_percent=potential_parents_percent,
            selected_feature_names=selected_feature_names,
            target_feature_values=sample,
            use_distance_fitness=False,
            dist_stds=dist_stds * 2,
        )
        orgs_to_save = find_arches_genetic._get_x_archs_with_features_genetic(
            num_orgs_to_return=500,
            distance_early_stopping_criteria_num_sigmas=0.5
        )
        orgs_to_save_rel = [o for o in orgs_to_save if len(list(nx.isolates(o.network))) == 0]
        graphs = [
            nx.to_numpy_array(g.network, weight=None, dtype=int)
            for g in orgs_to_save_rel
        ]
        data_to_save = {
            'graphs': graphs,
            'orgs': orgs_to_save_rel,
            'target_feature_values': sample,
            'selected_feature_names': selected_feature_names,
            'errors': dist_stds,
        }
        with open(
                f"{base_path}/{folder_path}/requiered_features_genetic_models/{folder_name}/{time_str}_data_for_{num_features}_features_{i}.pkl",
                'wb+') as fp:
            joblib.dump(data_to_save, fp)
        print(f'saved {len(orgs_to_save)} orgs')
