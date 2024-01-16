import random
from typing import Tuple, Any, Dict, List, Optional, Union

import pandas as pd
from joblib import Parallel, delayed, load

from get_network_teachability_data.get_networks_techability_data_parallel import GetNetworkTeachabilityData


class GetNetworkTeachabilityAllFeatures(GetNetworkTeachabilityData):
    def __init__(
            self,
            num_cores: int,
            folder: str,
            models_folder: str,
            res_folder: str,
            extended_results: bool,
            add_motifs: bool,
            new_models: bool,
            add_normlized: bool,
    ):
        super().__init__(num_cores, folder, models_folder, res_folder, extended_results, add_motifs, new_models,
                         add_normlized
                         )

    def _build_tabels_rows(
            self,
            csv_name: str,
            epochs: List[int],
    ) -> Optional[Dict[str, Any]]:
        try:
            results = pd.read_csv(f"{self.folder}/{csv_name}").drop("Unnamed: 0", axis=1, errors='ignore')
        except:
            print(f"{self.folder}/{csv_name}")
            return
        if results.shape[0] <= 1:
            return
        exp_name = results['exp_name'].iloc[0]
        with open(f"{self.models_folder}/{exp_name}.pkl", 'rb') as fp:
            organism = load(fp)
        model_data = organism.structural_features.get_features(
            layer_neuron_idx_mapping=organism.layer_neuron_idx_mapping,
        )
        epoch = results['iterations'].max()
        final_epoch_res = results[results['iterations'] == epoch]
        if final_epoch_res.shape[0] > 0:
            first_analysis = self._get_ephoc_data_from_df(
                first_analysis=model_data,
                final_epoch_res=final_epoch_res,
            )
            return first_analysis

    def _get_subset_columns_to_drop_dop(self):
        self.subset = [
            'modularity',
            'num_connections',
            'entropy',
            'normed_entropy',
            'connectivity_ratio',
            'num_neurons',
            'motifs_count_0',
            'motifs_count_1',
            'motifs_count_2',
        ]
