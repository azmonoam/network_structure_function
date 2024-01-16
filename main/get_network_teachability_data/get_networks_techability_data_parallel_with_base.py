from typing import Any, Dict, Optional

import pandas as pd

from get_network_teachability_data.get_networks_techability_data_parallel import GetNetworkTeachabilityData


class GetNetworkTeachabilityDataWithBase(GetNetworkTeachabilityData):
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
            base_file_path: str,
            ephoc_folder: Optional[str] = None,
    ):
        super().__init__(num_cores, folder, models_folder, res_folder, extended_results, add_motifs, new_models, add_normlized
                         )
        self.base_first_analsis = pd.read_csv(base_file_path).drop("Unnamed: 0", axis=1, errors='ignore')
        if ephoc_folder is not None:
            self.csv_name += ephoc_folder

    def _build_tabels_rows(
            self,
            csv_name: str,
            epoch: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        try:
            results = pd.read_csv(f"{self.folder}/{csv_name}").drop("Unnamed: 0", axis=1, errors='ignore')
        except:
            print(f"{self.folder}/{csv_name}")
            return
        if results.shape[0] <= 1:
            return
        exp_name = results['exp_name'].iloc[0]
        base_first_analsis_row = self.base_first_analsis[self.base_first_analsis['exp_name'] == exp_name]
        first_analysis = base_first_analsis_row.to_dict(orient='records')[0]
        if epoch is None:
            epoch = results['iterations'].max()
        final_epoch_res = results[results['iterations'] == epoch]

        if final_epoch_res.shape[0] > 0:
            first_analysis = self._get_ephoc_data_from_df(
                first_analysis=first_analysis,
                final_epoch_res=final_epoch_res,
            )
            return first_analysis
        return None
