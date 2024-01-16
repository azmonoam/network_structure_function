import random
from typing import Tuple, Any, Dict, List, Optional, Union

import pandas as pd
from joblib import Parallel, delayed, load

from get_network_teachability_data.get_networks_techability_data_parallel import GetNetworkTeachabilityData


class GetNetworkTeachabilityDataMultiEph(GetNetworkTeachabilityData):
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
        super().__init__(num_cores, folder, models_folder, res_folder, extended_results, add_motifs, new_models,add_normlized
                         )

    def _build_multiple_tabels_rows(
            self,
            csv_name: str,
            epochs: List[int],
    ) -> Optional[List[Dict[str, Any]]]:
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
        model_data = self._get_model_basic_data(
            organism=organism,
            exp_name=exp_name,
        )
        model_data = self._add_additional_data(
            organism=organism,
            first_analysis=model_data,
        )
        multi_eph_analsis = []
        for epoch in epochs:
            final_epoch_res = results[results['iterations'] == epoch]
            if final_epoch_res.shape[0] > 0:
                first_analysis = self._get_ephoc_data_from_df(
                    first_analysis=model_data,
                    final_epoch_res=final_epoch_res,
                )
                multi_eph_analsis.append(first_analysis)
        return multi_eph_analsis

    def _prepare_data_from_csv_wrapper(
            self,
            epoch: Union[Optional[int], List[int]] = None,
    ) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
        print(f"-- using {self.folder} as folder name --")
        csvs = self._get_list_of_files_to_run()
        if epoch is not None:
            self._test_one_file(
                csv_name=random.choice(csvs),
                epoch=epoch,
            )
        return Parallel(
            n_jobs=self.num_cores,
            timeout=9999,
        )(
            delayed
            (self._build_multiple_tabels_rows)(csv_name, epoch)
            for csv_name in csvs
        )

    def combine_all_data(
            self,
            existing_exps_analysis_full_path: Optional[str],
            epoch: Union[Optional[int], List[int]] = None,
    ):
        first_analysis_dfs = []
        self.existing_exps = None
        data_list = self._prepare_data_from_csv_wrapper(epoch)
        for i in range(len(epoch)):
            first_analysis_df = pd.DataFrame()
            for first_analysises_list in data_list:
                if first_analysises_list is None or (len(first_analysises_list) - 1) < i:
                    continue
                first_analysis_df = pd.concat([first_analysis_df, pd.DataFrame(first_analysises_list[i], index=[0], )],
                                              ignore_index=True)
            first_analysis_dfs.append(first_analysis_df)
        return first_analysis_dfs
