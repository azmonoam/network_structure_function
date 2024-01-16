from pydantic import BaseModel
from typing import Optional, List


class SelectedExpNames(BaseModel):
    feature_selection_folder: Optional[str] = None
    first_analsis_csv: Optional[str] = None
    source_folder: Optional[str] = None
    num_selected_features: Optional[int] = None
    selected_features_data: Optional[str] = None
    used_features_csv: Optional[str] = None
    dims: Optional[List[int]] = None
    num_layers: Optional[int] = None
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    correlated_features_csv_name: Optional[str] = None
    ergm_init_graphs_pkl_file_name: Optional[str] = None
    ergm_res_pkl_file_name_random_init: Optional[str] = None
    gen_graphs_pkl_file_name: Optional[str] = None
    ergm_res_pkl_file_name: Optional[str] = None
    num_neurons: Optional[int] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None


selected_exp_names = {
    'retina_xor':
        {
            (3, 4): SelectedExpNames(
                feature_selection_folder='exp_2023-09-12-20-38-43_nice_features',
                first_analsis_csv="2023-09-04-10-45-12_all_results_from_teach_archs_results_with_motifs_10000_ep.csv",
                source_folder="retina_3_layers_3_4",
                num_selected_features=6,
                selected_features_data="retina_xor_2023-09-12-20-38-43_all_train_test_masked_data_6_features_nice_features.pkl",
                used_features_csv='2023-09-12-20-38-43_1_60_used_features.csv',
                dims=[6, 3, 4, 2]
            ),
            (5, 2): SelectedExpNames(
                feature_selection_folder='exp_2023-10-12-10-46-16_nice_features',
                first_analsis_csv="2023-09-07-09-30-11_all_results_from_teach_archs_results_with_motifs_10000_ep.csv",
                source_folder="retina_3_layers_5_2",
                num_selected_features=6,
                selected_features_data='retina_xor_2023-10-12-10-46-16_all_train_test_masked_data_6_features.pkl',
                used_features_csv='2023-10-12-10-46-16_1_60_used_features.csv',
                dims=[6, 5, 2, 2],
            ),
            'random_2': SelectedExpNames(
                dims=None,
                num_layers=3,
                num_neurons=15,
                input_size=6,
                output_size=2,
                source_folder="retina_3_layers",
                feature_selection_folder='exp_2023-11-06-12-00-00_nice_features',
                first_analsis_csv=f"2023-10-24-10-00-54_all_results_from_teach_archs_results_with_motifs_5000_ep_no_duplicates.csv",
                num_selected_features=6,
                selected_features_data="retina_xor_2023-11-06-12-00-00_all_train_test_masked_data_6_features.pkl",
                used_features_csv="2023-11-06-12-00-00_1_70_used_features.csv",
                train_data_path='retina_xor_train_2023-11-06-11-31-00_adj_False_meta_True_10k_ep_nice_features.pkl',
                test_data_path='retina_xor_test_2023-11-06-11-31-00_adj_False_meta_True_10k_ep_nice_features.pkl',
                correlated_features_csv_name="2023-11-06-16-07-16_feature_correlation.csv",
                ergm_init_graphs_pkl_file_name='2023-11-26-15-23-30_all_good_archs_from_db_data_and_graph_per_dim.pkl',
                ergm_res_pkl_file_name=None,
            ),
            'random': SelectedExpNames(
                dims=None,
                num_layers=3,
                num_neurons=15,
                input_size=6,
                output_size=2,
                source_folder="retina_3_layers",
                feature_selection_folder='exp_2023-11-27-15-51-15_nice_features',
                first_analsis_csv=f"2023-10-24-10-00-54_all_results_from_teach_archs_results_with_motifs_5000_ep_no_duplicates.csv",
                num_selected_features=6,
                selected_features_data="retina_xor_2023-11-27-15-51-15_all_train_test_masked_data_6_features.pkl",
                used_features_csv='2023-11-27-15-51-15_1_70_used_features.csv',
                test_data_path='retina_xor_test_2023-11-27-15-26-59_adj_False_meta_True_10k_ep_nice_features_no_inv.pkl',
                train_data_path='retina_xor_train_2023-11-27-15-26-59_adj_False_meta_True_10k_ep_nice_features_no_inv.pkl',
                correlated_features_csv_name="2023-11-28-13-17-26_feature_correlation.csv",
                ergm_init_graphs_pkl_file_name="2023-11-27-16-24-51_all_good_archs_from_db_data_and_graph_per_dim.pkl",
                ergm_res_pkl_file_name="2023-11-28-10-26-09_good_archs_combined_ergm_samples.pkl",
                ergm_res_pkl_file_name_random_init="2023-12-01-14-09-53_good_archs_combined_ergm_samples_ransom_init.pkl",
                gen_graphs_pkl_file_name='2023-12-02-14-48-29_good_archs_combined_gen_samples_names_1s.pkl',
            ),
        },
    "xor": {
        (6, 5, 3): SelectedExpNames(
            feature_selection_folder='exp_2023-09-16-13-35-58_nice_features',
            first_analsis_csv="2023-09-05-09-24-47_all_results_from_teach_archs_results_with_motifs_10000_ep.csv",
            source_folder="xor_4_layers_6_5_3",
            num_selected_features=4,
            selected_features_data='2023-09-16-13-35-58_all_data_4_features_nica_features.pkl',
            used_features_csv="2023-09-16-13-35-58_1_70_used_features.csv",
            dims=[6, 6, 5, 3, 2],
        ),
        (6, 4, 4): SelectedExpNames(
            feature_selection_folder='exp_2023-10-12-16-24-09_nice_features',
            first_analsis_csv="2023-09-07-09-37-30_all_results_from_teach_archs_results_with_motifs_10000_ep.csv",
            source_folder="xor_4_layers_6_4_4",
            num_selected_features=4,
            selected_features_data='xor_2023-10-12-16-24-09_all_train_test_masked_data_4_features.pkl',
            used_features_csv="2023-10-12-16-24-09_1_70_used_features.csv",
            dims=[6, 6, 4, 4, 2],
        ),
        'random': SelectedExpNames(
            dims=None,
            num_layers=4,
            num_neurons=22,
            input_size=6,
            output_size=2,
            source_folder="xor_4_layers",
            feature_selection_folder='exp_2023-11-16-17-38-02_nice_features',
            first_analsis_csv="2023-11-16-14-52-18_all_results_from_teach_archs_results_with_motifs_5000_ep_no_duplicates.csv",
            num_selected_features=5,
            selected_features_data="xor_2023-11-16-17-38-02_all_train_test_masked_data_5_features.pkl",
            used_features_csv="2023-11-16-17-38-02_1_90_used_features.csv",
            train_data_path='xor_train_2023-11-16-15-28-43_adj_False_meta_True_10k_ep_nice_features.pkl',
            test_data_path='xor_test_2023-11-16-15-28-43_adj_False_meta_True_10k_ep_nice_features.pkl',
            correlated_features_csv_name='2023-11-17-11-50-45_feature_correlation.csv',
            ergm_init_graphs_pkl_file_name='2023-11-23-16-32-30_all_good_archs_from_db_data_and_graph_per_dim.pkl',
            ergm_res_pkl_file_name="2023-11-25-17-47-32_good_archs_combined_ergm_samples.pkl",
            ergm_res_pkl_file_name_random_init='2023-12-01-14-14-58_good_archs_combined_ergm_samples_random_init.pkl',
            gen_graphs_pkl_file_name='2023-12-02-14-47-44_good_archs_combined_gen_samples_names_1s.pkl',
        ),
    },
    "digits": {
        (6, 6): SelectedExpNames(
            feature_selection_folder='exp_2023_10_17_10_57_0_nice_features',
            first_analsis_csv="2023-09-10-10-15-41_all_results_from_teach_archs_results_with_motifs_2000_ep.csv",
            source_folder="digits_3_layers_6_6",
            num_selected_features=4,
            selected_features_data=None,
            used_features_csv=None,
            dims=[64, 6, 6, 10],
        ),
        (8, 4): SelectedExpNames(
            feature_selection_folder='exp_2023_09_15_14_35_0_nice_features',
            first_analsis_csv=f"2023-09-09-11-50-13_all_results_from_teach_archs_results_with_motifs_2000_ep.csv",
            source_folder="digits_3_layers_8_4",
            num_selected_features=6,
            selected_features_data=None,
            used_features_csv=None,
            dims=[64, 8, 4, 10],
        ),
        'random': SelectedExpNames(
            dims=None,
            num_layers=3,
            num_neurons=86,
            input_size=64,
            output_size=10,
            source_folder="digits_3_layers",
            feature_selection_folder='exp_2023_11_27_11_00_0_nice_features',
            first_analsis_csv="2023-11-26-13-45-59_all_results_from_teach_archs_results_with_motifs_1000_ep_no_duplicates_fixed.csv",
            num_selected_features=3,
            selected_features_data="digits_2023-11-27-14-19-53_all_train_test_masked_data_3_features.pkl",
            used_features_csv="2023-11-28-11-32-00_all_used_features.csv",
            train_data_path="digits_train_2023-11-26-16-04-07_adj_False_meta_True_10k_ep_nice_features.pkl",
            test_data_path="digits_test_2023-11-26-16-04-07_adj_False_meta_True_10k_ep_nice_features.pkl",
            correlated_features_csv_name="2023-11-28-12-45-38_feature_correlation.csv",
            ergm_init_graphs_pkl_file_name="2023-11-28-14-05-14_all_good_archs_from_db_data_and_graph_per_dim.pkl",
            ergm_res_pkl_file_name="2023-11-29-11-29-49_good_archs_combined_ergm_samples.pkl",
            ergm_res_pkl_file_name_random_init='2023-12-01-14-23-00_good_archs_combined_ergm_samples_random_init.pkl',
            gen_graphs_pkl_file_name='2023-11-29-10-46-48_good_archs_combined_gen_samples_names.pkl'
        ),
    },
}
