from parameters.digits_parameters import digit_structural_features_vec_length, \
    digit_structural_features_vec_length_with_motifs
from parameters.retina_parameters import retina_structural_features_vec_length_with_motifs, \
    retina_target_label_ranges
from parameters.xor_parameters import xor_structural_features_vec_length_with_motifs
from stractural_features_models.structural_features import CONSTANT_FEATURES, NICE_FEATURES_NO_DIST, NICE_FEATURES, \
    ONLY_TOP, NICE_FEATURES_NO_INV

main_predict_techability_params_digits = {
    1: {
        'consider_meta_data': False,
        'consider_adj_mat': True,
        "results_csvs": [
            "2023-06-26-12-59-26_all_results_from_digits_teach_archs_results_no_duplicates.csv",
        ],
        "results_model_path": [
            "digits_teach_archs_new_models",
        ],
        "folder_path": "teach_archs/digits",
        "structural_features_vec_length": digit_structural_features_vec_length
    },
    2: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        "results_csvs": [
            "2023-06-26-12-59-26_all_results_from_digits_teach_archs_results_no_duplicates.csv",
        ],
        "results_model_path": [
            "digits_teach_archs_new_models_with_motifs",
        ],
        "folder_path": "teach_archs/digits",
        "structural_features_vec_length": digit_structural_features_vec_length_with_motifs
    },
    3: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        "results_csvs": [
            "2023-06-26-12-59-26_all_results_from_digits_teach_archs_results_no_duplicates.csv",
        ],
        "results_model_path": [
            "digits_teach_archs_new_models_with_motifs",
        ],
        "folder_path": "teach_archs/digits",
        "structural_features_vec_length": digit_structural_features_vec_length_with_motifs
    },
}

main_predict_techability_params_retina = {
    1: {
        'consider_meta_data': False,
        'consider_adj_mat': True,
        "results_csvs": [
            "2023-04-13-16-23-07_all_results_from_retina_teach_archs_results.csv",
        ],
        "results_model_path": [
            "retina_teach_archs_models_with_motifs",
        ],
        "folder_path": "teach_archs/retina",
        "structural_features_vec_length": retina_structural_features_vec_length_with_motifs
    },
    2: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        "results_csvs": [
            "2023-04-13-16-23-07_all_results_from_retina_teach_archs_results.csv",
        ],
        "results_model_path": [
            "retina_teach_archs_models_with_motifs",
        ],
        "folder_path": "teach_archs/retina",
        "structural_features_vec_length": retina_structural_features_vec_length_with_motifs
    },
    3: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        "results_csvs": [
            "2023-04-13-16-23-07_all_results_from_retina_teach_archs_results.csv",
        ],
        "results_model_path": [
            "retina_teach_archs_models_with_motifs",
        ],
        "folder_path": "teach_archs/retina",
        "structural_features_vec_length": retina_structural_features_vec_length_with_motifs
    },
}
main_predict_techability_params_retina_multi_arch_4_layers = {
    "folder_path": "retina/retina_4_layers_6_4_2",
    "output_path": "teach_archs_regression_results",
    "data_folder": 'train_test_data',
    'sufix': '_adj_False_meta_True_2k_ephoc',
    'learning_rate': 0.001,
    'num_epochs': 2000,
    'batch_size': 512,
    1: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_10000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_10000_ep.pkl',
        'sufix': '_adj_False_meta_True_10000_eph_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    2: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_10000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_10000_ep.pkl',
        'sufix': '_adj_False_meta_True_10000_eph_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    3: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_100_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_100_ep.pkl',
        'sufix': '_adj_False_meta_True_100_eph_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    4: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_100_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_100_ep.pkl',
        'sufix': '_adj_False_meta_True_100_eph_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    5: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_1000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_1000_ep.pkl',
        'sufix': '_adj_False_meta_True_1000_eph_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    6: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_1000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_1000_ep.pkl',
        'sufix': '_adj_False_meta_True_1000_eph_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    7: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_2000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_2000_ep.pkl',
        'sufix': '_adj_False_meta_True_2000_eph_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    8: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_2000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_2000_ep.pkl',
        'sufix': '_adj_False_meta_True_2000_eph_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    9: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_4000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_4000_ep.pkl',
        'sufix': '_adj_False_meta_True_4000_eph_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    10: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_4000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_4000_ep.pkl',
        'sufix': '_adj_False_meta_True_4000_eph_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    11: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_8000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_8000_ep.pkl',
        'sufix': '_adj_False_meta_True_8000_eph_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    12: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_8000_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_8000_ep.pkl',
        'sufix': '_adj_False_meta_True_8000_eph_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    13: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_500_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_500_ep.pkl',
        'sufix': '_adj_False_meta_True_500_eph_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    14: {
        'train_path': 'retina_train_2023-09-03-11-45-09_adj_False_meta_True_500_ep.pkl',
        'test_path': 'retina_test_2023-09-03-11-45-09_adj_False_meta_True_500_ep.pkl',
        'sufix': '_adj_False_meta_True_500_eph_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,

    },
}

main_predict_techability_params_retina_xor = {
    "folder_path": "retina_xor/retina_3_layers/",
    "output_path": "teach_archs_regression_results",
    "data_folder": 'train_test_data',
    1: {
        'train_path': 'retina_xor_train_2023-10-24-10-20-05_adj_False_meta_True_10k_ep_nice_features_normed.pkl',
        'test_path': 'retina_xor_test_2023-10-24-10-20-05_adj_False_meta_True_10k_ep_nice_features_normed.pkl',
        'sufix': '_adj_False_meta_True_2k_ephoc_nice_features_normed',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,
    },
    2: {
        'train_path': 'retina_xor_train_2023-10-24-10-20-09_adj_False_meta_True_10k_ep_nice_features.pkl',
        'test_path': 'retina_xor_test_2023-10-24-10-20-09_adj_False_meta_True_10k_ep_nice_features.pkl',
        'sufix': '_adj_False_meta_True_2k_ephoc_nice_features',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,
    },
}
main_predict_techability_params_xor_new = {
    "folder_path": "xor/xor_4_layers/",
    "output_path": "teach_archs_regression_results",
    "data_folder": 'train_test_data',
    1: {
        'train_path': 'xor_train_2023-10-23-10-13-33_adj_False_meta_True_10k_ep_nice_features_normed.pkl',
        'test_path': 'xor_test_2023-10-23-10-13-33_adj_False_meta_True_10k_ep_nice_features_normed.pkl',
        'sufix': '_adj_False_meta_True_2k_ephoc_nice_features_normed',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,
    },
    2: {
        'train_path': 'xor_train_2023-10-23-10-13-33_adj_False_meta_True_10k_ep_nice_features.pkl',
        'test_path': 'xor_test_2023-10-23-10-13-33_adj_False_meta_True_10k_ep_nice_features.pkl',
        'sufix': '_adj_False_meta_True_2k_ephoc_nice_features',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,
    },
}
main_predict_techability_params_digis_new = {
    "folder_path": "digits/digits_3_layers_8_4/",
    "output_path": "teach_archs_regression_results",
    "data_folder": 'train_test_data',
    1: {
        'train_path': 'digits_train_2023-09-15-14-03-40_adj_False_meta_True_10k_ep_nice_features.pkl',
        'test_path': 'digits_train_2023-09-15-14-03-40_adj_False_meta_True_10k_ep_nice_features.pkl',
        'sufix': '_const_meta_2k_ephoc_nice_features',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,
    },
    2: {
        'train_path': 'digits_train_2023-09-15-14-03-40_adj_False_meta_True_10k_ep_nice_features.pkl',
        'test_path': 'digits_train_2023-09-15-14-03-40_adj_False_meta_True_10k_ep_nice_features.pkl',
        'sufix': '_const_meta_2k_ephoc_nice_features',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,
    },
}
main_predict_techability_params_retina_multi_arch_3_layers = {
    "folder_path": "retina/retina_3_layers_3_4/",
    "output_path": "teach_archs_regression_results",
    "data_folder": 'train_test_data',
    'sufix': '_2k_ephoc_nice_features',
    1: {
        'train_path': 'retina_xor_train_2023-09-12-18-46-01_adj_False_meta_True_10k_ep_nice_features.pkl',
        'test_path': 'retina_xor_test_2023-09-12-18-46-01_adj_False_meta_True_10k_ep_nice_features.pkl',
        'sufix': '_2k_ephoc_nice_features',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    2: {
        'train_path': 'retina_xor_train_2023-09-12-18-46-01_adj_False_meta_True_10k_ep_nice_features.pkl',
        'test_path': 'retina_xor_test_2023-09-12-18-46-01_adj_False_meta_True_10k_ep_nice_features.pkl',
        'sufix': '_2k_ephoc_nice_features',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    3: {
        'train_path': 'retina_train_2023-08-21-11-51-16_adj_False_meta_True.pkl',
        'test_path': 'retina_test_2023-08-21-11-51-16_adj_False_meta_True.pkl',
        'sufix': '_all_meta_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,
    },
    4: {
        'train_path': 'retina_train_2023-08-21-11-51-16_adj_False_meta_True.pkl',
        'test_path': 'retina_test_2023-08-21-11-51-16_adj_False_meta_True.pkl',
        'sufix': '_all_meta_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,
    },
    5: {
        'train_path': 'retina_train_2023-08-21-11-51-18_adj_True_meta_True.pkl',
        'test_path': 'retina_test_2023-08-21-11-51-18_adj_True_meta_True.pkl',
        'sufix': '_all_meta_and_adj_2k_ephoc',
        'learning_rate': 0.001,
        'num_epochs': 2000,
        'batch_size': 512,

    },
    6: {
        'train_path': 'retina_train_2023-08-21-11-51-18_adj_True_meta_True.pkl',
        'test_path': 'retina_test_2023-08-21-11-51-18_adj_True_meta_True.pkl',
        'sufix': '_all_meta_and_adj_2k_ephoc',
        'learning_rate': 0.005,
        'num_epochs': 2000,
        'batch_size': 512,
    }
}

main_prepere_techability_params_retina_multi_arch_multi_ep = {
    "results_csvs_dir": "first_analysis_results",
    "results_model_path": "teach_archs_models",
    "folder_path": "retina/retina_4_layers_6_4_2",
    "train_test_folder": 'train_test_data',
    1: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': CONSTANT_FEATURES,
        "pkl_name_addition": '_const_features_normed'
    },
    2: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': CONSTANT_FEATURES,
        "pkl_name_addition": '_const_features'
    },
    3: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': None,
        "pkl_name_addition": '_normed'
    },
    4: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': None,
        "pkl_name_addition": ''
    },
    5: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        'normalize_features': True,
        'features_list': None,
        "pkl_name_addition": '_normed'
    },
    6: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        'normalize_features': False,
        'features_list': None,
        "pkl_name_addition": ''
    },
}

main_prepere_techability_params_retina_multi_arch = {
    1: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': CONSTANT_FEATURES,
        "results_csvs": [
            '2023-08-20-12-33-37_all_results_from_teach_archs_results_with_motifs_no_duplicates.csv',
        ],
        "results_model_path": [
            "teach_archs_models",
        ],
        "folder_path": "retina/retina_3_layers_5_2",
        "pkl_name_addition": '_const_features_normed'
    },
    2: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': CONSTANT_FEATURES,
        "results_csvs": [
            '2023-08-20-12-33-37_all_results_from_teach_archs_results_with_motifs_no_duplicates.csv',
        ],
        "results_model_path": [
            "teach_archs_models",
        ],
        "folder_path": "retina/retina_3_layers_5_2",
        "pkl_name_addition": '_const_features'
    },
    3: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': None,
        "results_csvs": [
            '2023-08-20-12-33-37_all_results_from_teach_archs_results_with_motifs_no_duplicates.csv',
        ],
        "results_model_path": [
            "teach_archs_models",
        ],
        "folder_path": "retina/retina_3_layers_5_2",
        "pkl_name_addition": '_normed'
    },
    4: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': None,
        "results_csvs": [
            '2023-08-20-12-33-37_all_results_from_teach_archs_results_with_motifs_no_duplicates.csv',

        ],
        "results_model_path": [
            "teach_archs_models",
        ],
        "folder_path": "retina/retina_3_layers_5_2",
        "pkl_name_addition": ''
    },
    5: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        'normalize_features': True,
        'features_list': None,
        "results_csvs": [
            '2023-08-20-12-33-37_all_results_from_teach_archs_results_with_motifs_no_duplicates.csv',
        ],
        "results_model_path": [
            "teach_archs_models",
        ],
        "folder_path": "retina/retina_3_layers_5_2",
        "pkl_name_addition": '_normed'
    },
    6: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        'normalize_features': False,
        'features_list': None,
        "results_csvs": [
            '2023-08-20-12-33-37_all_results_from_teach_archs_results_with_motifs_no_duplicates.csv',
        ],
        "results_model_path": [
            "teach_archs_models",
        ],
        "folder_path": "retina/retina_3_layers_5_2",
        "pkl_name_addition": ''
    },
}

main_prepere_techability_params_retina_xor = {
    "results_model_path": "teach_archs_models",
    "folder_path": "retina_xor/retina_3_layers",
    "train_test_folder": 'train_test_data',
    "results_csv": "first_analysis_results/2023-10-24-10-00-54_all_results_from_teach_archs_results_with_motifs_5000_ep_no_duplicates.csv",
    1: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': NICE_FEATURES_NO_INV,
        "pkl_name_addition": '_10k_ep_nice_features_no_inv',
    },
    2: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': NICE_FEATURES,
        "pkl_name_addition": '_10k_ep_nice_features'
    },
    3: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': NICE_FEATURES,
        "pkl_name_addition": '_10k_ep_nice_features_normed'
    },
    4: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': ONLY_TOP,
        "pkl_name_addition": '_10k_ep_top_3'
    },
    5: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        'normalize_features': True,
        'features_list': None,
        "pkl_name_addition": '_10k_ep_normed'
    },
    6: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        'normalize_features': False,
        'features_list': None,
        "pkl_name_addition": '_10k_ep'
    },
}

main_prepere_techability_params_digits_new = {
    "results_model_path": "teach_archs_models",
    "folder_path": "digits/digits_3_layers",
    "train_test_folder": 'train_test_data',
    "results_csv": "first_analysis_results/2023-11-26-13-45-59_all_results_from_teach_archs_results_with_motifs_1000_ep_no_duplicates_fixed.csv",
    1: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': CONSTANT_FEATURES,
        "pkl_name_addition": '_10k_ep_const_features_normed',
    },
    2: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': NICE_FEATURES,
        "pkl_name_addition": '_10k_ep_nice_features'
    },
    3: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': NICE_FEATURES,
        "pkl_name_addition": '_10k_ep_nice_features_normed'
    },
    4: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': None,
        "pkl_name_addition": '_10k_ep'
    },
    5: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': NICE_FEATURES,
        "pkl_name_addition": '_10k_ep_nice_features'
    },
}

main_prepere_techability_params_xor_new = {
    "results_model_path": "teach_archs_models",
    "folder_path": "xor/xor_4_layers",
    "train_test_folder": 'train_test_data',
    "results_csv": "first_analysis_results/2023-11-16-14-52-18_all_results_from_teach_archs_results_with_motifs_5000_ep_no_duplicates.csv",
    1: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': CONSTANT_FEATURES,
        "pkl_name_addition": '_10k_ep_const_features_normed',
    },
    2: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': CONSTANT_FEATURES,
        "pkl_name_addition": '_10k_ep_const_features'
    },
    4: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': False,
        'features_list': NICE_FEATURES,
        "pkl_name_addition": '_10k_ep_nice_features'
    },
    5: {
        'consider_meta_data': True,
        'consider_adj_mat': False,
        'normalize_features': True,
        'features_list': NICE_FEATURES,
        "pkl_name_addition": '_10k_ep_nice_features_normed'
    },
    6: {
        'consider_meta_data': True,
        'consider_adj_mat': True,
        'normalize_features': False,
        'features_list': None,
        "pkl_name_addition": '_10k_ep'
    },
}
main_predict_techability_params_xor = {
    1: {
        'out_path': 'adj_True_meta_False',
        'consider_meta_data': False,
        'consider_adj_mat': True,
        "results_csvs": [
            "2023-04-10-17-25-10_all_results_from_xor_teach_archs_results.csv",
        ],
        "results_model_path": [
            "xor_teach_archs_models_with_motifs",
        ],
        "folder_path": "teach_archs/xors",
        "structural_features_vec_length": xor_structural_features_vec_length_with_motifs
    },
    2: {
        'out_path': 'adj_False_meta_True',
        'consider_meta_data': True,
        'consider_adj_mat': False,
        "results_csvs": [
            "2023-04-10-17-25-10_all_results_from_xor_teach_archs_results.csv",
        ],
        "results_model_path": [
            "xor_teach_archs_models_with_motifs",
        ],
        "folder_path": "teach_archs/xors",
        "structural_features_vec_length": xor_structural_features_vec_length_with_motifs
    },
    3: {
        'out_path': 'adj_True_meta_True',
        'consider_meta_data': True,
        'consider_adj_mat': True,
        "results_csvs": [
            "2023-04-10-17-25-10_all_results_from_xor_teach_archs_results.csv",
        ],
        "results_model_path": [
            "xor_teach_archs_models_with_motifs",
        ],
        "folder_path": "teach_archs/xors",
        "structural_features_vec_length": xor_structural_features_vec_length_with_motifs
    },
}

main_predict_techability_xor = {
    "folder_path": "teach_archs/xors",
    "output_path": "xor_teach_archs_regression_results",
    'results_csv_name': None,
    'results_model_path': None,
    1: {
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-54_adj_True_meta_False.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-54_adj_True_meta_False.pkl',
        'sufix': '_adj_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    2: {
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    3: {
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-49_adj_True_meta_True.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-49_adj_True_meta_True.pkl',
        'sufix': '',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    4: {
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-54_adj_True_meta_False.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-54_adj_True_meta_False.pkl',
        'sufix': '_adj_only',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,
    },
    5: {
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'sufix': '_meta_only',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,

    },
    6: {
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-49_adj_True_meta_True.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-49_adj_True_meta_True.pkl',
        'sufix': '',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,
    },
}
main_predict_techability_retina = {
    "folder_path": "teach_archs/retina",
    "output_path": "retina_teach_archs_regression_results",
    'results_csv_name': None,
    'results_model_path': None,
    1: {
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'sufix': '_adj_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    2: {
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl',
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,

    },
    3: {
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_True_meta_True.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_True_meta_True.pkl',
        'sufix': '',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    4: {
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'sufix': '_adj_only',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,
    },
    5: {
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl',
        'sufix': '_meta_only',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,

    },
    6: {
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_True_meta_True.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_True_meta_True.pkl',
        'sufix': '',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,
    },
}
main_predict_techability_digits = {
    "folder_path": "teach_archs/digits",
    "output_path": "digits_teach_archs_regression_results",
    'results_csv_name': None,
    'results_model_path': None,
    1: {
        'train_path': 'digits_train_test_data/digits_train_2023-06-26-14-53-44_adj_True_meta_False.pkl',
        'test_path': 'digits_train_test_data/digits_test_2023-06-26-14-53-44_adj_True_meta_False.pkl',
        'sufix': '_adj_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    2: {
        'train_path': 'digits_train_test_data/digits_train_2023-06-26-14-00-11_adj_False_meta_True.pkl',
        'test_path': 'digits_train_test_data/digits_test_2023-06-26-14-00-11_adj_False_meta_True.pkl',
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,

    },
    3: {
        'train_path': 'digits_train_test_data/digits_train_2023-06-26-15-36-02_adj_True_meta_True.pkl',
        'test_path': 'digits_train_test_data/digits_test_2023-06-26-15-36-02_adj_True_meta_True.pkl',
        'sufix': '',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    4: {
        'train_path': 'digits_train_test_data/digits_train_2023-06-26-14-53-44_adj_True_meta_False.pkl',
        'test_path': 'digits_train_test_data/digits_test_2023-06-26-14-53-44_adj_True_meta_False.pkl',
        'sufix': '_adj_only',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,
    },
    5: {
        'train_path': 'digits_train_test_data/digits_train_2023-06-26-14-00-11_adj_False_meta_True.pkl',
        'test_path': 'digits_train_test_data/digits_test_2023-06-26-14-00-11_adj_False_meta_True.pkl',
        'sufix': '_meta_only',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,

    },
    6: {
        'train_path': 'digits_train_test_data/digits_train_2023-06-26-15-36-02_adj_True_meta_True.pkl',
        'test_path': 'digits_train_test_data/digits_test_2023-06-26-15-36-02_adj_True_meta_True.pkl',
        'sufix': '',
        'learning_rate': 0.005,
        'num_epochs': 250,
        'batch_size': 512,
    },
}

main_predict_techability_xor_lightgbm_feature_selection = {
    'exp_path': "teach_archs/xors/xor_lightgbm_feature_selection/exp_2023-04-25-13-56-23",
    "output_path": "/teach_archs_regression_feature_selection_results/2023-05-22-11-35_500_eph",
    'learning_rate': 0.001,
    'num_epochs': 1000,
    'batch_size': 512,
    'csv_path': '2023-05-04-12-37-34_feature_selection.csv',
}
main_predict_techability_digits_lightgbm_feature_selection = {
    'exp_path': "teach_archs/digits/digits_lightgbm_feature_selection/exp_2023-07-01-10-52-11",
    "output_path": "/teach_archs_regression_feature_selection_results/2023-07-01-10-52-11_500_eph",
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
    'csv_path': 'partial_csvs/2023-07-03-14-45-23_5_20_feature_selection.csv',
}
main_predict_techability_retina_lightgbm_feature_selection = {
    "exp_path": "teach_archs/retina/retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31",
    "output_path": "/teach_archs_regression_feature_selection_results/2023-05-22-11-35_500_eph",
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
    'csv_path': '2023-05-04-12-30-24_feature_selection.csv'
}
main_predict_techability_xor_xgboost_feature_selection = {
    'exp_path': "teach_archs/xors/xor_xgboost_feature_selection/exp_2023-05-01-13-20-50",
    "output_path": "/teach_archs_regression_feature_selection_results/2023-05-22-11-35_500_eph",
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
    'csv_path': '2023-05-04-12-41-42_feature_selection.csv',
}

main_predict_techability_retina_xgboost_feature_selection = {
    "exp_path": "teach_archs/retina/retina_xgboost_feature_selection/exp_2023-04-29-16-49-26",
    "output_path": "/teach_archs_regression_feature_selection_results/2023-05-22-11-35_500_eph",
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
    'csv_path': '2023-05-04-12-14-43_feature_selection.csv'
}

main_predict_techability_xor_ind_feature_selection = {
    'exp_path': "teach_archs/xor/xor_feature_selection",
    "output_path": "/teach_archs_regression_feature_selection_results",
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
    1:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-35-29_masked_data_density_ratio_between_layers.pkl',
            "model_name": 'density_ratio_between_layers',
        },
    2:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-31-50_masked_data_num_paths.pkl',
            "model_name": 'num_paths',
        },
    3:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-41-37_masked_data_num_involved_neurons.pkl',
            "model_name": 'num_involved_neurons',
        },
    4:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-43-48_masked_data_num_paths_and_entropy.pkl',
            "model_name": 'num_paths_and_entropy',
        },
    5:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-30-45_masked_data_density_and_num_paths.pkl',
            "model_name": 'density_and_num_paths',
        },
    6:
        {
            'model_path': 'masked_data_and_models/2023-05-15-14-51-14_masked_data_in_out_connections_model_features.pkl',
            "model_name": 'in_out_connections',
        },
    7:
        {
            'model_path': "masked_data_and_models/2023-05-20-13-05-07_masked_data_num_paths_entropy_distances_between_pairs_density.pkl",
            "model_name": 'num_paths_entropy_distances_between_pairs_density',

        },
    8:
        {
            'model_path': "masked_data_and_models/2023-05-20-13-08-31_masked_data_num_paths_connectivity_ratio_between_layers_distances_between_pairs_density.pkl",
            "model_name": 'num_paths_connectivity_ratio_between_layers_distances_between_pairs_density',
        },
    11: {
        'model_path': "masked_data_and_models/2023-08-01-15-54-17_masked_data_motifs_connectivity_ratio_entropy.pkl",
        "model_name": 'motifs_count_density_entropy',
    },
}
main_predict_techability_retina_ind_feature_selection = {
    'exp_path': "teach_archs/retina/retina_feature_selection",
    "output_path": "/teach_archs_regression_feature_selection_results",
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
    1:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-34-55_masked_data_density_ratio_between_layers.pkl',
            "model_name": 'density_ratio_between_layers',
        },
    2:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-32-44_masked_data_num_paths.pkl',
            "model_name": 'num_paths',
        },
    3:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-41-49_masked_data_num_involved_neurons.pkl',
            "model_name": 'num_involved_neurons',
        },
    4:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-43-03_masked_data_num_paths_and_entropy.pkl',
            "model_name": 'num_paths_and_entropy',
        },
    5:
        {
            'model_path': 'masked_data_and_models/2023-05-18-14-33-29_masked_data_density_and_num_paths.pkl',
            "model_name": 'density_and_num_paths',
        },
    6:
        {
            'model_path': 'masked_data_and_models/2023-05-15-14-29-30_masked_data_in_out_connections_model_features.pkl',
            "model_name": 'in_out_connections',
        },
    7:
        {
            'model_path': "masked_data_and_models/2023-05-20-13-04-45_masked_data_num_paths_entropy_distances_between_pairs_density.pkl",
            "model_name": 'num_paths_entropy_distances_between_pairs_density',

        },
    8:
        {
            'model_path': "masked_data_and_models/2023-07-01-12-42-22_masked_data_motifs_count.pkl",
            "model_name": 'motifs_count',
        },
    9:
        {
            'model_path': "masked_data_and_models/2023-07-01-12-45-44_masked_data_motifs_count_connectivity_ratio.pkl",
            "model_name": 'motifs_count_density',
        },
    10: {
        'model_path': "masked_data_and_models/2023-07-01-12-46-53_masked_data_motifs_count_connectivity_ratio_entropy.pkl",
        "model_name": 'motifs_count_density_entropy',
    },
    11: {
        'model_path': "masked_data_and_models/2023-08-01-15-52-08_masked_data_motifs_connectivity_ratio_entropy.pkl",
        "model_name": 'motifs_count_density_entropy',
    },
}
main_predict_techability_digit_ind_feature_selection = {
    'exp_path': "teach_archs/digits/digits_feature_selection",
    "output_path": "/teach_archs_regression_feature_selection_results",
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
    11: {
        'model_path': "masked_data_and_models/2023-08-01-15-58-24_masked_data_motifs_connectivity_ratio_entropy.pkl",
        "model_name": 'motifs_count_density_entropy',
    },
}
predict_techability_logistic_regression_retina_selected_features = {
    "exp_path": "teach_archs/retina/retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31",
    "folder_path": "teach_archs/retina",
    "output_path": "retina_teach_archs_logistic_regression_results",
    'learning_rate': 0.001,
    'num_epochs': 1000,
    'batch_size': 512,
    'lambda_reg': 0.5,
    'csv_path': '2023-05-04-12-30-24_feature_selection.csv'
}
predict_techability_logistic_regression_xor_selected_features = {
    'exp_path': "teach_archs/xors/xor_lightgbm_feature_selection/exp_2023-04-25-13-56-23",
    "folder_path": "teach_archs/xors",
    "output_path": "xor_teach_archs_logistic_regression_results",
    'learning_rate': 0.001,
    'num_epochs': 250,
    'batch_size': 512,
    'lambda_reg': 0.5,
    'csv_path': '2023-05-04-12-37-34_feature_selection.csv',
}

predict_techability_logistic_regression_xor = {
    1: {
        "folder_path": "teach_archs/xors",
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'results_csv_name': None,
        'results_model_path': None,
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
        'lambda_reg': 0.5,
    },
    2: {
        "folder_path": "teach_archs/xors",
        "output_path": "xor_teach_archs_logistic_regression_results",
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'results_csv_name': None,
        'results_model_path': None,
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
        'lambda_reg': 1.0,
    },
    3: {
        "folder_path": "teach_archs/xors",
        "output_path": "xor_teach_archs_logistic_regression_results",
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'results_csv_name': None,
        'results_model_path': None,
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
        'lambda_reg': 2.0,
    },
    4: {
        "folder_path": "teach_archs/xors",
        "output_path": "xor_teach_archs_logistic_regression_results",
        'train_path': 'xor_train_test_data/xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'test_path': 'xor_train_test_data/xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl',
        'results_csv_name': None,
        'results_model_path': None,
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
        'lambda_reg': 3.0,
    },
}
predict_techability_logistic_regression_retina = {
    1: {
        "folder_path": "teach_archs/retina",
        "output_path": "retina_teach_archs_logistic_regression_results",
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'results_csv_name': None,
        'results_model_path': None,
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 1000,
        'batch_size': 512,
        'lambda_reg': 0.5,
    },
    2: {
        "folder_path": "teach_archs/retina",
        "output_path": "retina_teach_archs_logistic_regression_results",
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl',
        'results_csv_name': None,
        'results_model_path': None,
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 1000,
        'batch_size': 512,
        'lambda_reg': 1.0,

    },
    3: {
        "folder_path": "teach_archs/retina",
        "output_path": "retina_teach_archs_logistic_regression_results",
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_True_meta_True.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_True_meta_True.pkl',
        'results_csv_name': None,
        'results_model_path': None,
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 1000,
        'batch_size': 512,
        'lambda_reg': 2.0,
    },
    4: {
        "folder_path": "teach_archs/retina",
        "output_path": "retina_teach_archs_logistic_regression_results",
        'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_True_meta_False.pkl',
        'results_csv_name': None,
        'results_model_path': None,
        'sufix': '_meta_only',
        'learning_rate': 0.001,
        'num_epochs': 1000,
        'batch_size': 512,
        'lambda_reg': 3.0,
    },
}

baseline_predict_techability = {
    1: {
        'train_path': 'train_test_data/train_2023-02-12-15-53-25_adj_True_meta_False.pkl',
        'test_path': 'train_test_data/test_2023-02-12-15-53-25_adj_True_meta_False.pkl',
        'out_path': 'adj_True_meta_False',
        'learning_rate': 0.001,
        'num_epochs': 10000,
        'activate': True,
    },
    2: {
        'train_path': 'train_test_data/train_2023-02-12-15-53-25_adj_False_meta_True.pkl',
        'test_path': 'train_test_data/test_2023-02-12-15-53-25_adj_False_meta_True.pkl',
        'out_path': 'adj_False_meta_True',
        'learning_rate': 0.001,
        'num_epochs': 10000,
        'activate': True,
    },
    3: {
        'train_path': 'train_test_data/train_2023-02-12-15-53-24_adj_True_meta_True.pkl',
        'test_path': 'train_test_data/test_2023-02-12-15-53-24_adj_True_meta_True.pkl',
        'out_path': 'adj_True_meta_True',
        'learning_rate': 0.001,
        'num_epochs': 10000,
        'activate': True,
    },
}

main_teach_fast = {
    1: [0.04, 0.3, 25, True],
    2: [0.04, 0.4, 25, True],
    3: [0.04, 0.5, 25, True],
    4: [0.04, 0.6, 25, True],
    5: [0.04, 0.7, 25, True],
    6: [0.04, 0.8, 25, True],
    7: [0.04, 0.9, 25, True],
    8: [0.04, 1.0, 25, True],
    9: [0.035, 0.3, 25, True],
    10: [0.035, 0.4, 25, True],
    11: [0.035, 0.5, 25, True],
    12: [0.035, 0.6, 25, True],
    13: [0.035, 0.7, 25, True],
    14: [0.035, 0.8, 25, True],
    15: [0.035, 0.9, 25, True],
    16: [0.035, 1.0, 25, True],
    17: [0.04, 0.3, 15, True],
    18: [0.04, 0.4, 15, True],
    19: [0.04, 0.5, 15, True],
    20: [0.04, 0.6, 15, True],
    21: [0.04, 0.7, 15, True],
    22: [0.04, 0.8, 15, True],
    23: [0.04, 0.9, 15, True],
    24: [0.04, 1.0, 15, True],
    25: [0.035, 0.3, 15, True],
    26: [0.035, 0.4, 15, True],
    27: [0.035, 0.5, 15, True],
    28: [0.035, 0.6, 15, True],
    29: [0.035, 0.7, 15, True],
    30: [0.035, 0.8, 15, True],
    31: [0.035, 0.9, 15, True],
    32: [0.035, 1.0, 15, True],
    33: [0.05, 0.3, 25, True],
    34: [0.05, 0.4, 25, True],
    35: [0.05, 0.5, 25, True],
    36: [0.05, 0.6, 25, True],
    37: [0.05, 0.7, 25, True],
    38: [0.05, 0.8, 25, True],
    39: [0.05, 0.9, 25, True],
    40: [0.05, 1.0, 25, True],
    41: [0.01, 0.3, 25, True],
    42: [0.01, 0.4, 25, True],
    43: [0.01, 0.5, 25, True],
    44: [0.01, 0.6, 25, True],
    45: [0.01, 0.7, 25, True],
    46: [0.01, 0.8, 25, True],
    47: [0.01, 0.9, 25, True],
    48: [0.01, 1.0, 25, True],
    49: [0.05, 0.3, 15, True],
    50: [0.05, 0.4, 15, True],
    51: [0.05, 0.5, 15, True],
    52: [0.05, 0.6, 15, True],
    53: [0.05, 0.7, 15, True],
    54: [0.05, 0.8, 15, True],
    55: [0.05, 0.9, 15, True],
    56: [0.04, 1.0, 15, True],
    57: [0.01, 0.3, 15, True],
    58: [0.01, 0.4, 15, True],
    59: [0.01, 0.5, 15, True],
    60: [0.01, 0.6, 15, True],
    61: [0.01, 0.7, 15, True],
    62: [0.01, 0.8, 15, True],
    63: [0.01, 0.9, 15, True],
    64: [0.01, 1.0, 15, True],
    65: [0.1, 0.3, 25, True],
    66: [0.1, 0.4, 25, True],
    67: [0.1, 0.5, 25, True],
    68: [0.1, 0.6, 25, True],
    69: [0.1, 0.7, 25, True],
    70: [0.1, 0.8, 25, True],
    71: [0.1, 0.9, 25, True],
    72: [0.1, 1.0, 25, True],
    73: [0.1, 0.3, 15, True],
    74: [0.1, 0.4, 15, True],
    75: [0.1, 0.5, 15, True],
    76: [0.1, 0.6, 15, True],
    77: [0.1, 0.7, 15, True],
    78: [0.1, 0.8, 15, True],
    79: [0.1, 0.9, 15, True],
    80: [0.1, 1.0, 15, True],
    81: [0.04, 0.3, 20, True],
    82: [0.04, 0.4, 20, True],
    83: [0.04, 0.5, 20, True],
    84: [0.04, 0.6, 20, True],
    85: [0.04, 0.7, 20, True],
    86: [0.04, 0.8, 20, True],
    87: [0.04, 0.9, 20, True],
    88: [0.04, 1.0, 20, True],
    89: [0.035, 0.3, 20, True],
    90: [0.035, 0.4, 20, True],
    91: [0.035, 0.5, 20, True],
    92: [0.035, 0.6, 20, True],
    93: [0.035, 0.7, 20, True],
    94: [0.035, 0.8, 20, True],
    95: [0.035, 0.9, 20, True],
    96: [0.035, 1.0, 20, True],
    99: [0.1, 0.3, 20, True],
    100: [0.1, 0.4, 20, True],
    101: [0.1, 0.5, 20, True],
    102: [0.1, 0.6, 20, True],
    103: [0.1, 0.7, 20, True],
    104: [0.1, 0.8, 20, True],
    105: [0.1, 0.9, 20, True],
    106: [0.1, 1.0, 20, True],
    107: [0.01, 0.3, 20, True],
    108: [0.01, 0.4, 20, True],
    109: [0.01, 0.5, 20, True],
    110: [0.01, 0.6, 20, True],
    111: [0.01, 0.7, 20, True],
    112: [0.01, 0.8, 20, True],
    113: [0.01, 0.9, 20, True],
    114: [0.01, 1.0, 20, True],
    115: [0.05, 0.3, 20, True],
    116: [0.05, 0.4, 20, True],
    117: [0.05, 0.5, 20, True],
    118: [0.05, 0.6, 20, True],
    119: [0.05, 0.7, 20, True],
    120: [0.05, 0.8, 20, True],
    121: [0.05, 0.9, 20, True],
}

main_classify_good_bad_archs = {
    1: {
        'train_path': 'train_test_data/train_2023-02-16-17-09-36_adj_smart_vs_random_adj.pkl',
        'test_path': 'train_test_data/test_2023-02-16-17-09-36_adj_smart_vs_random_adj.pkl',
        'learning_rate': 0.001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    2: {
        'train_path': 'train_test_data/train_2023-02-16-17-09-36_adj_smart_vs_random_adj.pkl',
        'test_path': 'train_test_data/test_2023-02-16-17-09-36_adj_smart_vs_random_adj.pkl',
        'learning_rate': 0.0001,
        'num_epochs': 250,
        'batch_size': 512,
    },
    3: {
        'train_path': 'train_test_data/train_2023-02-16-17-09-36_adj_smart_vs_random_adj.pkl',
        'test_path': 'train_test_data/test_2023-02-16-17-09-36_adj_smart_vs_random_adj.pkl',
        'learning_rate': 0.00001,
        'num_epochs': 250,
        'batch_size': 512,
    },
}

retina_find_features_spectrom = {
    'task': 'retina',
    'results_folder': 'retina_teach_archs_requiered_features_kernel_dist/20_features',
    'train_path': f'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_True_meta_True.pkl',
    'test_path': f'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_True_meta_True.pkl',
    'used_features_csv': f"retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/2023-05-04-12-30-24_used_features.csv",
    'num_features': 20,
    'kernel_models_pkl_name': '2023-06-01-14-24-51_kernel_models.pkl',
    'target_label_ranges_pkl_name': '2023-06-01-14-24-51_target_label_ranges.pkl'
}

xor_find_features_spectrom = {
    'task': 'xors',
    'results_folder': 'xor_teach_archs_models_requiered_features/5_features',
    'train_path': f'xor_train_test_data/xor_train_2023-04-13-14-15-49_adj_True_meta_True.pkl',
    'test_path': f'xor_train_test_data/xor_test_2023-04-13-14-15-49_adj_True_meta_True.pkl',
    'used_features_csv': f"xor_lightgbm_feature_selection/exp_2023-04-25-13-56-23/2023-05-04-12-37-34_used_features.csv",
    'num_features': 5,
}

regression_tree_num_features = {
    1: [1, ],
    2: [3, 4, ],
    3: [5, 6, ],
    4: [7, 8, ],
    5: [9, 10],
    6: [20, 30, ],
    7: [40, 50],
    8: list(range(50, 200, 50)),
    9: list(range(200, 500, 100)),
    10: list(range(500, 800, 100)),
    11: list(range(800, 1200, 100)),
    12: list(range(1200, 1400, 100)),
    13: list(range(1400, 1600, 100)),
    14: list(range(1600, 2000, 100)),
    15: list(range(2000, 2400, 100)),
}

get_new_arch_params = {
    'retina':
        {
            'sample_path_name_addition': 'with_top_performance',
            'performance_range_ind': (-3, -1),
            'used_features_csv_name': 'retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/2023-05-04-12-30-24_used_features.csv',
            'target_label_ranges': retina_target_label_ranges,
            'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl',
            'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl',
            20: {
                'samples_path': 'retina_train_test_data/all_data_20_features_with_preformance.pkl'
            },
            5: {
                'samples_path': 'retina_train_test_data/all_data_5_features_with_preformance.pkl'
            },
            10: {
                'samples_path': 'retina_train_test_data/all_data_10_features_with_preformance.pkl'
            },
        }
}

get_new_arch_params_no_modularity = {
    'retina':
        {
            'sample_path_name_addition': 'with_top_performance_no_modularity',
            'performance_range_ind': (-3, -1),
            'used_features_csv_name': 'retina_lightgbm_feature_selection/no_modularity/exp_2023-07-15-13-12-18/2023-07-15-13-12-18_5_10_used_features.csv',
            'target_label_ranges': retina_target_label_ranges,
            'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_False_meta_True_no_modularity.pkl',
            'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_False_meta_True_no_modularity.pkl',
            10: {
                'samples_path': 'retina_train_test_data/all_data_10_features_with_preformance_no_modularity.pkl'
            },
            5: {
                'samples_path': 'retina_train_test_data/all_data_5_features_with_preformance_no_modularity.pkl'
            }
        }
}

get_new_arch_params_bottom_performance = {
    'retina':
        {
            'sample_path_name_addition': 'with_bottom_performance',
            'performance_range_ind': (2, 4),
            'used_features_csv_name': 'retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/2023-05-04-12-30-24_used_features.csv',
            'target_label_ranges': retina_target_label_ranges,
            'train_path': 'retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl',
            'test_path': 'retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl',
            5: {
                'samples_path': 'retina_train_test_data/all_data_5_features_with_with_bottom_performance.pkl'
            }
        }
}

predict_techability_retina_xor_after_feature_selection = {
    'exp_path': "retina_xor/retina_3_layers/lightgbm_feature_selection",
    'exp_folder': "exp_2023-11-27-15-51-15_nice_features",
    'output_path': 'teach_archs_regression_feature_selection_results_with_preds',
    #'layers_sized': [406, 1024, 512, 254, 128, 64, 1],
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
}

predict_techability_xor_after_feature_selection = {
    'exp_path': "xor/xor_4_layers/lightgbm_feature_selection",
    'exp_folder': "exp_2023-11-16-17-38-02_nice_features",
    'output_path': 'teach_archs_regression_feature_selection_results_with_preds',
    #'layers_sized': [406, 1024, 512, 254, 128, 64, 1],
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
}

predict_techability_digits_after_feature_selection = {
    'exp_path': "digits/digits_3_layers/lightgbm_feature_selection",
    'exp_folder': "exp_2023_11_27_11_00_0_nice_features",
    'output_path': "teach_archs_regression_feature_selection_results_1kep",
    'learning_rate': 0.001,
    'num_epochs': 1000,
    'batch_size': 512,
}

predict_techability_random_feature_selection = {
    'learning_rate': 0.001,
    'num_epochs': 500,
    'batch_size': 512,
}