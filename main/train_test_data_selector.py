from parameters.digits_parameters import digits_structural_features_full_name_vec, \
    digits_structural_features_full_name_vec_with_motifs
from parameters.retina_parameters import retina_structural_features_full_name_vec, \
    retina_structural_features_full_name_vec_with_motifs
from parameters.xor_parameters import xor_structural_features_full_name_vec, \
    xor_structural_features_full_name_vec_with_motifs


class DataSelector:
    def __init__(
            self,
            task: str,
            no_modularity: bool,
            with_motifs: bool,
    ):
        structural_features_full_name_vec_mapping = {
            'xor': xor_structural_features_full_name_vec,
            'retina': retina_structural_features_full_name_vec,
            'digits': digits_structural_features_full_name_vec,
        }
        structural_features_full_name_vec_with_motifs_mapping = {
            'xor': xor_structural_features_full_name_vec_with_motifs,
            'retina': retina_structural_features_full_name_vec_with_motifs,
            'digits': digits_structural_features_full_name_vec_with_motifs,
        }

        if task == 'xor':
            if no_modularity and with_motifs:
                train_path = 'xor_train_2023-07-31-17-45-01_adj_False_meta_True_with_motifs_no_modularity.pkl'
                test_path = 'xor_test_2023-07-31-17-45-01_adj_False_meta_True_with_motifs_no_modularity.pkl'
            elif with_motifs:
                train_path = 'xor_train_2023-07-31-17-45-01_adj_False_meta_True_with_motifs.pkl'
                test_path = 'xor_train_2023-07-31-17-45-01_adj_True_meta_True_with_motifs.pkl'
            else:
                train_path = 'xor_train_2023-04-13-14-15-53_adj_False_meta_True.pkl'
                test_path = 'xor_test_2023-04-13-14-15-53_adj_False_meta_True.pkl'
        elif task == 'retina':
            if no_modularity and with_motifs:
                train_path = 'retina_train_2023-07-01-12-35-33_adj_False_meta_True_with_motifs_no_modularity.pkl'
                test_path = 'retina_test_2023-07-01-12-35-33_adj_False_meta_True_with_motifs_no_modularity.pkl'
            elif no_modularity:
                train_path = 'retina_train_2023-04-16-15-02-58_adj_False_meta_True_no_modularity.pkl'
                test_path = 'retina_test_2023-04-16-15-02-58_adj_False_meta_True_no_modularity.pkl'
            elif with_motifs:
                train_path = 'retina_train_2023-07-01-12-35-33_adj_False_meta_True_with_motifs.pkl'
                test_path = 'retina_test_2023-07-01-12-35-33_adj_False_meta_True_with_motifs.pkl'
            else:
                train_path = 'retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl'
                test_path = 'retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl'
        elif task == 'digits':
            if no_modularity and with_motifs:
                train_path = 'digits_train_2023-08-01-14-45-08_adj_False_meta_True_with_motifs_no_modularity.pkl'
                test_path = 'digits_test_2023-08-01-14-45-08_adj_False_meta_True_with_motifs_no_modularity.pkl'
            elif with_motifs:
                train_path = 'digits_train_2023-08-01-14-45-08_adj_False_meta_True_with_motifs.pkl'
                test_path = 'digits_test_2023-08-01-14-45-08_adj_False_meta_True_with_motifs.pkl'
            else:
                train_path = 'digits_train_2023-06-26-14-00-11_adj_False_meta_True.pkl'
                test_path = 'digits_test_2023-06-26-14-00-11_adj_False_meta_True.pkl'
        if with_motifs:
            structural_features_full_name_vec = structural_features_full_name_vec_with_motifs_mapping[task]
        else:
            structural_features_full_name_vec = structural_features_full_name_vec_mapping[task]
        if no_modularity:
            modularity_index = structural_features_full_name_vec.index('modularity')
            structural_features_full_name_vec.pop(modularity_index)

        self.train_path = train_path
        self.test_path = test_path
        self.structural_features_full_name_vec = structural_features_full_name_vec
