import argparse
from pathlib import Path

from get_network_teachability_data.get_networks_techability_data_parallel_specipic_features import \
    GetNetworkTeachabilityAllFeatures

if __name__ == '__main__':
    extended_results = False
    add_motifs = True
    new_models = True
    add_normlized = True
    base_path = '/home/labs/schneidmann/noamaz/modularity/'
    # base_path = '/Volumes/noamaz/modularity'

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_folder', default='teach_archs_results')
    parser.add_argument('--models_folder', default='teach_archs_models')
    parser.add_argument('--base_folder', default='xor/xor_4_layers_6_5_3')
    parser.add_argument('--n_threads', default='1')
    parser.add_argument('--from_base', default='0')
    parser.add_argument('--ephoc_folder', default='')
    parser.add_argument('--task', default='xor')
    parser.add_argument('--job_num', default='1')

    args = parser.parse_args()
    base_folder = args.base_folder
    res_folder = args.res_folder
    models_folder = f"{base_path}/{base_folder}/{args.models_folder}"
    num_cores = int(args.n_threads)
    job_num = int(args.job_num)
    from_base = bool(int(args.from_base))
    ephoc_folder = args.ephoc_folder
    task = args.task
    folder = f"{base_path}/{base_folder}/{res_folder}"
    print(f"-- using {num_cores} cores --")

    epoch = None
    if task == 'digits':
        epoch = [200, 400, 1000, 1400, 2000]
    elif task == 'xor':
        if job_num != 0:
            add_normlized = False
            bases = {
                1: ('2023-09-18-17-00-39_J11_P1_n', 'ergm_results'),
                2: ('good_archs', 'requiered_features_genetic_models'),
                3: ('good_archs_1s', 'requiered_features_genetic_models'),
            }
            base_folder = f"{args.base_folder}/{bases[job_num][1]}/4_features/{bases[job_num][0]}"
            res_folder = args.res_folder
            models_folder = f"{base_path}/{base_folder}/{args.models_folder}"
            folder = f"{base_path}/{base_folder}/{res_folder}"
        epoch = 6000
    elif task == 'retina':
        if job_num != 0:
            add_normlized = False
            bases = {
                1: '2023-09-11-14-09-34_J9_P0',
                2: '2023-09-11-15-49-34_J80_P1',
                3: '2023-09-11-16-37-02_J7_P2',
                4: '2023-09-11-16-40-30_J7_P3',
                5: '2023-09-11-16-42-08_J8_P3',
                6: '2023-09-12-10-54-55_J7_P4',
                7: 'good_archs'
            }
            base_folder = f"{args.base_folder}/requiered_features_genetic_models/6_features/{bases[job_num]}"
            res_folder = args.res_folder
            models_folder = f"{base_path}/{base_folder}/{args.models_folder}"
            folder = f"{base_path}/{base_folder}/{res_folder}"
        epoch = 6000

    existing_exps_analysis_csv_name = None

    res_out_path = f'{base_path}/{base_folder}/first_analysis_results'
    get_net = GetNetworkTeachabilityAllFeatures(
        num_cores=num_cores,
        folder=folder,
        extended_results=extended_results,
        add_motifs=add_motifs,
        new_models=new_models,
        models_folder=models_folder,
        res_folder=res_folder,
        add_normlized=add_normlized,
    )
    first_analysis_df = get_net.combine_all_data(
        existing_exps_analysis_full_path=None,
        epoch=epoch,
    )

    first_analysis_df.to_csv(f'{res_out_path}/{get_net.csv_name}_all_features.csv')
    if not from_base:
        all_res_no_duplicates = first_analysis_df.drop_duplicates(get_net.subset)
        all_res_no_duplicates.to_csv(f'{res_out_path}/{get_net.csv_name}_no_duplicates_all_features.csv')
