import argparse
from pathlib import Path

from get_network_teachability_data.get_networks_techability_data_parallel import GetNetworkTeachabilityData
from get_network_teachability_data.get_networks_techability_data_parallel_molti_ep import \
    GetNetworkTeachabilityDataMultiEph
from get_network_teachability_data.get_networks_techability_data_parallel_with_base import \
    GetNetworkTeachabilityDataWithBase
from get_network_teachability_data.get_networks_techability_data_parallel_with_base_molti_ep import \
    GetNetworkTeachabilityDataWithBaseMultiEph

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
    parser.add_argument('--base_folder', default='retina/retina_4_layers_6_4_2')
    parser.add_argument('--n_threads', default='1')
    parser.add_argument('--from_base', default='0')
    parser.add_argument('--ephoc_folder', default='')
    parser.add_argument('--task', default='digits')
    parser.add_argument('--job_num', default='0')

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
        add_normlized = False
        bases = {
            0: '',
            1: 'ergm/3_features/per_dim_results',
            2: 'requiered_features_genetic_models/3_features/good_archs/per_dim_results/'
        }
        base_folder = f"{args.base_folder}/{bases[job_num]}"
        models_folder = f"{base_path}/{base_folder}/{args.models_folder}"
        res_folder = f"{bases[job_num]}"
        folder = f"{base_path}/{base_folder}/{args.res_folder}"
        epoch = [2000]
    elif task == 'xor':
        add_normlized = False
        bases = {
            0: '',
            1: 'ergm/5_features/per_dim_results',
            2: 'requiered_features_genetic_models/5_features/good_archs/per_dim_results/'
        }
        base_folder = f"{args.base_folder}/{bases[job_num]}"
        models_folder = f"{base_path}/{base_folder}/{args.models_folder}"
        res_folder = f"{bases[job_num]}"
        folder = f"{base_path}/{base_folder}/{args.res_folder}"
        epoch = [5000]
    elif task == 'retina_xor':
        if job_num != 0:
            add_normlized = False
            bases = {
                0: '',
                1: 'ergm/6_features/per_dim_results',
                2: 'requiered_features_genetic_models/6_features/good_archs/per_dim_results/'
            }
            base_folder = f"{args.base_folder}/{bases[job_num]}"
            models_folder = f"{base_path}/{base_folder}/{args.models_folder}"
            res_folder = f"{bases[job_num]}"
            folder = f"{base_path}/{base_folder}/{args.res_folder}"
        epoch = [5000]

    existing_exps_analysis_csv_name = None

    if existing_exps_analysis_csv_name is not None:
        existing_exps_analysis_full_path = f"{base_path}/{base_folder}/{existing_exps_analysis_csv_name}"
    else:
        existing_exps_analysis_full_path = None

    if not isinstance(epoch, list):
        res_out_path = f'{base_path}/{base_folder}'

        if from_base:
            base_file_path = f"{base_path}/{base_folder}/2023-08-07-16-15-08__first_analsis_general_no_duplicates.csv"
            folder = f"{base_path}/{base_folder}/{ephoc_folder}/{res_folder}"
            res_out_path = f'{base_path}/{base_folder}/{ephoc_folder}'

            get_net = GetNetworkTeachabilityDataWithBase(
                num_cores=num_cores,
                folder=folder,
                extended_results=extended_results,
                add_motifs=add_motifs,
                new_models=new_models,
                models_folder=models_folder,
                res_folder=res_folder,
                base_file_path=base_file_path,
                ephoc_folder=ephoc_folder,
                add_normlized=add_normlized,
            )
        else:
            get_net = GetNetworkTeachabilityData(
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
            existing_exps_analysis_full_path=existing_exps_analysis_full_path,
            epoch=epoch,
        )

        first_analysis_df.to_csv(f'{res_out_path}/{get_net.csv_name}.csv')
        if not from_base:
            all_res_no_duplicates = first_analysis_df.drop_duplicates(get_net.subset)
            all_res_no_duplicates.to_csv(f'{res_out_path}/{get_net.csv_name}_no_duplicates.csv')
    else:
        res_out_path = f'{base_path}/{base_folder}/first_analysis_results'
        res_out_folder = Path(res_out_path)
        res_out_folder.mkdir(exist_ok=True)
        if from_base:
            base_file_path = f"{base_path}/{base_folder}/2023-08-20-12-33-37_first_analsis_general_no_duplicates.csv"
            get_net = GetNetworkTeachabilityDataWithBaseMultiEph(
                num_cores=num_cores,
                folder=folder,
                extended_results=extended_results,
                add_motifs=add_motifs,
                new_models=new_models,
                models_folder=models_folder,
                res_folder=res_folder,
                base_file_path=base_file_path,
                add_normlized=add_normlized,
            )
            first_analysis_dfs = get_net.combine_all_data(
                existing_exps_analysis_full_path=existing_exps_analysis_full_path,
                epoch=epoch,
            )
        else:
            get_net = GetNetworkTeachabilityDataMultiEph(
                num_cores=num_cores,
                folder=folder,
                extended_results=extended_results,
                add_motifs=add_motifs,
                new_models=new_models,
                models_folder=models_folder,
                res_folder=res_folder,
                add_normlized=add_normlized,
            )
            first_analysis_dfs = get_net.combine_all_data(
                existing_exps_analysis_full_path=existing_exps_analysis_full_path,
                epoch=epoch,
            )
        for first_analysis_df, e in zip(first_analysis_dfs, epoch):
            first_analysis_df.to_csv(f'{res_out_path}/{get_net.csv_name}_{e}_ep.csv')
            all_res_no_duplicates = first_analysis_df.drop_duplicates(get_net.subset)
            all_res_no_duplicates.to_csv(f'{res_out_path}/{get_net.csv_name}_{e}_ep_no_duplicates.csv')
