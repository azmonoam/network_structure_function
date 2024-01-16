import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from tqdm import tqdm
from matplotlib.patches import Patch

base_path = '/home/labs/schneidmann/noamaz/modularity'
base_path = '/Volumes/noamaz/modularity'
local_plot_path = "/Users/noamazmon/PycharmProjects/network_modularity/plots/retina_from_label_to_arch_to_label/vea_latent_space"
ret_path = f'{base_path}/teach_archs/retina'
vae_model_path = f'{ret_path}/retina_teach_archs_requiered_features_vea/models/2023-06-04-12-07-38_vea_model_latent_dim_2.pkl'
predictive_model_path = f'{ret_path}/retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/teach_archs_regression_feature_selection_results/2023-05-22-11-35_500_eph/retina_2023-05-22-11-45-00_lr_0.001_bs_512_output_meta_only_20_features_model_cpu.pkl'
samples_path = f'{ret_path}/retina_train_test_data/all_data_20_features_with_preformance.pkl'
target_label_path = f"{ret_path}/retina_teach_archs_requiered_features_kernel_dist/20_features/2023-06-01-14-24-51_target_label_ranges.pkl"
latent_dim = vae_model_path.split('_')[-1].split('.pkl')[0]

if __name__ == '__main__':
    with open(vae_model_path, 'rb') as fp:
        vae = joblib.load(fp)

    with open(predictive_model_path, 'rb') as fp:
        predictive_model = joblib.load(fp)

    with open(target_label_path, 'rb') as fp:
        target_label_ranges = joblib.load(fp)

    with open(samples_path, 'rb') as fp:
        samples = joblib.load(fp)

    labels_range = (target_label_ranges[-2][0], target_label_ranges[-1][1])
    top_samples = [s[0] for s in samples if labels_range[0] <= s[1] <= labels_range[1]]
    list_of_lists = [[] for i in range(int(latent_dim))]
    for s in top_samples:
        encoded_vec = vae.encoder(s)
        for i, element in enumerate(encoded_vec):
            list_of_lists[i].append(element)

    required_performance_min = labels_range[0] / 1000
    required_performance_max = labels_range[1] / 1000
    required_performance_diff = required_performance_max - required_performance_min
    top_samples_in_latent_space_df = pd.DataFrame(list_of_lists).T.astype(float)
    means = top_samples_in_latent_space_df.mean(axis=0)
    covs = top_samples_in_latent_space_df.cov()
    num_experiments = 1500
    res_df = pd.DataFrame()
    multivariate_gau = multivariate_normal(mean=means, cov=covs)
    for i in tqdm(range(num_experiments)):
        random_sample = torch.tensor(multivariate_gau.rvs()).to(torch.float32)
        decoded_random_sample = vae.decoder(random_sample)
        normed_new_sample = torch.clone(decoded_random_sample)
        for i in range(6, 13):
            normed_new_sample[i] = torch.round(decoded_random_sample[i] * 2) / 2
        for i in range(13, 20):
            normed_new_sample[i] = torch.round(decoded_random_sample[i])
        prediction = predictive_model(normed_new_sample).item()
        res_dict = {
            'prediction': prediction / 1000,
            'required_performance_min': required_performance_min,
            'required_performance_max': required_performance_max,
            'is_within_required_performance_range':
                (labels_range[0] <= prediction <= labels_range[1]),
        }
        res_df = pd.concat([res_df, pd.DataFrame(res_dict, index=[0], )], ignore_index=True)

    prediction_no_outliers = res_df['prediction'][res_df['prediction'].between(
        res_df['prediction'].quantile(0.01),
        res_df['prediction'].quantile(0.99)
    )].sort_values()

    fig = plt.figure()
    ax = fig.add_subplot(111, )
    n, bins, patches = ax.hist(
        x=prediction_no_outliers,
        bins=np.arange(
            required_performance_min - (7 * required_performance_diff),
            required_performance_max + (8 * required_performance_diff),
            required_performance_diff,
        ),
        color='#4F6272',
    )
    patches[7].set_facecolor('#B7C3F3')
    h = [Patch(facecolor='#B7C3F3', label='Color Patch'), patches]
    ax.legend(h, ['predictions', 'target bin'])
    plt.xlabel('predicted mean performance')
    plt.xticks([round(b,3) for b in bins], rotation=70)
    plt.title(
        f"Predicted mean performance of architectures with structural features drown from a multivariate gaussian"
        f" distribution of the VAE's {latent_dim}D latent space of the target performance data "
        f"(target {round(required_performance_min, 4)}-{round(required_performance_max, 4)})",
        wrap=True,
    )
    plt.tight_layout(pad =3 )
    plt.savefig(f"{local_plot_path}/predicted_mean_performance_of_arch_from_{latent_dim}d_latent_multi_gaussian.png")
    plt.show()
