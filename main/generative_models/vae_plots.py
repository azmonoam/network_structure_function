import joblib
import pandas as pd
from torch.utils.data import DataLoader
import torch
from generative_models.variational_autoencoder import VariationalAutoencoder, train, get_data
import matplotlib.pyplot as plt
import numpy as np
COLORS = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897', '#f6bd60', '#e76f51', '#2a9d8f', "#553939", "#9080ff", "#7d8f69"]

num_features = 20
performance_trashold = 930
base_path = '/Volumes/noamaz/modularity'
train_path = f'{base_path}/teach_archs/retina/retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl'
test_path = f'{base_path}/teach_archs/retina/retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl'
used_features_csv_name = f'{base_path}/teach_archs/retina/retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/2023-05-04-12-30-24_used_features.csv'
with open(f"{train_path}", 'rb') as fp:
    train_data = joblib.load(fp)
with open(f"{test_path}", 'rb') as fp:
    test_data = joblib.load(fp)
with open(
        f"{base_path}/teach_archs/retina/retina_teach_archs_requiered_features_kernel_dist/20_features/2023-06-01-14-24-51_target_label_ranges.pkl",
        'rb') as fp:
    target_label_ranges = joblib.load(fp)
    """
    plt.plot(range(10, epochs), losses[10:])
    plt.show()

    test = random.sample(samples, k=1)
    e = vae.encoder(test[0])
    d  = vae.decoder(e)

    latent_sapce_x = []
    latent_sapce_y = []
    latent_space_points = []
    for s in samples:
        x, y = vae.encoder(s)
        x = x.item()
        y = y.item()
        latent_sapce_x.append(x)
        latent_sapce_y.append(y)
        latent_space_points.append([x, y])
    plt.scatter(latent_sapce_x, latent_sapce_y)
    plt.show()

    latent_space_dist = KernelDensity().fit(latent_space_points)
    sample = latent_space_dist.sample()
    new_vac = vae.decoder(torch.from_numpy(sample.flatten()).to(torch.float32))
"""
all_data = train_data + test_data
selected_features_df = pd.read_csv(f"{used_features_csv_name}").drop("Unnamed: 0", axis=1)
selected_features = selected_features_df[selected_features_df.sum(axis=1) == num_features]
mask_tensor = torch.tensor(selected_features.iloc[0]).to(torch.bool)

samples = [
    (torch.masked_select(sample, mask_tensor), performance)
    for sample, performance in all_data
]
latent_by_performances = {
    (min_val, max_val): {'x': [], 'y': []}
    for min_val, max_val in target_label_ranges
}
for s, performance in samples:
    x, y = vae.encoder(s)
    x = x.item()
    y = y.item()
    for min_val, max_val in latent_by_performances.keys():
        if min_val <= performance <= max_val:
            latent_by_performances[(min_val, max_val)]['x'].append(x)
            latent_by_performances[(min_val, max_val)]['y'].append(y)
for i, (k, v) in enumerate(latent_by_performances.items()):
    plt.scatter(v['x'], v['y'], c=COLORS[i], label=k)
plt.title('latent space by performance')
plt.xlabel('latent space - x coordinates')
plt.ylabel('latent space - y coordinates')
plt.legend()
plt.show()
for i, (k, v) in enumerate(latent_by_performances.items()):
    plt.scatter(np.mean(v['x']), np.mean(v['y']), c=COLORS[i], label=(round(k[0]/1000
                                                                            ,3), round(k[1]/1000,3)))
plt.title('mean latent space by performance')
plt.xlabel('latent space - x coordinates')
plt.ylabel('latent space - y coordinates')
plt.legend()
plt.show()

target_label_ranges_equal_bins = np.linspace(target_label_ranges[0][0], target_label_ranges[-1][-1], 10)
latent_by_performances_equal_bins = {
    (target_label_ranges_equal_bins[i], target_label_ranges_equal_bins[i+1]): {'x': [], 'y': []}
    for i in range(len(target_label_ranges_equal_bins)-1)}