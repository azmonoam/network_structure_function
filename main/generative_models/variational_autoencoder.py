import random
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

from parameters.general_paramters import RANDOM_SEED

random.seed(RANDOM_SEED)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalEncoder(nn.Module):
    def __init__(
            self,
            layers_size: List[int],
            latent_dim: int,
    ):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(layers_size[0], layers_size[1])
        self.linear2 = nn.Linear(layers_size[1], layers_size[2])
        self.linear3 = nn.Linear(layers_size[2], layers_size[3])
        self.linear4 = nn.Linear(layers_size[3], latent_dim)

        self.N = torch.distributions.Normal(0, 1)
#        self.N.loc = self.N.loc  # hack to get sampling on the GPU
 #       self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mu = self.linear4(x)
        sigma = torch.exp(self.linear4(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(
            self,
            layers_size: List[int],
            latent_dim: int,
    ):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, layers_size[3])
        self.linear2 = nn.Linear(layers_size[3], layers_size[2])
        self.linear3 = nn.Linear(layers_size[2], layers_size[1])
        self.linear4 = nn.Linear(layers_size[1], layers_size[0])

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = self.linear4(z)
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            layers_size: List[int],
            latent_dim: int,
    ):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = VariationalEncoder(
            layers_size=layers_size,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            layers_size=layers_size,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    losses = []
    for epoch in range(epochs):
        ephoc_losses = []
        for x in data:
          #  x = x.to(device)  # GPU
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            ephoc_losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f'Epoch: {epoch}, loss: {np.mean(ephoc_losses)}')
        losses.append(np.mean(ephoc_losses))
        if np.isnan(losses[-1]):
            raise Exception("Error: train loss is nan, exploding gradients or bug")
    return autoencoder, losses


def get_data(
        num_features: int,
        base_path: str,
        performance_trashold: float,
) -> List[torch.Tensor]:
    train_path = f'{base_path}/teach_archs/retina/retina_train_test_data/retina_train_2023-04-16-15-02-58_adj_False_meta_True.pkl'
    test_path = f'{base_path}/teach_archs/retina/retina_train_test_data/retina_test_2023-04-16-15-02-58_adj_False_meta_True.pkl'
    used_features_csv_name = f'{base_path}/teach_archs/retina/retina_lightgbm_feature_selection/exp_2023-04-25-12-22-31/2023-05-04-12-30-24_used_features.csv'
    with open(f"{train_path}", 'rb') as fp:
        train_data = joblib.load(fp)
    with open(f"{test_path}", 'rb') as fp:
        test_data = joblib.load(fp)
    all_data = train_data + test_data
    random.shuffle(all_data)

    selected_features_df = pd.read_csv(f"{used_features_csv_name}").drop("Unnamed: 0", axis=1)
    selected_features = selected_features_df[selected_features_df.sum(axis=1) == num_features]
    mask_tensor = torch.tensor(selected_features.iloc[0]).to(torch.bool)

    samples = [
        torch.masked_select(sample, mask_tensor)
        for sample, performance in all_data
        if performance.item() >= performance_trashold
    ]
    return samples
