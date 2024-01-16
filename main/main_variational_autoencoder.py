import argparse
from datetime import datetime as dt

import joblib
import pandas as pd
from torch.utils.data import DataLoader

from generative_models.variational_autoencoder import VariationalAutoencoder, train, get_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', default=0)

    args = parser.parse_args()
    job_num = int(args.job_num)
    num_features = 20
    #base_path = '/Volumes/noamaz/modularity'
    base_path = '/home/labs/schneidmann/noamaz/modularity'
    epochs = 100
    latent_dim = job_num
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    samples_path = f"{base_path}/teach_archs/retina/retina_train_test_data/train_and_test_togther.pkl"
    out_path = f"{base_path}/teach_archs/retina/retina_teach_archs_requiered_features_vea"

    if samples_path:
        with open(samples_path, 'rb') as fp:
            samples = joblib.load(fp)
    else:
        samples = get_data(
            performance_trashold=0,
            num_features=num_features,
            base_path=base_path,
        )
    loader = DataLoader(
        dataset=samples,
        batch_size=512,
        shuffle=True,
    )
    layers_size = [samples[0].shape[0], 1024, 512, 64]
    print(layers_size)
    vae = VariationalAutoencoder(
        layers_size=layers_size,
        latent_dim=latent_dim,
    )  # GPU
    vae, losses = train(vae, loader, epochs=epochs)
    red_df = pd.DataFrame(
        {
            'Epoch': list(range(epochs)),
            'losses': losses,
        }
    )
    red_df.to_csv(f"{out_path}/{time_str}_vea_model_loss_latent_dim_{latent_dim}.csv", )
    with open(f"{out_path}/{time_str}_vea_model_latent_dim_{latent_dim}.pkl", 'wb+') as fp:
        joblib.dump(vae, fp)
