import argparse
from datetime import datetime as dt

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from jobs_params import baseline_predict_techability
from utils.main_utils import compute_r2
import torch.nn.functional as functional


class LinearRegression(torch.nn.Module):

    def __init__(
            self,
            input_size: int,
            activate: bool = True,
    ):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)
        self.activate = activate
        self.activation = functional.relu

    def forward(self, x,):
        predict_y = self.linear(x)
        if not self.activate:
            return predict_y
        return self.activation(predict_y)


# base_path = '/Users/noamazmon/PycharmProjects/network_modularity'
base_path = '/home/labs/schneidmann/noamaz/modularity'

parser = argparse.ArgumentParser()
parser.add_argument('--job_num', default=0)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--epochs', default=2000)

args = parser.parse_args()
job_num = int(args.job_num)

train_path = f'{base_path}/{baseline_predict_techability[job_num]["train_path"]}'
test_path = f'{base_path}/{baseline_predict_techability[job_num]["test_path"]}'
time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
name = baseline_predict_techability[job_num]["out_path"]
learning_rate = baseline_predict_techability[job_num]['learning_rate']
num_epochs = baseline_predict_techability[job_num]['num_epochs']
activate = baseline_predict_techability[job_num]['activate']
out_path = f"{base_path}/predict_teacability_results/{time_str}_{name}_baseline_lr_{learning_rate}_activate_{activate}"

with open(train_path, 'rb') as fp:
    train_data = joblib.load(fp)
with open(test_path, 'rb') as fp:
    test_data = joblib.load(fp)
data = train_data + test_data
loader = torch.utils.data.DataLoader(
    dataset=data,
    batch_size=len(data),
    shuffle=False,
)
input_size = data[0][0].shape[0]
linear_model = LinearRegression(
    input_size=input_size,
    activate=activate,
)

all_epochs_losses = []
all_epochs_r2s = []
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    r2s_train = []
    all_batches_losses = []
    for data_input, labels in loader:
        predictions = linear_model(data_input)
        loss = criterion(predictions, labels.reshape(-1, 1))
        all_batches_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        r2s_train.append(
            compute_r2(
                target=labels.reshape(-1, 1),
                output=predictions,
            ).detach().numpy()
        )
    all_epochs_losses.append(np.mean(all_batches_losses))
    all_epochs_r2s.append(np.mean(r2s_train))
    print('epoch {}, loss function {}, r2: {}'.format(epoch, np.mean(all_batches_losses), np.mean(r2s_train)))
red_df = pd.DataFrame(
    {
        'Epoch': list(range(num_epochs)),
        'r2s': all_epochs_r2s,
        'losses': all_epochs_losses,
    }
)
red_df.to_csv(f"{out_path}.csv", )
