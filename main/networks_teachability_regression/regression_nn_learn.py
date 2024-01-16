from datetime import datetime as dt
from typing import List, Optional, Tuple
from torchmetrics.regression import MeanAbsolutePercentageError
import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from prepare_regression_adjacency_data import PrepareRegressionAdjacencyData
from networks_teachability_regression.regression_model import NN
from networks_teachability_regression.small_regression_model import SmallNN

from utils.main_utils import compute_r2, weights_init_uniform_rule


def get_data(
        base_path: str,
        results_csv_name: Optional[str],
        results_model_path: Optional[str],
        label_name: str,
        increase_label_scale: bool = True,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        consider_meta_data: bool = True,
        consider_adj_mat: bool = True,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    if train_path and test_path:
        try:
            with open(train_path, 'rb') as fp:
                train_data = joblib.load(fp)
            with open(test_path, 'rb') as fp:
                test_data = joblib.load(fp)
        except Exception as e:
            raise Exception(f'Error: could not open {train_path} or {test_path}')
    else:
        prepare_adjacency_data = PrepareRegressionAdjacencyData(
            base_path=base_path,
            results_csv_name=results_csv_name,
            results_model_path=results_model_path,
            label_name=label_name,
            increase_label_scale=increase_label_scale,
            consider_meta_data=consider_meta_data,
            consider_adj_mat=consider_adj_mat,
        )
        train_data, test_data = prepare_adjacency_data.create_test_and_train_data()
    return train_data, test_data


def regression_lnn_learning(
        layers_sized: List[int],
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        test_data: List[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int,
        learning_rate: float = 1e-5,
        batch_size: int = 1024,
        test_every: int = 100,
        sufix: str = '',
        output_path: str = '',
        task: str = '',
        exp_name: str = '',
        save_preds: bool = False,
        save_model: bool = True,
):
    time_str = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
    if exp_name == '':
        exp_name = time_str
    out_path = f"{output_path}/{task}_{exp_name}_lr_{learning_rate}_bs_{batch_size}_output{sufix}"

    if layers_sized[0] != train_data[0][0].shape[0]:
        layers_sized[0] = train_data[0][0].shape[0]

    USE_CUDA = torch.cuda.is_available()
    print('******* Running on {} *******'.format('CUDA' if USE_CUDA else 'CPU'), flush=True)

    if USE_CUDA:
        if len(layers_sized) > 3:
            model = NN(
                use_cuda=USE_CUDA,
                layers_sized=layers_sized
            ).cuda()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            model = SmallNN(
                use_cuda=USE_CUDA,
                layers_sized=layers_sized
            ).cuda()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        dtype = torch.float
        train_data = [(x.to(device='cuda:0', dtype=dtype), y.to(device='cuda:0', dtype=dtype)) for x, y in train_data]
        test_data = [(x.to(device='cuda:0', dtype=dtype), y.to(device='cuda:0', dtype=dtype)) for x, y in test_data]
        mean_abs_percentage_error = MeanAbsolutePercentageError().to(device='cuda:0')
    else:
        if len(layers_sized) > 3:
            model = NN(
                use_cuda=USE_CUDA,
                layers_sized=layers_sized
            )
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            model = SmallNN(
                use_cuda=USE_CUDA,
                layers_sized=layers_sized
            )
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        mean_abs_percentage_error = MeanAbsolutePercentageError()

    # initialize weights
    model.apply(weights_init_uniform_rule)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    all_epochs_train_losses = []
    all_epochs_test_losses = []
    all_epochs_r2s_test = []
    all_epochs_r2s_train = []
    all_epochs_mae_test = []
    all_epochs_mae_train = []
    all_epochs_train_losses_no_increase = []
    all_epochs_test_losses_no_increase = []
    all_epochs_mape_test = []
    all_epochs_mape_train = []
    criterion = torch.nn.MSELoss()
    best_model = None
    best_mape_train = 100000
    best_mape_test = 100000
    best_train_mse = 100000
    best_test_mse = 100000
    best_test_r2 = None
    best_train_r2 = None
    best_ephoc = None
    for epoch in range(epochs):
        r2s_train = []
        all_batches_losses = []
        all_batches_losses_no_increse = []
        all_batches_train_mae = []
        all_batches_train_mape = []
        for train_input, train_labels in train_loader:
            predictions = model(train_input)
            loss = criterion(predictions, train_labels.reshape(-1, 1))
            all_batches_train_mae.append(
                (loss.item() ** 0.5) / (10 ** 6)
            )
            all_batches_losses.append(loss.item())
            all_batches_losses_no_increse.append(
                criterion(
                    predictions / 1000, train_labels.reshape(-1, 1) / 1000
                ).item()
            )
            all_batches_train_mape.append(
                mean_abs_percentage_error(
                    predictions, train_labels.reshape(-1, 1)
                ).item()
            )
            if np.isnan(all_batches_losses[-1]):
                raise Exception("Error: train loss is nan, exploding gradients or bug")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r2s_train.append(
                compute_r2(
                    target=train_labels.reshape(-1, 1),
                    output=predictions,
                ).detach().cpu().numpy()
            )
        all_epochs_train_losses.append(np.mean(all_batches_losses))
        all_epochs_r2s_train.append(np.mean(r2s_train))
        all_epochs_mae_train.append((np.mean(all_batches_train_mae)))
        all_epochs_train_losses_no_increase.append((np.mean(all_batches_losses_no_increse)))
        all_epochs_mape_train.append(np.mean(all_batches_train_mape))
        if epoch % test_every == 0:
            r2s_test = []
            all_batches_test_losses = []
            all_batches_test_mae = []
            all_batches_test_losses_no_increase = []
            all_batches_test_mape = []
            for test_input, test_label in test_loader:
                test_outputs = model(test_input)
                r2s_test.append(
                    compute_r2(
                        target=test_label.reshape(-1, 1),
                        output=test_outputs,
                    ).detach().cpu().numpy()
                )
                test_loss = criterion(
                    test_outputs, test_label.reshape(-1, 1)
                ).item()
                all_batches_test_losses.append(
                    test_loss
                )
                all_batches_test_mae.append(
                    (test_loss ** 0.5) / (10 ** 6)
                )
                all_batches_test_losses_no_increase.append(
                    criterion(
                        test_outputs / 1000, test_label.reshape(-1, 1) / 1000
                    ).item()
                )
                all_batches_test_mape.append(
                    mean_abs_percentage_error(
                        test_outputs, test_label.reshape(-1, 1)
                    ).item()
                )
            all_epochs_test_losses.append(np.mean(all_batches_test_losses))
            all_epochs_r2s_test.append(np.mean(r2s_test))
            all_epochs_mae_test.append((np.mean(all_batches_test_mae)))
            all_epochs_test_losses_no_increase.append((np.mean(all_batches_test_losses_no_increase)))
            all_epochs_mape_test.append((np.mean(all_batches_test_mape)))
            if all_epochs_mape_test[-1] < best_mape_test:
                best_model = model
                best_mape_train = all_epochs_mape_train[-1]
                best_mape_test = all_epochs_mape_test[-1]
                best_train_mse = all_epochs_mae_train[-1]
                best_test_mse = all_epochs_mae_test[-1]
                best_test_r2 = all_epochs_r2s_test[-1]
                best_train_r2 = all_epochs_r2s_train[-1]
                best_ephoc = epoch
    red_df = pd.DataFrame(
        {
            'Epoch': list(range(epochs)),
            'r2s train': all_epochs_r2s_train,
            'r2s test': all_epochs_r2s_test,
            'losses train': all_epochs_train_losses,
            'losses test': all_epochs_test_losses,
            'losses train no increase': all_epochs_train_losses_no_increase,
            'losses test no increase': all_epochs_test_losses_no_increase,
            'mae train no increase': all_epochs_mae_train,
            'mae test no increase': all_epochs_mae_test,
            'mape train': all_epochs_mape_train,
            'mape test': all_epochs_mape_test,
        }
    )
    best_res = pd.DataFrame(
        {
            'best ephoc': [best_ephoc],
            'best mape train': [best_mape_train],
            'best mape test': [best_mape_test],
            'best mse train': [best_train_mse],
            'best mse test': [best_test_mse],
            'best r2 train': [best_train_r2],
            'best r2 test': [best_test_r2],
        }
    )
    red_df.to_csv(f"{out_path}.csv", )
    best_res.to_csv(f"{out_path}_best.csv", )
    if save_model:
        with open(f"{out_path}_model.pkl", 'wb+') as fp:
            joblib.dump(model, fp)
        with open(f"{out_path}_best_model.pkl", 'wb+') as fp:
            joblib.dump(best_model, fp)
    if save_preds:
        res = pd.DataFrame()
        test_pred = []
        test_label_no_increse = []
        for test_input, test_label in test_loader:
            test_outputs = best_model(test_input)
            test_pred += (test_outputs.reshape(-1).detach() / 1000).tolist()
            test_label_no_increse += (test_label / 1000).tolist()
        res['test_pred'] = test_pred
        res['test_label'] = test_label_no_increse
        res.to_csv(f"{out_path}_prediction_results.csv")
    if save_model:
        with open(f"{out_path}_best_model_cpu.pkl", 'wb+') as fp:
            joblib.dump(best_model.cpu(), fp)
