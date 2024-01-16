from typing import List, Tuple, Optional, Union

import joblib
import lightgbm
import torch
import numpy as np

from sklearn import metrics


def decompose_tensor_list(
        tensor_list: List[Tuple[torch.Tensor, torch.Tensor]],
        ind_to_drop: Optional[List[int]]
) -> Tuple[np.ndarray, List[int]]:
    inputs = []
    labels = []
    for ind, (x, y) in enumerate(tensor_list):
        x = x.numpy()
        if ind_to_drop is not None:
            x = np.delete(x, ind_to_drop)
        inputs.append(x)
        labels.append(np.float32(y.item()))
    inputs = np.stack(inputs, axis=0)
    return inputs, labels


def _get_data(
        ind_to_drop: Optional[List[int]],
        train_path: Optional[str],
        test_path: Optional[str],
        train_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        test_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],

) -> Tuple[np.ndarray, List[float], np.ndarray, List[float]]:
    if train_data is None and test_data is None:
        try:
            with open(train_path, 'rb') as fp:
                train_data = joblib.load(fp)
            with open(test_path, 'rb') as fp:
                test_data = joblib.load(fp)
        except Exception as e:
            raise Exception(f'Error: could not open {train_path} or {test_path}')
    test_inputs, test_labels = decompose_tensor_list(
        tensor_list=test_data,
        ind_to_drop=ind_to_drop,
    )
    train_inputs, train_labels = decompose_tensor_list(
        tensor_list=train_data,
        ind_to_drop=ind_to_drop,
    )
    return test_inputs, test_labels, train_inputs, train_labels


def tree_regression(
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        train_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        test_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        reg_alpha: float = 0.0,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        subsample_for_bin: int = 200000,
        ind_to_drop: Optional[List[int]] = None,
        feature_names: Union[str, List[str]] = 'auto',
) -> Tuple[lightgbm.LGBMRegressor, float, float]:
    test_inputs, test_labels, train_inputs, train_labels = _get_data(
        train_path=train_path,
        test_path=test_path,
        train_data=train_data,
        test_data=test_data,
        ind_to_drop=ind_to_drop,
    )
    model = lightgbm.LGBMRegressor(
        reg_alpha=reg_alpha,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample_for_bin=subsample_for_bin,
        force_col_wise=True
    )
    model.fit(
        train_inputs,
        train_labels,
        eval_metric='mean_squared_error',
        eval_set=[(test_inputs, test_labels), (train_inputs, train_labels)],
        eval_names=['test', 'train'],
        feature_name=feature_names,
    )
    model.score(test_inputs, test_labels)
    predicted_train_inputs = model.predict(train_inputs)
    predicted_test_inputs = model.predict(test_inputs)
    train_r2 = metrics.r2_score(predicted_train_inputs, train_labels)
    test_r2 = metrics.r2_score(predicted_test_inputs, test_labels)
    print('Training r2 score {:.4f}'.format(train_r2))
    print('Testing r2 score {:.4f}'.format(test_r2))
    return model, train_r2, test_r2
