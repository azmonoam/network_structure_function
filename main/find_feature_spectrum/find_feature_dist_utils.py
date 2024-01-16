import random
from typing import List, Tuple, Optional

import joblib
import pandas as pd
import torch

NodeType = Tuple[int, int]


def get_selected_feature_names(
        used_features_csv_name: str,
        num_features: int,
) -> List[str]:
    selected_features_df = pd.read_csv(used_features_csv_name).drop("Unnamed: 0",
                                                                    axis=1)
    selected_features = selected_features_df[selected_features_df.sum(axis=1) == num_features]
    return selected_features[selected_features == 1].dropna(axis=1).columns


def get_features_of_samples(
        task_path: str,
        train_path: str,
        test_path: str,
        used_features_csv_name: str,
        num_features: int,
        target_samples_path: Optional[str] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    with open(f"{task_path}/{train_path}", 'rb') as fp:
        train_data = joblib.load(fp)
    with open(f"{task_path}/{test_path}", 'rb') as fp:
        test_data = joblib.load(fp)
    all_data = train_data + test_data
    random.shuffle(all_data)

    selected_features_df = pd.read_csv(f"{used_features_csv_name}").drop("Unnamed: 0", axis=1)
    selected_features = selected_features_df[selected_features_df.sum(axis=1) == num_features]
    mask_tensor = torch.tensor(selected_features.iloc[0]).to(torch.bool)
    samples = [
        (torch.masked_select(sample, mask_tensor), performance)
        for sample, performance in all_data
    ]
    if target_samples_path:
        with open(target_samples_path, 'wb+') as fp:
            joblib.dump(samples, fp)
    return samples
