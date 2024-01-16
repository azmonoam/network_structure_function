import os
import random

import pandas as pd

base_path = '/home/labs/schneidmann/noamaz/modularity'
successful_results_path = f'{base_path}/successful_results_4'
files = [
    file_name
    for file_name in os.listdir(successful_results_path)
]
random.shuffle(files)

pd.DataFrame(
    {
        'files': files
    }
).to_csv(f"{base_path}/successful_results_4_pkl_names.csv", index=False)
