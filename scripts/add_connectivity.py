import pandas as pd

base_path = '/home/labs/schneidmann/noamaz/modularity'

path = f'{base_path}/2023-02-05-18-06-48_first_analysis.csv'
first_analysis_df = pd.read_csv(f"{path}").drop("Unnamed: 0", axis=1)
first_analysis_df['connectivity'] = [
    float(name.split('_')[-2])
    for name in first_analysis_df['exp_name'].to_list()
]
out_path = f'{base_path}/2023-02-05-18-06-48_first_analysis_with_connectivity.csv'

first_analysis_df.to_csv(out_path)
