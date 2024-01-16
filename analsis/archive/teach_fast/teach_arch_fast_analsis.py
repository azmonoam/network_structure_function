from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analsis.analsis_utils.utils import COLORS

results_path = '/xor_teach_fast_res'
results_path = "/Volumes/noamaz/modularity/xor_teach_fast_res"

out_path = '/plots/xor_tech_fast'
time = dt.now().strftime("%Y-%m-%d")

files_lr_01_ep_10 = ['2023-02-19-10-53_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
                     '2023-02-19-10-53_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
                     '2023-02-19-10-53_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
                     '2023-02-19-10-53_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
                     '2023-02-19-10-53_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
                     '2023-02-19-10-53_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
                     '2023-02-19-10-53_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
                     '2023-02-19-10-53_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv']
files_lr_01_ep_15 = ['2023-02-19-10-53_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
                     '2023-02-19-10-53_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
                     '2023-02-19-11-46_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
                     '2023-02-19-10-53_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
                     '2023-02-19-10-53_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
                     '2023-02-19-10-53_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv']
files_lr_03_ep_10 = ['2023-02-19-10-53_lr_0.03_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
                     '2023-02-19-10-53_lr_0.03_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
                     '2023-02-19-11-46_lr_0.03_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
                     '2023-02-19-10-53_lr_0.03_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
                     '2023-02-19-10-53_lr_0.03_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
                     '2023-02-19-10-53_lr_0.03_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
                     '2023-02-19-10-53_lr_0.03_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
                     '2023-02-19-10-53_lr_0.03_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv']
files_lr_03_ep_12 = ['2023-02-19-10-53_lr_0.03_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
                     '2023-02-19-10-53_lr_0.03_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
                     '2023-02-19-10-53_lr_0.03_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
                     '2023-02-19-10-53_lr_0.03_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
                     '2023-02-19-10-53_lr_0.03_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
                     '2023-02-19-10-53_lr_0.03_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
                     '2023-02-19-10-53_lr_0.03_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
                     '2023-02-19-10-53_lr_0.03_ep_12_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv']
files_lr_1_ep_10 = ['2023-02-19-10-53_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
                    '2023-02-19-10-53_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
                    '2023-02-19-10-53_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
                    '2023-02-19-10-53_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
                    '2023-02-19-10-53_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
                    '2023-02-19-10-53_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
                    '2023-02-19-10-53_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
                    '2023-02-19-10-53_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv']
files_lr_005_ep_12 = [
    '2023-02-19-18-29_lr_0.05_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
    '2023-02-19-18-29_lr_0.05_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
    '2023-02-19-18-29_lr_0.05_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
    '2023-02-19-18-29_lr_0.05_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
    '2023-02-19-18-29_lr_0.05_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
    '2023-02-19-18-28_lr_0.05_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
    '2023-02-19-18-29_lr_0.05_ep_12_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
    '2023-02-19-18-29_lr_0.05_ep_12_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv',
]
files_lr_005_ep_10 = [
    '2023-02-19-18-29_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
    '2023-02-19-18-29_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
    '2023-02-19-18-29_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
    '2023-02-19-18-29_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
    '2023-02-19-18-29_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
    '2023-02-19-18-29_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
    '2023-02-19-18-29_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
    '2023-02-19-18-29_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv',
]
files_lr_007_ep_10 = [
    '2023-02-19-18-29_lr_0.07_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
    '2023-02-19-18-29_lr_0.07_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
    '2023-02-19-18-29_lr_0.07_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
    '2023-02-19-18-29_lr_0.07_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
    '2023-02-19-18-29_lr_0.07_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
    '2023-02-19-18-29_lr_0.07_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
    '2023-02-19-18-29_lr_0.07_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
    '2023-02-19-18-29_lr_0.07_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv',
]
files_lr_06_ep_10 = [
    '2023-02-19-18-29_lr_0.6_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
    '2023-02-19-18-29_lr_0.6_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
    '2023-02-19-18-29_lr_0.6_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
    '2023-02-19-18-29_lr_0.6_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
    '2023-02-19-18-29_lr_0.6_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
    '2023-02-19-18-29_lr_0.6_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
    '2023-02-19-18-29_lr_0.6_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
]
files_lr_004_ep_10 = [
    '2023-02-20-14-24_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
    '2023-02-20-14-24_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
    '2023-02-20-14-23_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
    '2023-02-20-14-24_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
    '2023-02-20-14-24_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
    '2023-02-20-14-24_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
    '2023-02-20-14-24_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
    '2023-02-20-14-24_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv',
]
files_lr_0035_ep_10 = [
    '2023-02-20-14-24_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv',
    '2023-02-20-14-24_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv',
    '2023-02-20-14-24_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv',
    '2023-02-20-14-24_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv',
    '2023-02-20-14-24_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv',
    '2023-02-20-14-24_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv',
    '2023-02-20-14-24_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv',
    '2023-02-20-14-24_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv',
]

files_xors_lr_004_ep_10 = [
    '2023-03-13-19-42_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-19-42_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-07_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-14-16_lr_0.04_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_01_ep_10 = [
    '2023-03-13-23-08_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.1_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_01_ep_15 = [
    '2023-03-13-23-08_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_01_ep_20 = [
    '2023-03-13-23-08_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_001_ep_10 = [
    '2023-03-13-23-08_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.01_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_001_ep_15 = [
    '2023-03-13-23-08_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_001_ep_20 = [
    '2023-03-13-23-08_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_004_ep_15 = [
    '2023-03-13-23-08_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_004_ep_20 = [
    '2023-03-13-23-08_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_005_ep_10 = [
    '2023-03-13-23-08_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.05_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_005_ep_15 = [
    '2023-03-13-23-08_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9',

]
files_xors_lr_005_ep_20 = [
    '2023-03-13-23-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9',

]
files_xors_lr_0035_ep_10 = [
    '2023-03-13-23-08_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.035_ep_10_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_0035_ep_15 = [
    '2023-03-13-23-08_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]
files_xors_lr_0035_ep_20 = [
    '2023-03-13-23-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3',
    '2023-03-13-23-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4',
    '2023-03-13-23-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5',
    '2023-03-13-23-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6',
    '2023-03-13-23-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7',
    '2023-03-13-23-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8',
    '2023-03-13-23-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9',
    '2023-03-13-23-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_1.0',
]

files_xors_lr_004_ep_15_1K_gen = [
    "2023-03-30-14-58_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-14-58_lr_0.04_ep_15_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]
files_xors_lr_004_ep_20_1K_gen = [
    "2023-03-30-14-58_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-17-26_lr_0.04_ep_20_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",

]
files_xors_lr_004_ep_25_1K_gen = [
    "2023-03-30-14-58_lr_0.04_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.04_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.04_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.04_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.04_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.04_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.04_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-14-58_lr_0.04_ep_25_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]

files_xors_lr_001_ep_15_1K_gen = [
    "2023-03-30-14-58_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-14-58_lr_0.01_ep_15_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]
files_xors_lr_001_ep_20_1K_gen = [
    "2023-03-30-15-12_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-15-12_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-18-36_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-18-36_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-18-36_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-18-14_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-18-14_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-18-13_lr_0.01_ep_20_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",

]
files_xors_lr_001_ep_25_1K_gen = [
    "2023-03-30-14-58_lr_0.01_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.01_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.01_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.01_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-18-07_lr_0.01_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-18-07_lr_0.01_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-18-07_lr_0.01_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-18-41_lr_0.01_ep_25_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]

files_xors_lr_0035_ep_15_1K_gen = [
    "2023-03-30-14-58_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-14-58_lr_0.035_ep_15_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]
files_xors_lr_0035_ep_20_1K_gen = [
    "2023-03-30-14-58_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-18-08_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-18-07_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-18-07_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-18-07_lr_0.035_ep_20_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]
files_xors_lr_0035_ep_25_1K_gen = [
    "2023-03-30-14-58_lr_0.035_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.035_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.035_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.035_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.035_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.035_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.035_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-14-58_lr_0.035_ep_25_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]

files_xors_lr_005_ep_15_1K_gen = [
    "2023-03-30-18-07_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-18-06_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-18-36_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-18-36_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.05_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",

]

files_xors_lr_005_ep_20_1K_gen = [
    "2023-03-30-18-13_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-18-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-18-08_lr_0.05_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",

]

files_xors_lr_005_ep_25_1K_gen = [
    "2023-03-30-14-58_lr_0.05_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.05_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-14-58_lr_0.05_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-14-58_lr_0.05_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-14-58_lr_0.05_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.05_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.05_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-14-58_lr_0.05_ep_25_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]
files_xors_lr_01_ep_15_1K_gen = [
    "2023-03-30-22-13_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-22-13_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-22-13_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-22-13_lr_0.1_ep_15_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
]
files_xors_lr_01_ep_20_1K_gen = [
    "2023-03-30-14-58_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-15-44_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-15-43_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-15-43_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-14-58_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",

    "2023-03-30-15-13_lr_0.1_ep_20_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]
files_xors_lr_01_ep_25_1K_gen = [
    "2023-03-30-14-58_lr_0.1_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.3.csv",
    "2023-03-30-14-58_lr_0.1_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.4.csv",
    "2023-03-30-18-16_lr_0.1_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.5.csv",
    "2023-03-30-18-08_lr_0.1_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.6.csv",
    "2023-03-30-18-08_lr_0.1_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.7.csv",
    "2023-03-30-18-16_lr_0.1_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.8.csv",
    "2023-03-30-14-58_lr_0.1_ep_25_pop_1000_reinitiate_True_connectivity_ratio_0.9.csv",
    "2023-03-30-14-58_lr_0.1_ep_25_pop_1000_reinitiate_True_connectivity_ratio_1.0.csv",
]

files = [
    files_xors_lr_004_ep_15_1K_gen,
    files_xors_lr_004_ep_20_1K_gen,
    files_xors_lr_004_ep_25_1K_gen,
    files_xors_lr_001_ep_15_1K_gen,
    files_xors_lr_001_ep_20_1K_gen,
    files_xors_lr_001_ep_25_1K_gen,
    files_xors_lr_0035_ep_15_1K_gen,
    files_xors_lr_0035_ep_20_1K_gen,
    files_xors_lr_0035_ep_25_1K_gen,
    files_xors_lr_005_ep_15_1K_gen,
    files_xors_lr_005_ep_20_1K_gen,
    files_xors_lr_005_ep_25_1K_gen,
    files_xors_lr_01_ep_15_1K_gen,
    files_xors_lr_01_ep_20_1K_gen,
    files_xors_lr_01_ep_25_1K_gen
]


def plot_single(
        results: pd.DataFrame,
        epochs: str,
        connectivity: str,
        reinitiate: str,
        out_path: str,
        file_name: str,
):
    plt.figure()
    plt.plot(results['generation'], results['performances'], c=COLORS[0])
    plt.xlabel('generation')
    plt.ylabel('performance')
    plt.title(f'performance of the best network after {epochs} epochs\n'
              f'(learning rate: {lr}, connectivity ratio: {connectivity}, reinitiate: {reinitiate})')
    plt.savefig(f'{out_path}/{file_name.split(".csv")[0]}.png')
    plt.close()


def pop_std(x):
    return x.std(ddof=0)


plt.figure()
for file_list in files:
    for i, file_name in enumerate(file_list):
        results = pd.read_csv(f'{results_path}/{file_name}').rename(columns={"Unnamed: 0": "generation", })
        splited = file_name.split('_')
        reinitiate = splited[-4]
        lr = splited[2]
        epochs = splited[4]
        connectivity = splited[-1].split('.csv')[0]
        vals = np.linspace(results['generation'].min(), results['generation'].max(), 20)
        bins = vals.tolist()
        labels = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        results['generation_2'] = pd.cut(x=results['generation'], bins=bins, labels=labels, include_lowest=True)
        r2 = results.groupby(['generation_2'], as_index=False).agg(
            {'performances': ['mean']}).dropna()
        plt.plot(r2['generation_2'], r2['performances'], c=COLORS[i], label=connectivity)
    plt.legend(title='architecture connectivity')
    plt.title(f'performance of the best network after {epochs} epochs (learning rate: {lr}) - XOR')
    plt.xlabel('generation')
    plt.ylabel('performance')
    plt.savefig(f'{out_path}/{time}_performance_per_generation_lr_{lr}_epochs_{epochs}_all_connectivity_xor.png')
    plt.show()
    plt.close()
