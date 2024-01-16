import matplotlib.pyplot as plt
import pandas as pd

from analsis.analsis_utils.utils import COLORS

results_path = '/predict_teacability_results/2023-02-07-14-26-09_lr_0.001_bs_512_output.csv'
out_path = '/plots/predict_teacability'
baseline_results_path = '/predict_teacability_results/2023-02-07-15-24-43_baseline_lr_0.001.csv'

lr = 0.001
results = pd.read_csv(results_path)
baseline_results = pd.read_csv(baseline_results_path)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
cut =200
ax1.plot(results['Epoch'][:cut], results['losses'][:cut], c=COLORS[0], label='model',)
ax1.plot(baseline_results['Epoch'][:cut], baseline_results['losses'][:cut], label='baseline', c=COLORS[2])
ax2.plot(results['Epoch'][:cut], results['r2s train'][:cut], label='train', c=COLORS[1])
ax2.plot(results['Epoch'][:cut], results['r2s test'][:cut], label='test', c=COLORS[3])
ax2.plot(baseline_results['Epoch'][:cut], baseline_results['r2s'][:cut], label='baseline', c=COLORS[2])

ax1.set_xlabel('Epoch')
ax1.set_ylabel('loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('R2')
ax1.set_title('Prediction of teachability mean performance')
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
fig.savefig(f'{out_path}/predict_teacability_results_lr_{lr}_with_baseline_2.png')

results_path = '/predict_teacability_results/2023-02-07-15-24-43_baseline_lr_0.001.csv'
out_path = '/plots/predict_teacability'

lr = 0.001
results = pd.read_csv(results_path)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
ax1.plot(results['Epoch'], results['losses'], c=COLORS[0])
ax2.plot(results['Epoch'], results['r2s'], c=COLORS[1])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('R2')
ax1.set_title('Baseline - Prediction of teachability mean performance')
plt.legend()
plt.show()
fig.savefig(f'{out_path}/baseline_predict_teacability_results_lr_{lr}.png')

results_path = '/predict_teacability_results/2023-02-07-14-22-24_lr_0.01_bs_1024_output.csv'
out_path = '/plots/predict_teacability'

lr = 0.01
results = pd.read_csv(results_path)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
ax1.plot(results['Epoch'][:200], results['losses'][:200], c=COLORS[0])
ax2.plot(results['Epoch'][:200], results['r2s train'][:200], label='train', c=COLORS[1])
ax2.plot(results['Epoch'][:200], results['r2s test'][:200], label='test', c=COLORS[2])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('R2')
ax1.set_title('Prediction of teachability mean performance')
plt.legend()
plt.show()
fig.savefig(f'{out_path}/predict_teacability_results_lr_{lr}.png')
