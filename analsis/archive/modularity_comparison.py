import matplotlib.pyplot as plt

from analsis.analsis_utils.utils import COLORS, get_organism_list_from_pkl_folder, get_modularity_list, get_performance_list

pa_exp_name = '2022_12_11_no_connection_20prec'
pacc_exp_name = '2022_12_12_25prec_connection_chance_20prec_perents_sorted'

pa_best_organisms = get_organism_list_from_pkl_folder(
    folder=f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{pa_exp_name}/best_network',
)
pacc_best_organisms = get_organism_list_from_pkl_folder(
    folder=f'/Users/noamazmon/PycharmProjects/network_modularity/experiment_data/{pacc_exp_name}/best_network',
)

fig, axs = plt.subplots(2, 1)
axs[0].plot(get_performance_list(pa_best_organisms), label='performance only', color=COLORS[0])
axs[0].plot(get_performance_list(pacc_best_organisms), label='performance and connection cost', color=COLORS[5])
axs[0].legend(loc="lower right")
axs[0].set_xticks([])
axs[0].set_ylabel('Performance')
axs[1].plot(get_modularity_list(pa_best_organisms), label='performance only', color=COLORS[0])
axs[1].plot(get_modularity_list(pacc_best_organisms), label='performance and connection cost', color=COLORS[5])
axs[1].set_ylabel('Modularity')
axs[1].set_xlabel('Generation')
plt.suptitle('Performance only vs.performance and connection cost')
fig.tight_layout()
plt.show()
fig.savefig(f'/Users/noamazmon/PycharmProjects/network_modularity/20221211_plots/pa_vs_pacc_{pacc_exp_name}_{pa_exp_name}.png')