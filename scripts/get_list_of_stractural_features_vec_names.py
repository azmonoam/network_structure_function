import numpy as np
from stractural_features_models.structural_features import flatten_lists
import joblib

with open(
        f"/Volumes/noamaz/modularity/teach_archs/xors/xor_teach_archs_new_models/best_2023-03-13-13-49_2699_tr_0.9_50.pkl",
        'rb') as fp:
    organism = joblib.load(fp)

with open(
        f"/Volumes/noamaz/modularity/teach_archs/retina/retina_teach_archs_new_models/2023-02-18-16-00_5795.pkl",
        'rb') as fp:
    organism = joblib.load(fp)

per_layer_params = [
    "total_connectivity_ratio_between_layers",
    "layer_connectivity_rank",
    "max_connectivity_between_layers_per_layer",
]
per_layer_per_neuron_params = [
    'out_connections_per_layer',
    'in_connections_per_layer',
]
per_neuron_params = [
    "num_paths_to_output_per_input_neuron",
    "num_involved_neurons_in_paths_per_input_neuron",
]

# full name
a = {}
for val in organism.structural_features.__dict__.values():
    a = {**a, **val.__dict__}
a.pop('communities')
all_vals_names = []
for key, val in a.items():
    if isinstance(val, list):
        if key in per_layer_per_neuron_params:
            for ind in range(len(val)):
                all_vals_names += [f"{key}_{neuron} " for neuron in organism.layer_neuron_idx_mapping[ind]]
            continue
        else:
            all_vals = flatten_lists(val)
    elif isinstance(val, np.ndarray):
        all_vals = val.tolist()
    else:
        all_vals = [val]
    if key in per_layer_params or key == 'distances_between_input_neuron':
        all_vals_names += [f"{key}_{i}" for i in range(len(all_vals))]
    elif key in per_neuron_params:
        all_vals_names += [f"{key}_{i}" for i in organism.layer_neuron_idx_mapping[0]]
    else:
        all_vals_names += [key] * len(all_vals)

# simple name no neuron name only layers with reapets
a = {}
for val in organism.structural_features.__dict__.values():
    a = {**a, **val.__dict__}
a.pop('communities')
all_vals_names = []
for key, val in a.items():
    if isinstance(val, list):
        if key in per_layer_per_neuron_params:
            layer = 0
            for sub_val in val:
                all_vals_names += [f"{key}_{layer}"] * len(sub_val)
                layer += 1
            continue
        else:
            all_vals = flatten_lists(val)
    elif isinstance(val, np.ndarray):
        all_vals = val.tolist()
    else:
        all_vals = [val]
    if key in per_layer_params:
        all_vals_names += [f"{key}_{i}" for i in range(len(all_vals))]
    else:
        all_vals_names += [key] * len(all_vals)
