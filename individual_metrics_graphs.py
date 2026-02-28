import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from data import DATA_DNSMOS, DATA_NISQA, DATA_ALL

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['pdf.use14corefonts'] = False

matplotlib.rcParams.update({'font.size': 12})

def get_data_diff(data, x_key, y_key, category, options, idx_valx=0, idx_valy=0):
    """Get the data for a comparative plot based on configuration options.

    Args:
        data (list): List of dictionaries with data.
        x_key (string): Parameter to evaluete in the X axes.
        y_key (string): Parameter to evaluete in the Y axes.
        category (string): Category for the comparison.
        options (string): Name of the option to compare.
    """

    original = {'HOURS': 24.30,
                'STOI': [0.99, 0.01],
                'PESQ': [2.82, 0.72],
                'SI-SDR': [19.08, 8.93],
                'T30': [0.98, 0.57],
                'C50': [15.86, 5.48],
                'D50': [90.77, 10.79],
                'SNR': [19.08, 8.93],
                'F0': [200.01, 103.57],
                }
    
    x_vals = []
    y_vals = []
    
    for dic in data:
        if dic[category] == options:
            if x_key == 'HOURS':
                diff_x = (1 - (dic[x_key]/original[x_key])) * 100
            else:
                diff_x =  (1 - original[x_key][idx_valx]/dic[x_key][idx_valx]) * 100

            # Porcentual diff (asumiendo que el resultado del procesado aumenta)
            # diff_y =  (1 - original[y_key][idx_valy]/dic[y_key][idx_valy]) * 100
            
            diff_y =  (1 - dic[y_key][idx_valy]/original[y_key][idx_valy]) * 100
            
            # MCD case
            #diff_y = (dic[y_key][idx_valy])/10
            
            # if diff_y < 0:
            #     diff_y *= -1

            x_vals.append(diff_x)
            y_vals.append(diff_y)
    
    return x_vals, y_vals

def get_data_raw(data, x_key, y_key, category, options, idx_valx=0, idx_valy=0):
    """Get the data for a comparative plot based on configuration options.

    Args:
        data (list): List of dictionaries with data.
        x_key (string): Parameter to evaluete in the X axes.
        y_key (string): Parameter to evaluete in the Y axes.
        category (string): Category for the comparison.
        options (string): Name of the option to compare.
    """
    x_vals = []
    y_vals = []
    
    for dic in data:
        if dic[category] == options:
            if x_key == 'HOURS':
                # diff_x = (dic[x_key]/original[x_key]) * 100
                diff_x = dic[x_key]
            else:
                diff_x = dic[x_key][idx_valx]

            diff_y = dic[y_key][idx_valy]

            x_vals.append(diff_x)
            y_vals.append(diff_y)
    
    return x_vals, y_vals

type_filt = "DNSMOS"

# Análisis de variables respecto a reducción del dataset en horas
if type_filt == "DNSMOS":
    data = DATA_DNSMOS
    mos_vals = ["2.7", "3", "3.2", "3.4"]
else:
    data = DATA_NISQA
    mos_vals = ["3", "3.5", "3.8", "4.2"]

category = "Denoising"  # Key de el elemento a comparar
options = ["DeepFilterNet", "Demucs", "No denoising"]  # Nombre (value) de los elementos a comparar
x_axis = "HOURS"
y_axis = "T30"
x_label = "Reducción del dataset [%]"
y_label = f"Mejora {y_axis} [%]"
name = f"{y_axis} vs hours (dnsmos)"
ftsize = 12
graph_uncert = False
graph_diff = True

fig, ax = plt.subplots(figsize=(6.5,2.8))
marker_style = ["o", "*", "^"]
marker_size = [7, 9, 7]
colors_plots = ["#2ca02c", "#1f77b4", "#ff7f0e"]

# Pack into lists so we can plot with a for-loop and a styles list
for i in range(len(options)):
    if graph_diff:
        x_data, y_data = get_data_diff(data, x_axis, y_axis, category, options[i], idx_valx=0, idx_valy=0)
    else:
        x_data, y_data = get_data_raw(data, x_axis, y_axis, category, options[i], idx_valx=0, idx_valy=0)
    
    x_data = np.array(x_data) 
    y_data = np.array(y_data) 
    idx = np.argsort(x_data)
    x_sorted = x_data[idx]
    y_sorted = y_data[idx]

    ax.plot(x_sorted, y_sorted, color=colors_plots[i], marker=marker_style[i], label=options[i], markersize=marker_size[i], linewidth=2.5)

    # Add labels
    assert len(x_sorted) == len(mos_vals)
    for j in range(len(x_sorted)):
        if i == 1:
            ax.text(x_sorted[j], y_sorted[j] + 1.5, mos_vals[j], ha="center", va="bottom", fontsize=9, fontweight="bold", color=colors_plots[i])
        else:
            ax.text(x_sorted[j], y_sorted[j] + 1, mos_vals[j], ha="center", va="bottom", fontsize=9, fontweight="bold", color=colors_plots[i])

    if graph_uncert:
        _, y_uncert = get_data_raw(data, x_axis, y_axis, category, options[i], idx_valx=0, idx_valy=1)
        y_uncert = np.array(y_uncert) 
        unc_sorted = y_uncert[idx]
        plt.fill_between(x_sorted, y_sorted - unc_sorted, y_sorted + unc_sorted, alpha=0.25)

    ax.legend(fontsize=ftsize)
    
ax.set_xlabel(x_label, fontsize=ftsize)
ax.set_ylabel(y_label, fontsize=ftsize)
# ax.tick_params(axis='both', which='major')
ax.grid(True)
fig.tight_layout()
# plt.subplots_adjust(left=0.09, right=0.995, top=0.999, bottom=0.19)
out_path = os.path.join("graph_finales", name + ".pdf")
plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
plt.show()
