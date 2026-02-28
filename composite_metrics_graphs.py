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

def calculate_metrics(pro):

    raw = {'HOURS': 24.30,
                'STOI': [0.99, 0.01],
                'PESQ': [2.82, 0.72],
                'SI-SDR': [19.08, 8.93],
                'T30': [0.98, 0.57],
                'C50': [15.86, 5.48],
                'D50': [90.77, 10.79],
                'SNR': [19.08, 8.93],
                'F0': [200.01, 103.57],
                }

    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    
    # w1 = 0.8
    # w2 = 1.1
    # w3 = 1.2
    # w4 = 1

    DR = 1 - (pro['HOURS']/raw['HOURS'])
    SQ = (raw['PESQ'][0]/pro['PESQ'][0]) + (raw['SI-SDR'][0]/pro['SI-SDR'][0]) + (raw['SNR'][0]/pro['SNR'][0])
    AP = (pro['T30'][0]/raw['T30'][0]) + (raw['C50'][0]/pro['C50'][0])
    SD = np.abs(1 - (pro['F0'][1]/raw['F0'][1])) + (pro['MCD'][0]/5)
    TOT = (DR * w1 + SQ * w2 + AP * w3 + SD * w4)

    return [DR * w1, SQ * w2, AP * w3, SD * w4, TOT]


def get_data_diff(data, x_key, y_key, category, options, idx_valx=0, idx_valy=0):
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
            DR, SQ, AP, SD, TOT = calculate_metrics(dic)

            if x_key == "RD":  # Reducción de datos
                x_vals.append(DR)

            if x_key == "CS":
                x_vals.append(SQ)

            if x_key == "CA":
                x_vals.append(AP)

            if y_key == "CS":  # Calidad de señal
                y_vals.append(SQ)

            if y_key == "CA":  # Condiciones acústicas
                y_vals.append(AP)

            if y_key == "DH":  # Diferencias del habla
                y_vals.append(SD)
    
    return x_vals, y_vals

def naming_fig(tag):
    if tag == "RD":  
        return "Reducción de datos (RD)"

    if tag == "CS":  
        return "Calidad señal (CS)"

    if tag == "CA":
        return "Condiciones acústicas (CA)"

    if tag == "DH":  
        return "Diferencias del habla (DH)"
    

# Análisis de variables respecto a reducción del dataset en horas
type_filt = "DNS"
if type_filt == "DNSMOS":
    data = DATA_DNSMOS
    # mos_vals = ["2.7", "3", "3.2", "3.4"]
    mos_vals = ["3.4", "3.2", "3", "2.7"]
else:
    data = DATA_NISQA
    # mos_vals = ["3", "3.5", "3.8", "4.2"]
    mos_vals = ["4.2", "3.8", "3.5", "3"]

all_figs = [("RD", "CS"), ("RD", "CA"), ("RD", "DH"), ("CS", "CA"), ("CS", "DH"), ("CA", "DH")]

for fig_graph in all_figs:
    category = "Denoising"  # Key de el elemento a comparar
    options = ["DeepFilterNet", "Demucs", "No denoising"]  # Nombre (value) de los elementos a comparar
    x_axis = fig_graph[0]
    y_axis = fig_graph[1]
    x_label = naming_fig(x_axis)
    y_label = naming_fig(y_axis)
    name = f"{x_axis} vs {y_axis} (nisqa)"
    ftsize = 12

    fig, ax = plt.subplots(figsize=(6.5,3))
    marker_style = ["o", "*", "^"]
    marker_size = [7, 9, 7]
    colors_plots = ["#2ca02c", "#1f77b4", "#ff7f0e"]

    # Pack into lists so we can plot with a for-loop and a styles list

    for i in range(len(options)):

        x_data, y_data = get_data_diff(data, x_axis, y_axis, category, options[i], idx_valx=0, idx_valy=0)
        x_data = np.array(x_data) 
        y_data = np.array(y_data) 
        idx = np.argsort(x_data)
        x_sorted = x_data[idx]
        y_sorted = y_data[idx]

        ax.plot(x_sorted, y_sorted, color=colors_plots[i], marker=marker_style[i], label=options[i], markersize=marker_size[i], linewidth=2.5)

        # Add labels
        assert len(x_sorted) == len(mos_vals)
        for j in range(len(x_sorted)):
            ax.text(x_sorted[j], y_sorted[j] + 0.01, mos_vals[j], ha="center", va="bottom", fontsize=9, fontweight="bold", color=colors_plots[i])

    ax.legend(fontsize=ftsize)
    ax.set_xlabel(x_label, fontsize=ftsize)
    ax.set_ylabel(y_label, fontsize=ftsize)
    # ax.tick_params(axis='both', which='major')
    ax.grid(True)
    fig.tight_layout()
    # plt.subplots_adjust(left=0.095, right=0.995, top=0.95, bottom=0.185)
    out_path = os.path.join("graph_finales", name + ".pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.show()
