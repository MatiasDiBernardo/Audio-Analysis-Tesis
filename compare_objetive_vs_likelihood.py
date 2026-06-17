# Compara las 8 configuraciones definidas en data_likelihood.py:
#   - Estimación de densidad (likelihood) vs métricas objetivas CS y CA (superpuestas)
#
# Cada configuración se identifica por color; las métricas por la forma del marcador:
#   - círculo ('o') -> CS (Calidad de Señal, eje izquierdo)
#   - cuadrado ('s') -> CA (Calidad Acústica, eje derecho)
#
# Simplificaciones de matching likelihood <-> métricas objetivas:
#   - likelihood ("Demucs", "NISQA", "3")          -> métricas 'Demucs_NISQA_3'
#   - likelihood ("DeepFilterNet", "NISQA", "3.8") -> métricas 'DeepFilterNet_NISQA_3-8'

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from data_likelihood import LHOOD_VALUES
from data_computed_datasets import config_obj_metrics


# Mapeo entre las 8 configuraciones de likelihood y sus contrapartes
# en config_obj_metrics (data_computed_datasets.py).
# (likelihood_key) -> obj_metrics_key
CONFIG_MAPPING = {
    ("Demucs",        "DNSMOS", "2.7"): "Demucs_DNSMOS_2-7",
    ("Demucs",        "DNSMOS", "3.4"): "Demucs_DNSMOS_3-4",
    ("Demucs",        "NISQA",  "3"):   "Demucs_NISQA_3",          # simplificación
    ("Demucs",        "NISQA",  "4.2"): "Demucs_NISQA_4-2",
    ("DeepFilterNet", "DNSMOS", "2.7"): "DeepFilterNet_DNSMOS_2-7",
    ("DeepFilterNet", "NISQA",  "3.8"): "DeepFilterNet_NISQA_3-8", # simplificación
    ("No denoising",  "DNSMOS", "2.7"): "NoDenoising_DNSMOS_2-7",
    ("No denoising",  "NISQA",  "4.2"): "NoDenoising_NISQA_4-2",
}


def _short_label(lhood_key):
    """Etiqueta corta para identificar la configuración."""
    algo, metric, val = lhood_key
    algo_short = {"DeepFilterNet": "DFN", "No denoising": "NoDen", "Demucs": "Demucs"}.get(algo, algo)
    return f"{algo_short}_{metric}_{val}"


# Un color por tipo de denoising
DENOISER_COLORS = {
    "Demucs":        "tab:blue",
    "DeepFilterNet": "tab:orange",
    "No denoising":  "tab:green",
}


def _collect_data():
    """Devuelve listas alineadas con los datos para las 8 configuraciones."""
    labels, lhood_means = [], []
    cs_values, ca_values = [], []
    denoisers = []

    for lhood_key, obj_key in CONFIG_MAPPING.items():
        if obj_key not in config_obj_metrics:
            print(f"[warning] Objective metrics key faltante: {obj_key}")
            continue

        l_mean, _ = LHOOD_VALUES[lhood_key]
        obj = config_obj_metrics[obj_key]

        labels.append(_short_label(lhood_key))
        denoisers.append(lhood_key[0])
        lhood_means.append(l_mean)
        cs_values.append(obj["CS"])
        ca_values.append(obj["CA"])

    return {
        "labels": labels,
        "denoisers": denoisers,
        "lhood_means": np.array(lhood_means),
        "cs": np.array(cs_values),
        "ca": np.array(ca_values),
    }


def plot_density_vs_cs_and_ca(data):
    """Scatter superpuesto: densidad (x) vs CS y CA (dos ejes Y).

    - Cada tipo de denoising se grafica con un color distinto.
    - CS se dibuja con círculos sobre el eje Y izquierdo.
    - CA se dibuja con cuadrados sobre el eje Y derecho.
    """
    order = np.argsort(data["lhood_means"])
    x = data["lhood_means"][order]
    cs = data["cs"][order]
    ca = data["ca"][order]
    labels = [data["labels"][i] for i in order]
    denoisers = [data["denoisers"][i] for i in order]

    # Color según tipo de denoising
    colors = [DENOISER_COLORS.get(d, "gray") for d in denoisers]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    # Aumentar tamaño de los ticks en ambos ejes
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)

    # CS (círculo, eje izquierdo) y CA (cuadrado, eje derecho)
    for xi, cs_i, ca_i, color in zip(x, cs, ca, colors):
        ax1.plot(xi, cs_i, marker='o', markersize=10, color=color,
                 linestyle='None', markeredgecolor='black', markeredgewidth=0.5)
        ax2.plot(xi, ca_i, marker='s', markersize=10, color=color,
                 linestyle='None', markeredgecolor='black', markeredgewidth=0.5)

    # Etiqueta de configuración al lado de cada par de puntos
    for xi, cs_i, name in zip(x, cs, labels):
        ax1.annotate(name, (xi, cs_i), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=10, alpha=0.8)

    ax1.set_xlabel('Estimación de densidad (likelihood medio)', fontsize=14)
    ax1.set_ylabel('CS (Calidad de Señal)', fontsize=14)
    ax2.set_ylabel('CA (Calidad Acústica)', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Leyenda 1: color -> tipo de denoising
    denoiser_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
               markeredgecolor='black', markersize=10, label=name)
        for name, color in DENOISER_COLORS.items()
    ]
    legend_denoisers = ax1.legend(handles=denoiser_handles, title='Denoising',
                                  loc='upper left', fontsize=11, title_fontsize=12)
    ax1.add_artist(legend_denoisers)

    # Leyenda 2: marcador -> métrica
    metric_handles = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=10, label='CS (eje izquierdo)'),
        Line2D([0], [0], marker='s', color='gray', linestyle='None',
               markersize=10, label='CA (eje derecho)'),
    ]
    ax1.legend(handles=metric_handles, title='Métrica',
               loc='lower right', fontsize=11, title_fontsize=12)

    plt.tight_layout()

    corr_cs = np.corrcoef(x, cs)[0, 1]
    corr_ca = np.corrcoef(x, ca)[0, 1]
    print(f"Correlation (Density vs CS): {corr_cs:.4f}")
    print(f"Correlation (Density vs CA): {corr_ca:.4f}")


if __name__ == "__main__":
    data = _collect_data()
    plot_density_vs_cs_and_ca(data)
    plt.show()
