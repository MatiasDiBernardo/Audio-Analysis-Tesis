# Compara las 8 configuraciones definidas en data_likelihood.py:
#   - Estimación de densidad (likelihood) vs MOS
#
# Simplificaciones de matching MOS <-> likelihood:
#   - 'Demucs_NISQA_3-5'        -> usa likelihood ("Demucs", "NISQA", "3")          # simplificación
#   - 'DeepFilterNet_NISQA_4-2' -> usa likelihood ("DeepFilterNet", "NISQA", "3.8") # simplificación

import matplotlib.pyplot as plt
import numpy as np

from data_mos import rta_TTS_MOS_QwenTTS_es as config_TTS_MOS_es
from data_likelihood import LHOOD_VALUES

# Mapeo entre las 8 configuraciones de likelihood y sus contrapartes
# en config_TTS_MOS_es (data_mos.py).
# (likelihood_key) -> mos_key
CONFIG_MAPPING = {
    ("Demucs",        "DNSMOS", "2.7"): "Demucs_DNSMOS_2-7",
    ("Demucs",        "DNSMOS", "3.4"): "Demucs_DNSMOS_3-4",
    ("Demucs",        "NISQA",  "3"):   "Demucs_NISQA_3-5",         # simplificación
    ("Demucs",        "NISQA",  "4.2"): "Demucs_NISQA_4-2",
    ("DeepFilterNet", "DNSMOS", "2.7"): "DeepFilterNet_DNSMOS_2-7",
    ("DeepFilterNet", "NISQA",  "3.8"): "DeepFilterNet_NISQA_4-2",  # simplificación
    ("No denoising",  "DNSMOS", "2.7"): "NoDenoising_DNSMOS_2-7",
    ("No denoising",  "NISQA",  "4.2"): "NoDenoising_NISQA_4-2",
}


def _short_label(lhood_key):
    """Etiqueta corta para anotar puntos."""
    algo, metric, val = lhood_key
    algo_short = {"DeepFilterNet": "DFN", "No denoising": "NoDen", "Demucs": "Demucs"}.get(algo, algo)
    return f"{algo_short}_{metric}_{val}"


def _collect_data():
    """Devuelve listas alineadas con los datos para las 8 configuraciones."""
    labels, lhood_means = [], []
    mos_means, mos_stds = [], []

    for lhood_key, mos_key in CONFIG_MAPPING.items():
        if mos_key not in config_TTS_MOS_es:
            print(f"[warning] MOS key faltante: {mos_key}")
            continue

        l_mean, _ = LHOOD_VALUES[lhood_key]
        m_mean, m_std = config_TTS_MOS_es[mos_key]

        labels.append(_short_label(lhood_key))
        lhood_means.append(l_mean)
        mos_means.append(m_mean)
        mos_stds.append(m_std)

    return {
        "labels": labels,
        "lhood_means": np.array(lhood_means),
        "mos_means": np.array(mos_means),
        "mos_stds": np.array(mos_stds),
    }


def plot_density_vs_mos(data):
    """Scatter: densidad (x) vs MOS (y), ordenado por densidad."""
    order = np.argsort(data["lhood_means"])
    x = data["lhood_means"][order]
    y = data["mos_means"][order]
    labels = [data["labels"][i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, y, 'o', markersize=10, linewidth=2,
            color='dodgerblue', alpha=0.8, label='Qwen TTS')

    ax.tick_params(axis='both', labelsize=16)

    for xi, yi, name in zip(x, y, labels):
        ax.annotate(name, (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=14, alpha=0.8)

    ax.set_xlabel('Estimación de densidad (media de likelihood)', fontsize=18)
    ax.set_ylabel('MOS del TTS (media)', fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16)
    plt.tight_layout()

    corr = np.corrcoef(x, y)[0, 1]
    print(f"Correlación (Densidad vs MOS): {corr:.4f}")
    print(f"Número de configuraciones representadas: {len(x)}")


if __name__ == "__main__":
    data = _collect_data()
    plot_density_vs_mos(data)
    plt.show()
