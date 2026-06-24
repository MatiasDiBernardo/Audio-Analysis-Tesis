# Compara las 8 configuraciones definidas en data_likelihood.py:
#   - Estimación de densidad (likelihood) vs MOS
#
# Simplificaciones de matching MOS <-> likelihood:
#   - 'Demucs_NISQA_3-5'        -> usa likelihood ("Demucs", "NISQA", "3")          # simplificación
#   - 'DeepFilterNet_NISQA_4-2' -> usa likelihood ("DeepFilterNet", "NISQA", "3.8") # simplificación

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data_mos import rta_TTS_MOS_QwenTTS_es, rta_TTS_MOS_F5TTS_es
from data_likelihood import LHOOD_VALUES

# Modelos TTS a comparar: (nombre_dataset, label, color)
TTS_MODELS = [
    (rta_TTS_MOS_QwenTTS_es, "Qwen TTS", "dodgerblue"),
    (rta_TTS_MOS_F5TTS_es,   "F5 TTS",   "darkorange"),
]

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


def _collect_data(config_TTS_MOS_es):
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


def plot_density_vs_mos(datasets):
    """Scatter: densidad (x) vs MOS (y), ordenado por densidad.

    `datasets` es una lista de tuplas (data, label, color).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Asumimos misma configuración (mismos x) para todos los modelos.
    base = datasets[0][0]
    order_base = np.argsort(base["lhood_means"])
    x_all = base["lhood_means"][order_base]
    labels_all = [base["labels"][i] for i in order_base]

    # Para anclar las etiquetas por encima del punto más alto en cada x
    # y para fijar después un ylim con suficiente margen.
    y_max_per_x = np.full_like(x_all, -np.inf, dtype=float)
    y_min_per_x = np.full_like(x_all, np.inf, dtype=float)

    for data, label, color in datasets:
        order = np.argsort(data["lhood_means"])
        x = data["lhood_means"][order]
        y = data["mos_means"][order]

        # Regresión lineal + correlación de Pearson con p-valor
        reg = stats.linregress(x, y)
        slope, intercept = reg.slope, reg.intercept
        r, p_value = reg.rvalue, reg.pvalue

        # Recta de regresión extendida un poco más allá del rango de los datos
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        scatter_label = label
        ax.plot(x, y, 'o', markersize=10,
                color=color, alpha=0.8, label=scatter_label)
        ax.plot(x_line, y_line, '--', linewidth=2, color=color, alpha=0.7)

        y_max_per_x = np.maximum(y_max_per_x, y)
        y_min_per_x = np.minimum(y_min_per_x, y)

        significancia = "significativa" if p_value < 0.05 else "NO significativa"
        print(f"[{label}] Correlación (Densidad vs MOS): r = {r:.4f}, "
              f"p = {p_value:.4g} ({significancia} al 5%)")
        print(f"[{label}] Recta de regresión: y = {slope:.4f} * x + {intercept:.4f}")
        print(f"[{label}] Número de configuraciones representadas: {len(x)}")

    # Etiquetas de configuración: ancladas sobre el punto más alto de cada x,
    # con offsets verticales alternados para evitar colisiones cuando dos
    # puntos quedan próximos en el eje X.
    y_offsets_pts = [14, 32]
    for k, (xi, yi_top, name) in enumerate(zip(x_all, y_max_per_x, labels_all)):
        ax.annotate(name, (xi, yi_top), textcoords="offset points",
                    xytext=(0, y_offsets_pts[k % len(y_offsets_pts)]),
                    ha='center', fontsize=12, alpha=0.85, color='black')

    # Márgenes para que las etiquetas y puntos queden dentro del recuadro.
    x_range = float(x_all.max() - x_all.min())
    y_range = float(y_max_per_x.max() - y_min_per_x.min())
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0
    ax.set_xlim(x_all.min() - 0.08 * x_range,
                x_all.max() + 0.08 * x_range)
    ax.set_ylim(y_min_per_x.min() - 0.10 * y_range,
                y_max_per_x.max() + 0.30 * y_range)

    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel('Estimación de densidad (media de likelihood)', fontsize=18)
    ax.set_ylabel('MOS del TTS (media)', fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc='upper left')
    plt.tight_layout()


if __name__ == "__main__":
    datasets = [(_collect_data(mos_data), label, color)
                for mos_data, label, color in TTS_MODELS]
    plot_density_vs_mos(datasets)
    plt.show()
