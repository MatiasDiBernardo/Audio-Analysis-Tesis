# Itera sobre TTS samples y evalúa la calidad de los samples (MOS)
# Compara las configuraciones contra una métrica objetiva para los modelos
# TTS disponibles (Qwen TTS y F5 TTS) en el mismo gráfico.

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data_mos import rta_TTS_MOS_QwenTTS_es, rta_TTS_MOS_F5TTS_es
from data_computed_datasets import config_obj_metrics

# Variable to select which objective metric to plot
obj_metrics = 'CA'  # Options: 'RD', 'CS', 'CA', 'DH', 'TOT'

# Modelos TTS a comparar: (dataset_mos, label, color)
TTS_MODELS = [
    (rta_TTS_MOS_QwenTTS_es, "Qwen TTS", "dodgerblue"),
    (rta_TTS_MOS_F5TTS_es,   "F5 TTS",   "darkorange"),
]

# Abreviaciones de algoritmo para que las etiquetas ocupen menos espacio.
_ALGO_SHORT = {
    "DeepFilterNet": "DFN",
    "NoDenoising":   "NoDen",
    "Demucs":        "Demucs",
}


def _short_label(config_name):
    """Versión abreviada del nombre de configuración para anotar puntos."""
    parts = config_name.split("_", 1)
    if len(parts) != 2:
        return config_name
    algo, rest = parts
    return f"{_ALGO_SHORT.get(algo, algo)}_{rest}"


def _collect_data(config_TTS_MOS_es, obj_metrics_type):
    """Devuelve listas alineadas con las configuraciones comunes."""
    # Orden determinista para que todos los modelos compartan el mismo eje X.
    common_configs = sorted(
        set(config_TTS_MOS_es.keys()) & set(config_obj_metrics.keys())
    )

    metric_values, mos_means, mos_stds, names = [], [], [], []
    for config in common_configs:
        metric_values.append(config_obj_metrics[config][obj_metrics_type])
        m_mean, m_std = config_TTS_MOS_es[config]
        mos_means.append(m_mean)
        mos_stds.append(m_std)
        names.append(config)

    return {
        "labels": names,
        "metric": np.array(metric_values),
        "mos_means": np.array(mos_means),
        "mos_stds": np.array(mos_stds),
    }


def plot_mos_vs_objective_metric(obj_metrics_type='RD'):
    """Scatter: métrica objetiva (x) vs MOS (y) para todos los modelos TTS.

    Dibuja también la recta de regresión lineal de cada modelo e imprime
    el coeficiente de correlación de Pearson con su p-valor.
    """
    datasets = [(_collect_data(mos_data, obj_metrics_type), label, color)
                for mos_data, label, color in TTS_MODELS]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Asumimos misma configuración (mismos x) para todos los modelos.
    base = datasets[0][0]
    order_base = np.argsort(base["metric"])
    x_all = base["metric"][order_base]
    labels_all = [base["labels"][i] for i in order_base]

    y_max_per_x = np.full_like(x_all, -np.inf, dtype=float)
    y_min_per_x = np.full_like(x_all, np.inf, dtype=float)

    for data, label, color in datasets:
        order = np.argsort(data["metric"])
        x = data["metric"][order]
        y = data["mos_means"][order]

        # Regresión lineal + correlación de Pearson con p-valor
        reg = stats.linregress(x, y)
        slope, intercept = reg.slope, reg.intercept
        r, p_value = reg.rvalue, reg.pvalue

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        ax.plot(x, y, 'o', markersize=10,
                color=color, alpha=0.8, label=label)
        ax.plot(x_line, y_line, '--', linewidth=2, color=color, alpha=0.7)

        y_max_per_x = np.maximum(y_max_per_x, y)
        y_min_per_x = np.minimum(y_min_per_x, y)

        significancia = "significativa" if p_value < 0.05 else "NO significativa"
        print(f"[{label}] Correlación ({obj_metrics_type} vs MOS): r = {r:.4f}, "
              f"p = {p_value:.4g} ({significancia} al 5%)")
        print(f"[{label}] Recta de regresión: y = {slope:.4f} * x + {intercept:.4f}")
        print(f"[{label}] Número de configuraciones representadas: {len(x)}")

    # Etiquetas de configuración: ancladas sobre el punto más alto de cada x,
    # con colocación por niveles que evita colisiones horizontales (varios
    # niveles verticales disponibles; si dos labels están más cerca que un
    # umbral en X, se apila el segundo en un nivel superior).
    x_range_lbl = float(x_all.max() - x_all.min()) or 1.0
    x_threshold = 0.12 * x_range_lbl  # separación mínima requerida en X
    level_offsets = [14, 34, 54, 74]  # offsets verticales en puntos

    # Ajustes manuales (en puntos) para labels específicos.
    label_y_overrides = {
        "DFN_DNSMOS_3-2":  +20,  # subir
        "DFN_NISQA_4-2":   -30,  # bajar un poco
        "Demucs_DNSMOS_3-4": -30,  # bajar un poco
    }

    placed = []  # lista de (x, nivel)
    for xi, yi_top, name in zip(x_all, y_max_per_x, labels_all):
        chosen_level = len(level_offsets) - 1  # por defecto el más alto
        for lvl in range(len(level_offsets)):
            conflict = any(abs(xi - px) < x_threshold and pl == lvl
                           for px, pl in placed)
            if not conflict:
                chosen_level = lvl
                break
        placed.append((xi, chosen_level))

        short_name = _short_label(name)
        y_offset = level_offsets[chosen_level] + label_y_overrides.get(short_name, 0)

        ax.annotate(short_name, (xi, yi_top),
                    textcoords="offset points",
                    xytext=(0, y_offset),
                    ha='center', fontsize=11, alpha=0.85, color='black')

    # Márgenes para que las etiquetas y puntos queden dentro del recuadro.
    x_range = float(x_all.max() - x_all.min())
    y_range = float(y_max_per_x.max() - y_min_per_x.min())
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0
    max_level_used = max(lvl for _, lvl in placed)
    top_margin = 0.20 + 0.15 * max_level_used  # más margen si hay más niveles
    ax.set_xlim(x_all.min() - 0.10 * x_range,
                x_all.max() + 0.10 * x_range)
    ax.set_ylim(y_min_per_x.min() - 0.10 * y_range,
                y_max_per_x.max() + top_margin * y_range)

    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(f'Métrica objetiva: {obj_metrics_type}', fontsize=18)
    ax.set_ylabel('MOS del TTS (media)', fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc='upper left')
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Change obj_metrics variable to plot different objective metrics
    plot_mos_vs_objective_metric(obj_metrics)
