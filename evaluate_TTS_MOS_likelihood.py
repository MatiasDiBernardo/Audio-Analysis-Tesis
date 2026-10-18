from __future__ import annotations

import html
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from data_mos import rta_TTS_MOS_QwenTTS_es, rta_TTS_MOS_F5TTS_es
from data_likelihood import LHOOD_VALUES_FILT


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "MOS_vs_likelihood_filtrado.png"

TTS_MODELS = [
    (rta_TTS_MOS_QwenTTS_es, "Qwen TTS", "#1e90ff"),
    (rta_TTS_MOS_F5TTS_es, "F5 TTS", "#ff8c00"),
]

# Exact matches only: same denoiser, quality metric, and threshold on both
# sides of the comparison. Two unmatched NISQA configurations are omitted.
CONFIG_MAPPING = [
    (("Demucs", "DNSMOS", "2.7"), "Demucs_DNSMOS_2-7"),
    (("Demucs", "DNSMOS", "3.4"), "Demucs_DNSMOS_3-4"),
    (("Demucs", "NISQA", "4.2"), "Demucs_NISQA_4-2"),
    (("DeepFilterNet", "DNSMOS", "2.7"), "DeepFilterNet_DNSMOS_2-7"),
    (("No denoising", "DNSMOS", "2.7"), "NoDenoising_DNSMOS_2-7"),
    (("No denoising", "NISQA", "4.2"), "NoDenoising_NISQA_4-2"),
]


def _short_label(lhood_key: tuple[str, str, str]) -> str:
    algo, metric, value = lhood_key
    short = {"DeepFilterNet": "DFN", "No denoising": "NoDen", "Demucs": "Demucs"}[algo]
    return f"{short}_{metric}_{value}"


def _collect_data(mos_data: dict) -> dict[str, list]:
    """Collect aligned means for the six exact configuration pairs."""
    labels, likelihood_means, mos_means = [], [], []
    for likelihood_key, mos_key in CONFIG_MAPPING:
        if likelihood_key not in LHOOD_VALUES_FILT:
            raise KeyError(f"Falta likelihood filtrado para {likelihood_key}")
        if mos_key not in mos_data:
            raise KeyError(f"Falta MOS para la configuración exacta {mos_key}")
        likelihood_mean, _ = LHOOD_VALUES_FILT[likelihood_key]
        mos_mean, _ = mos_data[mos_key]
        labels.append(_short_label(likelihood_key))
        likelihood_means.append(likelihood_mean)
        mos_means.append(mos_mean)
    return {"labels": labels, "x": likelihood_means, "y": mos_means}


def _correlation(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    n = len(x)
    x_mean, y_mean = sum(x) / n, sum(y) / n
    ss_x = sum((value - x_mean) ** 2 for value in x)
    ss_y = sum((value - y_mean) ** 2 for value in y)
    r = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / math.sqrt(ss_x * ss_y)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / ss_x
    intercept = y_mean - slope * x_mean
    t_value = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    p_value = _regularized_beta(df / (df + t_value * t_value), df / 2, 0.5)
    return r, p_value, slope, intercept


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction used to evaluate the regularized beta function."""
    max_iter, eps, tiny = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return h


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1 - factor * _beta_continued_fraction(b, a, 1 - x) / b


def _nice_step(span: float, target_ticks: int = 8) -> float:
    raw = span / target_ticks
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10 ** exponent)
    nice = min((1, 2, 2.5, 5, 10), key=lambda candidate: abs(candidate - fraction))
    return nice * (10 ** exponent)


def _tick_values(low: float, high: float, target_ticks: int = 8) -> list[float]:
    step = _nice_step(high - low, target_ticks)
    start = math.ceil((low - 1e-10) / step) * step
    count = int((high - start) / step) + 1
    return [round(start + i * step, 10) for i in range(max(0, count))]


def _svg_plot(datasets: list[tuple[dict, str, str]], output: Path) -> None:
    width, height = 1200, 700
    left, right, top, bottom = 90.0, 1173.0, 16.0, 627.0
    plot_w, plot_h = right - left, bottom - top
    base = datasets[0][0]
    labels = base["labels"]
    x_values = base["x"]
    all_y = [value for data, _, _ in datasets for value in data["y"]]
    x_range, y_range = max(x_values) - min(x_values), max(all_y) - min(all_y)
    x_min, x_max = min(x_values) - 0.08 * x_range, max(x_values) + 0.08 * x_range
    y_min, y_max = min(all_y) - 0.10 * y_range, max(all_y) + 0.30 * y_range

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def py(y: float) -> float:
        return bottom - (y - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans, sans-serif" fill="#222">',
    ]

    # Grid and tick labels, using the same light dashed style as the source plot.
    for x_tick in _tick_values(x_min, x_max):
        x = px(x_tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#b0b0b0" stroke-width="1" stroke-dasharray="6 5" opacity="0.3"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom + 27:.2f}" text-anchor="middle" font-size="21.33">{x_tick:.1f}</text>')
    for y_tick in _tick_values(y_min, y_max):
        y = py(y_tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#b0b0b0" stroke-width="1" stroke-dasharray="6 5" opacity="0.3"/>')
        parts.append(f'<text x="{left - 12:.2f}" y="{y + 7:.2f}" text-anchor="end" font-size="21.33">{y_tick:.1f}</text>')

    # Original four-sided axes frame.
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1"/>')

    # Regression lines and means for each TTS system.
    y_top_per_x = [-math.inf] * len(labels)
    y_bottom_per_x = [math.inf] * len(labels)
    for data, name, color in datasets:
        x, y = data["x"], data["y"]
        r, p_value, slope, intercept = _correlation(x, y)
        x_line = (min(x), max(x))
        line_points = " ".join(
            f"{px(xi):.2f},{py(slope * xi + intercept):.2f}" for xi in x_line
        )
        parts.append(f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="2.8" stroke-dasharray="8 6" opacity="0.7"/>')
        for xi, yi in zip(x, y):
            parts.append(f'<circle cx="{px(xi):.2f}" cy="{py(yi):.2f}" r="7" fill="{color}" opacity="0.8"/>')
        for i, yi in enumerate(y):
            y_top_per_x[i] = max(y_top_per_x[i], yi)
            y_bottom_per_x[i] = min(y_bottom_per_x[i], yi)
        print(f"[{name}] Pearson r={r:.4f}, p={p_value:.4g}; n={len(x)} configuraciones")

    # Configuration labels above the higher mean at each x, alternated like the original.
    order = sorted(range(len(x_values)), key=lambda i: x_values[i])
    label_offsets = (20, 44)
    for k, index in enumerate(order):
        x, y = x_values[index], y_top_per_x[index]
        label_y = py(y) - label_offsets[k % len(label_offsets)]
        label = html.escape(labels[index])
        label_x = px(x)
        estimated_width = len(labels[index]) * 8.2
        anchor = "middle"
        if label_x - estimated_width / 2 < left + 2:
            label_x, anchor = left + 4, "start"
        elif label_x + estimated_width / 2 > right - 2:
            label_x, anchor = right - 4, "end"
        parts.append(f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" font-size="16" opacity="0.85">{label}</text>')

    # Legend box and entries.
    legend_x, legend_y = 101.0, 27.0
    legend_w, legend_h = 193.0, 69.0
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" rx="5" fill="white" fill-opacity="0.8" stroke="#ccc" stroke-width="1.2"/>')
    for i, (_, name, color) in enumerate(datasets):
        cy = legend_y + 18 + i * 30
        parts.append(f'<circle cx="{legend_x + 32}" cy="{cy:.2f}" r="7" fill="{color}" opacity="0.8"/>')
        parts.append(f'<text x="{legend_x + 72}" y="{cy + 8:.2f}" font-size="21.33">{html.escape(name)}</text>')

    # Axis titles.
    parts.append(f'<text x="{(left + right) / 2:.2f}" y="680" text-anchor="middle" font-size="24">Estimación de densidad (media de likelihood)</text>')
    parts.append(f'<text x="27" y="{(top + bottom) / 2:.2f}" text-anchor="middle" font-size="24" transform="rotate(-90 27 {(top + bottom) / 2:.2f})">MOS del TTS (media)</text>')
    parts.extend(['</g>', '</svg>'])

    svg = "\n".join(parts)
    with tempfile.NamedTemporaryFile("w", suffix=".svg", encoding="utf-8", delete=False) as stream:
        stream.write(svg)
        svg_path = Path(stream.name)
    try:
        subprocess.run(["magick", "-background", "white", str(svg_path), "-depth", "8", str(output)], check=True)
    finally:
        svg_path.unlink(missing_ok=True)


def main() -> None:
    datasets = [(_collect_data(mos_data), label, color)
                for mos_data, label, color in TTS_MODELS]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT
    _svg_plot(datasets, output)
    print(f"Figura filtrada guardada en: {output}")


if __name__ == "__main__":
    main()
