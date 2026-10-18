import html
import math
from pathlib import Path
import subprocess
import tempfile

from data_likelihood import LHOOD_VALUES_FILT
from data_computed_datasets import config_obj_metrics


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "calidad_vs_likelihood_filtrado.pdf"

# Exact match: denoiser, metric, and threshold agree between likelihood and
# objective metrics for every configuration below.
CONFIG_MAPPING = {
    ("Demucs", "DNSMOS", "2.7"): "Demucs_DNSMOS_2-7",
    ("Demucs", "DNSMOS", "3.4"): "Demucs_DNSMOS_3-4",
    ("Demucs", "NISQA", "3"): "Demucs_NISQA_3",
    ("Demucs", "NISQA", "4.2"): "Demucs_NISQA_4-2",
    ("DeepFilterNet", "DNSMOS", "2.7"): "DeepFilterNet_DNSMOS_2-7",
    ("DeepFilterNet", "NISQA", "3.8"): "DeepFilterNet_NISQA_3-8",
    ("No denoising", "DNSMOS", "2.7"): "NoDenoising_DNSMOS_2-7",
    ("No denoising", "NISQA", "4.2"): "NoDenoising_NISQA_4-2",
}

DENOISER_COLORS = {
    "Demucs": "#1f77b4",
    "DeepFilterNet": "#ff7f0e",
    "No denoising": "#2ca02c",
}


def _short_label(lhood_key):
    algo, metric, value = lhood_key
    short = {"DeepFilterNet": "DFN", "No denoising": "NoDen"}.get(algo, algo)
    return f"{short}_{metric}_{value}"


def _collect_data():
    labels, denoisers, likelihood, cs, ca = [], [], [], [], []
    for likelihood_key, objective_key in CONFIG_MAPPING.items():
        if likelihood_key not in LHOOD_VALUES_FILT:
            raise KeyError(f"Missing filtered likelihood key: {likelihood_key}")
        if objective_key not in config_obj_metrics:
            raise KeyError(f"Missing objective metrics key: {objective_key}")
        likelihood_mean, _ = LHOOD_VALUES_FILT[likelihood_key]
        objective = config_obj_metrics[objective_key]
        labels.append(_short_label(likelihood_key))
        denoisers.append(likelihood_key[0])
        likelihood.append(likelihood_mean)
        cs.append(objective["CS"])
        ca.append(objective["CA"])
    order = sorted(range(len(likelihood)), key=likelihood.__getitem__)
    return {
        "labels": [labels[i] for i in order],
        "denoisers": [denoisers[i] for i in order],
        "x": [likelihood[i] for i in order],
        "cs": [cs[i] for i in order],
        "ca": [ca[i] for i in order],
    }


def _tick_values(low, high, step):
    start = math.ceil((low - 1e-10) / step) * step
    count = int((high - start) / step) + 1
    return [round(start + i * step, 10) for i in range(max(0, count))]


def _correlation(x, y):
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    ss_x = sum((value - x_mean) ** 2 for value in x)
    ss_y = sum((value - y_mean) ** 2 for value in y)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / math.sqrt(ss_x * ss_y)


def _svg_plot(data, output):
    # 12 x 7 inches at 120 dpi, matching the dimensions of the original PDF.
    width, height = 1440, 840
    left, right, top, bottom = 90.0, 1348.0, 22.0, 760.0
    plot_w, plot_h = right - left, bottom - top
    x_min, x_max = min(data["x"]) - 0.05, max(data["x"]) + 0.05
    cs_min, cs_max = min(data["cs"]) - 0.08, max(data["cs"]) + 0.08
    ca_min, ca_max = min(data["ca"]) - 0.08, max(data["ca"]) + 0.08

    def px(x):
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def py_cs(y):
        return bottom - (y - cs_min) / (cs_max - cs_min) * plot_h

    def py_ca(y):
        return bottom - (y - ca_min) / (ca_max - ca_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="12in" height="7in" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans, sans-serif" fill="#222">',
    ]

    for tick in _tick_values(x_min, x_max, 0.1):
        x = px(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#b0b0b0" stroke-width="1" stroke-dasharray="4 5" opacity="0.35"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom + 27:.2f}" text-anchor="middle" font-size="19.2">{tick:.1f}</text>')
    for tick in _tick_values(cs_min, cs_max, 0.1):
        y = py_cs(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#b0b0b0" stroke-width="1" stroke-dasharray="4 5" opacity="0.35"/>')
        parts.append(f'<text x="{left - 12:.2f}" y="{y + 7:.2f}" text-anchor="end" font-size="19.2">{tick:.1f}</text>')
    for tick in _tick_values(ca_min, ca_max, 0.1):
        y = py_ca(tick)
        parts.append(f'<text x="{right + 12:.2f}" y="{y + 7:.2f}" text-anchor="start" font-size="19.2">{tick:.1f}</text>')

    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.5"/>')

    # The denoiser is encoded by color; labels show the exact acoustic metric and threshold.
    # Two lines keep nearby Demucs configurations legible without hiding the data.
    for i, (x, cs, ca, label, denoiser) in enumerate(zip(data["x"], data["cs"], data["ca"], data["labels"], data["denoisers"])):
        color, cx = DENOISER_COLORS[denoiser], px(x)
        cy1, cy2 = py_cs(cs), py_ca(ca)
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy1:.2f}" r="8" fill="{color}" stroke="black" stroke-width="1"/>')
        parts.append(f'<rect x="{cx - 8:.2f}" y="{cy2 - 8:.2f}" width="16" height="16" fill="{color}" stroke="black" stroke-width="1"/>')
        # Alternating levels keep labels apart for configurations with nearly
        # identical likelihood means (notably the two No denoising points).
        offset = 30 if i % 2 == 0 else 145
        label_y = max(top + 30, min(cy1, cy2) - offset)
        _, metric, threshold = label.split("_", 2)
        parts.append(f'<text x="{cx:.2f}" y="{label_y:.2f}" text-anchor="middle" font-size="16" opacity="0.85"><tspan x="{cx:.2f}">{html.escape(metric)}</tspan><tspan x="{cx:.2f}" dy="18">{html.escape(threshold)}</tspan></text>')

    # Denoiser legend (upper left).
    lx, ly, lw, lh = 101, 32, 194, 117
    parts.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="5" fill="white" fill-opacity="0.8" stroke="#ccc" stroke-width="1.5"/>')
    parts.append(f'<text x="{lx + lw / 2}" y="{ly + 23}" text-anchor="middle" font-size="19.2">Denoising</text>')
    for i, (name, color) in enumerate(DENOISER_COLORS.items()):
        cy = ly + 48 + i * 31
        parts.append(f'<circle cx="{lx + 25}" cy="{cy - 5}" r="8" fill="{color}" stroke="black" stroke-width="1"/>')
        parts.append(f'<text x="{lx + 50}" y="{cy + 2}" font-size="17.6">{html.escape(name)}</text>')

    # Marker legend (lower right).
    lx, ly, lw, lh = 1110, 659, 230, 91
    parts.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="5" fill="white" fill-opacity="0.8" stroke="#ccc" stroke-width="1.5"/>')
    parts.append(f'<text x="{lx + lw / 2}" y="{ly + 23}" text-anchor="middle" font-size="19.2">Métrica</text>')
    parts.append(f'<circle cx="{lx + 26}" cy="{ly + 47}" r="8" fill="gray"/>')
    parts.append(f'<text x="{lx + 51}" y="{ly + 53}" font-size="17.6">CS (eje izquierdo)</text>')
    parts.append(f'<rect x="{lx + 18}" y="{ly + 65}" width="16" height="16" fill="gray"/>')
    parts.append(f'<text x="{lx + 51}" y="{ly + 80}" font-size="17.6">CA (eje derecho)</text>')

    # Axis labels, sized to match the original 14 pt titles.
    parts.append(f'<text x="{(left + right) / 2:.2f}" y="829" text-anchor="middle" font-size="23.3">Estimación de densidad (likelihood medio filtrado)</text>')
    parts.append(f'<text x="27" y="{(top + bottom) / 2:.2f}" text-anchor="middle" font-size="23.3" transform="rotate(-90 27 {(top + bottom) / 2:.2f})">CS (Calidad de Señal)</text>')
    parts.append(f'<text x="1414" y="{(top + bottom) / 2:.2f}" text-anchor="middle" font-size="23.3" transform="rotate(-90 1414 {(top + bottom) / 2:.2f})">CA (Calidad Acústica)</text>')
    parts.extend(['</g>', '</svg>'])

    with tempfile.NamedTemporaryFile("w", suffix=".svg", encoding="utf-8", delete=False) as stream:
        stream.write("\n".join(parts))
        svg_path = Path(stream.name)
    try:
        subprocess.run(["magick", "-density", "120", str(svg_path), "-units", "PixelsPerInch", str(output)], check=True)
    finally:
        svg_path.unlink(missing_ok=True)

    print(f"Correlation (filtered likelihood vs CS): {_correlation(data['x'], data['cs']):.4f}")
    print(f"Correlation (filtered likelihood vs CA): {_correlation(data['x'], data['ca']):.4f}")


if __name__ == "__main__":
    _svg_plot(_collect_data(), OUTPUT)
    print(f"Filtered figure saved to: {OUTPUT}")
