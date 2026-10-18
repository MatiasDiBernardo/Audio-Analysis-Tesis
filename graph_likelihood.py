"""Generate the filtered likelihood KDE used as Figure 13 in the thesis.

The plot is intentionally implemented with the Python standard library so it
can be regenerated without a separate plotting environment. It pools the
consolidated observations for the three configurations shown in the thesis,
excludes values strictly above FILTER_VALUE, and writes a vector PDF.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import zlib


ROOT = Path(__file__).resolve().parent
FILTER_VALUE = 4.0
X_MIN = 0.85
X_MAX = 4.0
N_GRID = 500

# Keep each plotted configuration explicit. The labels and file order here
# must stay paired; the previous script accidentally mismatched them.
CONFIGURATIONS = [
    ("Demucs_NISQA_3_consolidado.csv", "Demucs (NISQA 3)", (31, 119, 180)),
    ("DeepFilterNet_DNSMOS_2-7_consolidado.csv", "DeepFilterNet (DNSMOS 2.7)", (255, 127, 14)),
    ("NoDenoising_DNSMOS_2-7_consolidado.csv", "Sin denoising (DNSMOS 2.7)", (44, 160, 44)),
]


def read_filtered_values(filename: str) -> list[float]:
    path = ROOT / "Likelihood" / filename
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            value = float(row["logp"])
            if math.isfinite(value) and value <= FILTER_VALUE:
                values.append(value)
    if len(values) < 2:
        raise ValueError(f"No hay suficientes observaciones filtradas en {path}")
    return values


def kde(values: list[float], xs: list[float]) -> list[float]:
    """Gaussian KDE with Scott's bandwidth, matching scipy.stats.gaussian_kde."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    bandwidth = math.sqrt(variance) * n ** (-1 / 5)
    normalizer = 1 / (n * bandwidth * math.sqrt(2 * math.pi))
    output = []
    for x in xs:
        output.append(
            normalizer
            * sum(math.exp(-0.5 * ((x - value) / bandwidth) ** 2) for value in values)
        )
    return output


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(output: Path, series: list[tuple[str, tuple[int, int, int], list[float]]]) -> None:
    # 10 x 6 inch page, in PDF points.
    page_w, page_h = 720.0, 432.0
    left, right, bottom, top = 67.0, 688.0, 58.0, 360.0
    plot_w, plot_h = right - left, top - bottom
    xs = [X_MIN + (X_MAX - X_MIN) * i / (N_GRID - 1) for i in range(N_GRID)]
    curves = [(label, color, kde(values, xs)) for label, color, values in series]
    y_max = max(max(ys) for _, _, ys in curves) * 1.12
    y_step = 0.25

    def px(x: float) -> float:
        return left + (x - X_MIN) / (X_MAX - X_MIN) * plot_w

    def py(y: float) -> float:
        return bottom + y / y_max * plot_h

    def rgb(color: tuple[int, int, int]) -> str:
        return " ".join(f"{component / 255:.5f}" for component in color)

    commands = ["1 J 1 j", "0.85 w"]
    # Light horizontal grid and tick labels.
    y_tick = 0.0
    while y_tick <= y_max + 1e-8:
        y = py(y_tick)
        commands.extend([
            "0.86 0.86 0.86 RG",
            f"{left:.2f} {y:.2f} m {right:.2f} {y:.2f} l S",
            "0.20 0.20 0.20 rg",
            f"BT /F1 10 Tf 1 0 0 1 {left - 32:.2f} {y - 3:.2f} Tm ({y_tick:.2f}) Tj ET",
        ])
        y_tick += y_step

    # X ticks and axis lines.
    for x_tick in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        x = px(x_tick)
        commands.extend([
            "0.20 0.20 0.20 RG",
            f"{x:.2f} {bottom:.2f} m {x:.2f} {bottom - 4:.2f} l S",
            "0.20 0.20 0.20 rg",
            f"BT /F1 10 Tf 1 0 0 1 {x - 8:.2f} {bottom - 19:.2f} Tm ({x_tick:.1f}) Tj ET",
        ])
    commands.extend([
        "0.20 0.20 0.20 RG 1 w",
        f"{left:.2f} {bottom:.2f} m {left:.2f} {top:.2f} l S",
        f"{left:.2f} {bottom:.2f} m {right:.2f} {bottom:.2f} l S",
    ])

    # Filled KDEs and outlines. Each fill uses a PDF transparency graphics state.
    for index, (label, color, ys) in enumerate(curves):
        commands.append(f"/GS{index} gs")
        commands.append(f"{rgb(color)} rg")
        commands.append(f"{px(xs[0]):.2f} {bottom:.2f} m")
        for x, y in zip(xs, ys):
            commands.append(f"{px(x):.2f} {py(y):.2f} l")
        commands.append(f"{px(xs[-1]):.2f} {bottom:.2f} l h f")
        commands.append("/GSfull gs")
        commands.append(f"{rgb(color)} RG 1.8 w")
        commands.append(f"{px(xs[0]):.2f} {py(ys[0]):.2f} m")
        for x, y in zip(xs[1:], ys[1:]):
            commands.append(f"{px(x):.2f} {py(y):.2f} l")
        commands.append("S")

    # Labels.
    commands.extend([
        "0.15 0.15 0.15 rg",
        f"BT /F1 12 Tf 1 0 0 1 {left + plot_w / 2 - 102:.2f} 20 Tm (Verosimilitud \\(Log-Likelihood\\)) Tj ET",
        f"BT /F1 12 Tf 0 1 -1 0 18 {bottom + plot_h / 2 - 67:.2f} Tm (Densidad de probabilidad) Tj ET",
    ])

    # Legend: white box with a thin border.
    legend_x, legend_y, legend_w, legend_h = 431.0, 243.0, 244.0, 101.0
    commands.extend([
        "1 1 1 rg 0.80 0.80 0.80 RG 0.7 w",
        f"{legend_x:.2f} {legend_y:.2f} {legend_w:.2f} {legend_h:.2f} re B",
        "0.15 0.15 0.15 rg",
        f"BT /F1 10 Tf 1 0 0 1 {legend_x + 10:.2f} {legend_y + legend_h - 17:.2f} Tm (Variante de preprocesamiento) Tj ET",
    ])
    for index, (label, color, _) in enumerate(curves):
        y = legend_y + legend_h - 39 - index * 20
        commands.extend([
            f"{rgb(color)} RG 2 w {legend_x + 10:.2f} {y + 3:.2f} m {legend_x + 29:.2f} {y + 3:.2f} l S",
            "0.15 0.15 0.15 rg",
            f"BT /F1 9 Tf 1 0 0 1 {legend_x + 36:.2f} {y:.2f} Tm ({_pdf_escape(label)}) Tj ET",
        ])

    stream = zlib.compress("\n".join(commands).encode("latin-1"))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.0f} {page_h:.0f}] /Resources << /Font << /F1 4 0 R >> /ExtGState << /GS0 5 0 R /GS1 6 0 R /GS2 7 0 R /GSfull 8 0 R >> >> /Contents 9 0 R >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /ExtGState /ca 0.15 /CA 0.15 >>",
        b"<< /Type /ExtGState /ca 0.15 /CA 0.15 >>",
        b"<< /Type /ExtGState /ca 0.15 /CA 0.15 >>",
        b"<< /Type /ExtGState /ca 1 /CA 1 >>",
        b"<< /Length " + str(len(stream)).encode() + b" /Filter /FlateDecode >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    output.write_bytes(pdf)


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kde_loglikelihood_filtrado.pdf"
    series = []
    for filename, label, color in CONFIGURATIONS:
        values = read_filtered_values(filename)
        print(f"{label}: n={len(values)}, media={sum(values) / len(values):.5f}")
        series.append((label, color, values))
    make_pdf(output, series)
    print(f"Figura filtrada guardada en: {output}")


if __name__ == "__main__":
    main()
