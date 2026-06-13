"""
Compute mean and standard deviation of likelihood values across multiple runs
for each of the 8 config variants stored in the `All_likelihood/` folder, and
compare the results against the reference dictionary in `data_likelihood.py`.

Each CSV file in `All_likelihood/` is named:
    <Denoiser>_<Metric>_<Threshold>_<RunIdx>.csv
e.g. `DeepFilterNet_DNSMOS_2-7_0003.csv`. Threshold uses '-' as decimal point.

Inside each file there are two columns: `audio_id, logp`. Each file corresponds
to one run; per run we average `logp` across audios. The mean/std reported per
variant are then computed across the per-run means.
"""

from __future__ import annotations

import re
from pathlib import Path
from statistics import mean, pstdev, stdev

import pandas as pd

from data_likelihood import LHOOD_VALUES


CSV_DIR = Path(__file__).parent / "All_likelihood"

# Drop any logp value strictly greater than this threshold (outlier filter).
# Set to None to disable filtering.
FILTER_VALUE: float | None = 4.0

# Map filename denoiser tokens to dictionary keys
DENOISER_MAP = {
    "DeepFilterNet": "DeepFilterNet",
    "Demucs": "Demucs",
    "NoDenoising": "No denoising",
}

FNAME_RE = re.compile(
    r"^(?P<denoiser>DeepFilterNet|Demucs|NoDenoising)_"
    r"(?P<metric>DNSMOS|NISQA)_"
    r"(?P<threshold>[\d-]+)_"
    r"(?P<run>\d+)\.csv$"
)


def parse_filename(name: str) -> tuple[str, str, str] | None:
    m = FNAME_RE.match(name)
    if not m:
        return None
    denoiser = DENOISER_MAP[m.group("denoiser")]
    metric = m.group("metric")
    threshold = m.group("threshold").replace("-", ".")
    return denoiser, metric, threshold


def compute_stats(
    csv_dir: Path,
    filter_value: float | None = FILTER_VALUE,
) -> dict[tuple[str, str, str], list[float]]:
    """Return {(denoiser, metric, threshold): [mean, std]} computed by pooling
    all `logp` values across every run of the variant.

    If ``filter_value`` is not None, every ``logp`` strictly greater than that
    threshold is dropped before computing statistics (outlier removal).
    """
    pooled: dict[tuple[str, str, str], list[float]] = {}
    dropped: dict[tuple[str, str, str], int] = {}
    for f in sorted(csv_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() != ".csv":
            continue
        key = parse_filename(f.name)
        if key is None:
            print(f"[skip] Unrecognised filename: {f.name}")
            continue
        df = pd.read_csv(f)
        if "logp" not in df.columns:
            print(f"[skip] No 'logp' column in {f.name}")
            continue
        logp = df["logp"].astype(float)
        if filter_value is not None:
            mask = logp <= filter_value
            n_drop = int((~mask).sum())
            if n_drop:
                dropped[key] = dropped.get(key, 0) + n_drop
            logp = logp[mask]
        pooled.setdefault(key, []).extend(logp.tolist())

    if filter_value is not None and dropped:
        print(f"[filter] Dropped logp > {filter_value}:")
        for key in sorted(dropped):
            print(f"    {key}: {dropped[key]} value(s)")

    stats: dict[tuple[str, str, str], list[float]] = {}
    for key, vals in pooled.items():
        if len(vals) < 2:
            stats[key] = [mean(vals), 0.0]
        else:
            stats[key] = [mean(vals), stdev(vals)]
    return stats


def compare(
    computed: dict[tuple[str, str, str], list[float]],
    reference: dict[tuple[str, str, str], list[float]],
    tol: float = 1e-3,
) -> None:
    print(f"{'Variant':<45} {'mean(comp)':>11} {'mean(ref)':>11} "
          f"{'std(comp)':>11} {'std(ref)':>11}  match")
    print("-" * 100)
    all_ok = True
    for key in sorted(reference.keys()):
        ref_mean, ref_std = reference[key]
        if key not in computed:
            print(f"{str(key):<45}  MISSING in computed results")
            all_ok = False
            continue
        c_mean, c_std = computed[key]
        ok = abs(c_mean - ref_mean) < tol and abs(c_std - ref_std) < tol
        all_ok &= ok
        print(f"{str(key):<45} {c_mean:>11.5f} {ref_mean:>11.5f} "
              f"{c_std:>11.5f} {ref_std:>11.5f}  {'OK' if ok else 'DIFF'}")

    extra = set(computed) - set(reference)
    for key in sorted(extra):
        c_mean, c_std = computed[key]
        print(f"{str(key):<45} {c_mean:>11.5f} {'-':>11} "
              f"{c_std:>11.5f} {'-':>11}  EXTRA")

    print("-" * 100)
    print("All values match within tolerance." if all_ok
          else "Some values differ from the reference.")


def main() -> None:
    print(f"Reading CSV files from: {CSV_DIR}")
    print(f"Filter: dropping logp > {FILTER_VALUE}"
          if FILTER_VALUE is not None else "Filter: disabled")
    computed = compute_stats(CSV_DIR, filter_value=FILTER_VALUE)

    print("\nComputed LHOOD_VALUES = {")
    for key in sorted(computed.keys()):
        m, s = computed[key]
        print(f"    {key!r}: [{m:.5f}, {s:.5f}],")
    print("}\n")

    print("Comparison against data_likelihood.LHOOD_VALUES:")
    compare(computed, LHOOD_VALUES)


if __name__ == "__main__":
    main()
