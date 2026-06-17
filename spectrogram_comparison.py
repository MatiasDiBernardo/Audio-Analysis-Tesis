"""
Comparación de espectrogramas: Sin procesar vs Demucs vs DeepFilterNet.

Recorre recursivamente el dataset apuntado por `ROOT` y empareja audios con el
mismo nombre presentes en:
  - alguna carpeta `Demucs_*/`
  - alguna carpeta `DeepFilterNet_*/`
  - (opcional) `Original/` o alguna carpeta `NoDenoising_*/`

Por defecto usa el dataset completo
`F:\\Clase\\ArchiVoz\\Dataset_Regiones\\All_variants`. Si esa ruta no existe,
hace fallback a `Small_variants/` dentro del workspace.

Para cada coincidencia muestra los espectrogramas log-amplitud en una sola
figura. Al cerrar la ventana de matplotlib se pasa al siguiente ejemplo.

Opcionalmente se puede mezclar el orden con `--shuffle` o limitar los
ejemplos con `--max N`.
"""

import argparse
import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ---- Configuración ----------------------------------------------------------
DEFAULT_ROOT = Path(r"F:\Clase\ArchiVoz\Dataset_Regiones\All_variants")
FALLBACK_ROOT = Path(__file__).parent / "Small_variants"

DEMUCS_PREFIX = "Demucs_"
DFN_PREFIX = "DeepFilterNet_"
NODEN_PREFIX = "NoDenoising_"
ORIGINAL_DIR_NAME = "Original"

AUDIO_EXTS = {".mp3", ".wav", ".flac"}

N_FFT = 2048
HOP_LENGTH = 512
FMAX = 8000      # límite superior del eje de frecuencia
TOP_DB = 80      # rango dinámico del colormap


# ---- Indexado ---------------------------------------------------------------
def index_folders(root: Path):
    """Devuelve {filename: {"clean": [...], "demucs": [...], "dfn": [...]}}.

    Recorre cada subcarpeta de primer nivel de `root` y dentro de ella busca
    recursivamente archivos de audio. Soporta tanto carpetas planas como
    carpetas anidadas por hablante (p. ej. `Demucs_X/audio_70/audio_70_0001.mp3`).
    """
    index = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name == ORIGINAL_DIR_NAME:
            kind, variant = "clean", "Original"
        elif name.startswith(NODEN_PREFIX):
            kind, variant = "clean", "NoDenoising:" + name[len(NODEN_PREFIX):]
        elif name.startswith(DEMUCS_PREFIX):
            kind, variant = "demucs", name[len(DEMUCS_PREFIX):]
        elif name.startswith(DFN_PREFIX):
            kind, variant = "dfn", name[len(DFN_PREFIX):]
        else:
            continue
        for p in sub.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
                continue
            entry = index.setdefault(p.name,
                                     {"clean": [], "demucs": [], "dfn": []})
            entry[kind].append((p, variant))
    return index


# ---- Espectrogramas ---------------------------------------------------------
def compute_log_spectrogram(path: Path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max, top_db=TOP_DB)
    return S_db, sr, len(y) / sr


def plot_panels(filename, panels):
    """panels: list of (title, path)."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3 * n + 0.5), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Espectrogramas — {filename}", fontsize=13, fontweight="bold")

    last_img = None
    for idx, (ax, (title, p)) in enumerate(zip(axes, panels)):
        S_db, sr, dur = compute_log_spectrogram(p)
        last_img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            ax=ax,
            cmap="magma",
            vmin=-TOP_DB,
            vmax=0,
        )
        ax.set_ylim(0, FMAX)
        ax.set_title(title)
        ax.set_ylabel("Frecuencia [Hz]")
        # Quitar el "Time" que librosa añade en cada subplot;
        # solo dejamos la etiqueta en el último.
        if idx < len(panels) - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Tiempo [s]")

    fig.colorbar(last_img, ax=axes, format="%+2.0f dB",
                 location="right", shrink=0.85, pad=0.02, label="dB")

    print(f"  Mostrando: {filename}  (cerrá la ventana para continuar)")
    plt.show()
    plt.close(fig)


# ---- Loop principal ---------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None,
                    help=f"Carpeta raíz del dataset (default: {DEFAULT_ROOT})")
    ap.add_argument("--shuffle", action="store_true",
                    help="Mezclar el orden de los ejemplos.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Semilla para --shuffle.")
    ap.add_argument("--max", dest="max_n", type=int, default=None,
                    help="Limitar la cantidad de ejemplos a mostrar.")
    ap.add_argument("--require-clean", action="store_true",
                    help="Mostrar sólo audios que también tengan versión sin "
                         "procesar (Original/NoDenoising).")
    return ap.parse_args()


def resolve_root(arg_root):
    if arg_root is not None:
        if not arg_root.is_dir():
            raise SystemExit(f"No se encontró la carpeta: {arg_root}")
        return arg_root
    if DEFAULT_ROOT.is_dir():
        return DEFAULT_ROOT
    if FALLBACK_ROOT.is_dir():
        print(f"[aviso] No existe {DEFAULT_ROOT}.\n"
              f"        Usando fallback local: {FALLBACK_ROOT}\n")
        return FALLBACK_ROOT
    raise SystemExit(f"No se encontró ni {DEFAULT_ROOT} ni {FALLBACK_ROOT}.")


def main():
    args = parse_args()
    root = resolve_root(args.root)
    print(f"Dataset: {root}\n")

    index = index_folders(root)

    # Mínimo requerido: Demucs y DeepFilterNet del mismo nombre de archivo.
    matches = [(name, e) for name, e in index.items()
               if e["demucs"] and e["dfn"]
               and (not args.require_clean or e["clean"])]
    matches.sort(key=lambda x: x[0])

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(matches)

    if args.max_n is not None:
        matches = matches[:args.max_n]

    if not matches:
        raise SystemExit("No se encontraron audios procesados por Demucs Y "
                         "DeepFilterNet con el mismo nombre.")

    n_with_clean = sum(1 for _, e in matches if e["clean"])
    print(f"Encontradas {len(matches)} coincidencias Demucs+DeepFilterNet.")
    print(f"  De ellas, {n_with_clean} también tienen versión sin procesar.\n")

    for i, (name, entry) in enumerate(matches, 1):
        demucs_path, demucs_variant = entry["demucs"][0]
        dfn_path, dfn_variant = entry["dfn"][0]

        panels = []
        if entry["clean"]:
            clean_path, clean_variant = entry["clean"][0]
            panels.append((f"Sin procesar [{clean_variant}]", clean_path))
        panels.append((f"Demucs [{demucs_variant}]", demucs_path))
        panels.append((f"DeepFilterNet [{dfn_variant}]", dfn_path))

        print(f"[{i}/{len(matches)}] {name}")
        for title, p in panels:
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            print(f"     {title:40s}  ->  {rel}")

        try:
            plot_panels(name, panels)
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario.")
            break
        except Exception as e:
            print(f"     [!] Error procesando {name}: {e}")
            continue

    print("\nListo.")


if __name__ == "__main__":
    main()
