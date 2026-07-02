"""
Descarga y unifica datasets de Roboflow Universe para el reentrenamiento de
best_100e.pt (Tier 3 — cámara lateral/baja, ver docs/PLAN_player_tracking.md).

Pensado para correr en Kaggle:
    1. Añadir ROBOFLOW_API_KEY como Kaggle Secret y adjuntarla al notebook
       (se expone como variable de entorno automáticamente).
    2. pip install -r requirements.txt
    3. python datasets/download_datasets.py

Salida: datasets/merged/{train,valid,test}/{images,labels} + data.yaml,
listo para train_jugadores.py.

IMPORTANTE: los datasets en DATASET_SPECS se localizaron por búsqueda en
Roboflow Universe (footage de cámara lateral/baja de fútbol). No se ha
verificado manualmente la licencia ni el contenido exacto de cada versión —
revisar en universe.roboflow.com antes de la primera ejecución y ajustar la
lista si alguno no es adecuado.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

try:
    from roboflow import Roboflow
except ImportError:
    sys.exit("Falta el paquete 'roboflow'. Instalar con: pip install roboflow")

# best_100e.pt solo se usa hoy para jugador/portero/árbitro — el balón lo cubre
# modelo_balon.pt por separado (Tier ya completo). No se incluye "ball" aquí a
# propósito: sus cajas se descartan en build_remap() para no meter ruido de
# objeto pequeño en el fine-tuning de este modelo.
CANONICAL_CLASSES = ["goalkeeper", "player", "referee"]

# version=None -> se resuelve la última versión publicada del proyecto en tiempo
# de ejecución (evita hardcodear un número que puede quedar desactualizado).
# Solo datasets de cámara lateral/tribuna (coherentes con el caso Dinamó); se
# descartó "football-player-detection" de yolo-fifa por ser cámara aérea vertical
# detrás de portería.
DATASET_SPECS = [
    {"workspace": "object-detection-in-football-q1i5u", "project": "football-side-camera", "version": None},
    {"workspace": "roboflow-jvuqo", "project": "football-players-detection-3zvbc", "version": None},
    {"workspace": "ilyes-talbi-ptwsp", "project": "futbol-players", "version": None},
]

RAW_DIR = Path(__file__).parent / "raw"
MERGED_DIR = Path(__file__).parent / "merged"
SPLITS = ["train", "valid", "test"]

# Alias conocidos -> clase canónica. Cualquier nombre de clase no listado aquí
# (incluido el balón) se descarta (p.ej. "ball", "goal", "logo", "crowd").
CLASS_ALIASES = {
    "goalkeeper": "goalkeeper", "gk": "goalkeeper",
    "player": "player", "players": "player",
    "team a": "player", "team b": "player", "home": "player", "away": "player",
    "referee": "referee", "ref": "referee",
    "main referee": "referee", "side referee": "referee", "assistant referee": "referee",
}


def download(spec, api_key):
    dest = RAW_DIR / spec["project"]
    if dest.exists():
        print(f"  ↷ ya descargado: {dest}")
        return dest

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(spec["workspace"]).project(spec["project"])
    version_num = spec["version"] or project.versions()[-1].version
    dataset = project.version(version_num).download("yolov8", location=str(dest))
    return Path(dataset.location)


def load_class_names(data_yaml_path):
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return names


def build_remap(source_names):
    """Índice de clase del dataset origen -> índice canónico (o None si se descarta)."""
    remap = {}
    for idx, name in enumerate(source_names):
        canon = CLASS_ALIASES.get(name.strip().lower())
        remap[idx] = CANONICAL_CLASSES.index(canon) if canon else None
    return remap


def remap_and_copy(dataset_dir, dataset_tag, remap):
    for split in SPLITS:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue

        out_img_dir = MERGED_DIR / split / "images"
        out_lbl_dir = MERGED_DIR / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for label_file in lbl_dir.glob("*.txt"):
            lines_out = []
            for line in label_file.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                new_cls = remap.get(int(parts[0]))
                if new_cls is None:
                    continue  # clase no mapeable (p.ej. "goal", "logo") -> se descarta la caja
                lines_out.append(" ".join([str(new_cls)] + parts[1:]))
            if not lines_out:
                continue  # imagen sin cajas útiles tras el remap -> se descarta

            img_file = next(img_dir.glob(f"{label_file.stem}.*"), None)
            if img_file is None:
                continue

            new_stem = f"{dataset_tag}_{label_file.stem}"
            shutil.copy(img_file, out_img_dir / f"{new_stem}{img_file.suffix}")
            (out_lbl_dir / f"{new_stem}.txt").write_text("\n".join(lines_out) + "\n")


def write_merged_yaml():
    cfg = {
        "path": str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES,
    }
    with open(MERGED_DIR / "data.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        sys.exit("Falta ROBOFLOW_API_KEY (env var o --api-key). En Kaggle: añadir como Secret.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for spec in DATASET_SPECS:
        tag = spec["project"]
        print(f"→ {tag}")
        try:
            dataset_dir = download(spec, args.api_key)
        except Exception as e:
            print(f"  ✗ error descargando {tag}: {e}")
            continue

        names = load_class_names(dataset_dir / "data.yaml")
        remap = build_remap(names)
        unmapped = [n for i, n in enumerate(names) if remap[i] is None]
        if unmapped:
            print(f"  ⚠ clases sin mapear (se descartan): {unmapped}")
        remap_and_copy(dataset_dir, tag, remap)

    write_merged_yaml()
    n_train = len(list((MERGED_DIR / "train" / "images").glob("*"))) if (MERGED_DIR / "train" / "images").exists() else 0
    n_val = len(list((MERGED_DIR / "valid" / "images").glob("*"))) if (MERGED_DIR / "valid" / "images").exists() else 0
    print(f"\n✅ Dataset unificado en {MERGED_DIR} — train={n_train} imgs, valid={n_val} imgs")


if __name__ == "__main__":
    main()
