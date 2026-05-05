"""
Filtra un dataset YOLOv8-Pose eliminando keypoints concretos por índice.

Caso de uso:
  Nuestro dataset original de cancha tiene 25 keypoints. El keypoint #19
  (Center_bottom, justo bajo la cámara) aparece en solo el 0.5% de los frames
  → genera gradiente puro de ruido y sabotea la convergencia de pose_loss.

  Este script genera un dataset nuevo sin esos keypoints, re-indexado de forma
  contigua (YOLO requiere índices 0..N-1 sin huecos), y un data.yaml acorde.

Uso típico en Colab:
  python filter_keypoints_dataset.py \\
      --src /content/football-ai-4 \\
      --dst /content/football-ai-4-filtered \\
      --drop 19

Tras ejecutar, entrenar con:
  data="/content/football-ai-4-filtered/data.yaml"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def filter_label_file(src_path: Path, dst_path: Path, keep_indices: list[int]) -> None:
    """Reescribe un .txt de YOLO-Pose dejando solo los keypoints en keep_indices."""
    lines_out = []
    with src_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # Formato: cls cx cy w h  kpt0_x kpt0_y kpt0_v  kpt1_x kpt1_y kpt1_v ...
            header = parts[:5]
            kpt_flat = parts[5:]
            assert len(kpt_flat) % 3 == 0, f"Keypoints malformados en {src_path}"
            n_kpts = len(kpt_flat) // 3
            kpts = [kpt_flat[i * 3 : (i + 1) * 3] for i in range(n_kpts)]
            kept = [kpts[i] for i in keep_indices if i < n_kpts]
            new_parts = header + [v for triplet in kept for v in triplet]
            lines_out.append(" ".join(new_parts))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def copy_images(src_dir: Path, dst_dir: Path) -> int:
    """Copia (o linkea) imágenes manteniendo nombres."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in src_dir.iterdir():
        if img.is_file():
            shutil.copy2(img, dst_dir / img.name)
            count += 1
    return count


def process_split(src_root: Path, dst_root: Path, split: str, keep_indices: list[int]) -> tuple[int, int]:
    """Procesa un split (train/valid/test). Devuelve (n_imagenes, n_labels)."""
    src_img = src_root / split / "images"
    src_lbl = src_root / split / "labels"
    dst_img = dst_root / split / "images"
    dst_lbl = dst_root / split / "labels"

    n_imgs = 0
    n_lbls = 0

    if src_img.exists():
        n_imgs = copy_images(src_img, dst_img)

    if src_lbl.exists():
        for lbl in src_lbl.glob("*.txt"):
            filter_label_file(lbl, dst_lbl / lbl.name, keep_indices)
            n_lbls += 1

    return n_imgs, n_lbls


def write_data_yaml(src_root: Path, dst_root: Path, n_kpts_new: int) -> None:
    """Genera data.yaml nuevo con kpt_shape actualizado."""
    src_yaml_path = src_root / "data.yaml"
    if not src_yaml_path.exists():
        raise FileNotFoundError(f"No existe {src_yaml_path}")

    with src_yaml_path.open("r") as f:
        data = yaml.safe_load(f)

    data["kpt_shape"] = [n_kpts_new, 3]
    data["flip_idx"] = list(range(n_kpts_new))  # identidad — sin flip simétrico

    # Las rutas relativas se mantienen iguales (../train/images, etc.)
    dst_yaml_path = dst_root / "data.yaml"
    with dst_yaml_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="Carpeta raíz del dataset original (con data.yaml)")
    parser.add_argument("--dst", required=True, type=Path, help="Carpeta raíz del dataset filtrado")
    parser.add_argument("--drop", required=True, type=int, nargs="+",
                        help="Índices de keypoints a eliminar (ej. --drop 19 ó --drop 14 19 24)")
    parser.add_argument("--n-kpts", type=int, default=25,
                        help="Número total de keypoints en el dataset original (default: 25)")
    args = parser.parse_args()

    drop_set = set(args.drop)
    keep_indices = [i for i in range(args.n_kpts) if i not in drop_set]
    n_kpts_new = len(keep_indices)

    print(f"Dataset origen: {args.src}")
    print(f"Dataset destino: {args.dst}")
    print(f"Keypoints originales: {args.n_kpts}")
    print(f"Eliminando índices: {sorted(drop_set)}")
    print(f"Keypoints resultantes: {n_kpts_new}")
    print(f"Mapeo nuevo→viejo: {dict(enumerate(keep_indices))}")
    print()

    args.dst.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    total_lbls = 0
    for split in ("train", "valid", "test"):
        if not (args.src / split).exists():
            print(f"  [skip] {split}/ no existe")
            continue
        n_imgs, n_lbls = process_split(args.src, args.dst, split, keep_indices)
        print(f"  {split}: {n_imgs} imágenes, {n_lbls} labels procesados")
        total_imgs += n_imgs
        total_lbls += n_lbls

    write_data_yaml(args.src, args.dst, n_kpts_new)

    print()
    print(f"Total: {total_imgs} imágenes, {total_lbls} labels")
    print(f"data.yaml generado en {args.dst / 'data.yaml'}")
    print()
    print("Listo. Entrenar con:")
    print(f'  data="{args.dst / "data.yaml"}"')


if __name__ == "__main__":
    main()
