"""
Fine-tuning de YOLOv11m para detección de jugador/portero/árbitro en cámara
lateral/baja (Tier 3, ver docs/PLAN_player_tracking.md).

Parte de yolo11m.pt (COCO-pretrained de Ultralytics), no de best_100e.pt: son
arquitecturas distintas (v8 vs v11) y no hay transfer de pesos posible entre
ellas. Ver la sección T3.b del plan para el razonamiento de esta decisión.

Pensado para correr en Kaggle con GPU, después de datasets/download_datasets.py:
    1. python datasets/download_datasets.py   # genera datasets/merged/
    2. python datasets/train_jugadores.py

El checkpoint final queda en <project>/<name>/weights/best.pt (por defecto
datasets/runs/train_jugadores/weights/best.pt). Promoverlo a
best_jugadores_v2.pt en la raíz del repo es un paso manual (T3.d), después de
validar mAP + inspección visual con video_OG.mp4 (T3.c) — este script no lo
hace automáticamente.
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA_YAML = Path(__file__).parent / "merged" / "data.yaml"
DEFAULT_PROJECT = Path(__file__).parent / "runs"
DEFAULT_RUN_NAME = "train_jugadores"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.getenv("TRAIN_DATA_YAML", str(DEFAULT_DATA_YAML)))
    parser.add_argument("--base-weights", default=os.getenv("TRAIN_BASE_WEIGHTS", "yolo11m.pt"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("TRAIN_EPOCHS", "100")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("TRAIN_IMGSZ", "1280")))
    parser.add_argument("--batch", type=int, default=int(os.getenv("TRAIN_BATCH", "16")))
    parser.add_argument("--device", default=os.getenv("TRAIN_DEVICE", "0"))
    parser.add_argument("--patience", type=int, default=int(os.getenv("TRAIN_PATIENCE", "30")))
    parser.add_argument("--project", default=os.getenv("TRAIN_PROJECT", str(DEFAULT_PROJECT)))
    parser.add_argument("--name", default=os.getenv("TRAIN_RUN_NAME", DEFAULT_RUN_NAME))
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"No existe {args.data}. Corre primero: python datasets/download_datasets.py"
        )

    model = YOLO(args.base_weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Augmentation geométrico agresivo (T3.b) para cubrir variaciones de
        # ángulo/altura de cámara que el dataset no representa homogéneamente.
        degrees=15.0,
        shear=10.0,
        perspective=0.0008,
        translate=0.2,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print("\n✅ Entrenamiento terminado")
    print(f"   Pesos: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print("   (objetivo T3.c: mAP50 >= 0.85 en el holdout)")


if __name__ == "__main__":
    main()
