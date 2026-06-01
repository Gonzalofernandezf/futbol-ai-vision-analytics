"""
Diagnóstico visual del modelo de campo.

Muestrea 1 frame cada SAMPLE_EVERY_SEC segundos del vídeo, corre
modelo_cancha.pt sobre cada frame, dibuja los bboxes con clase + confianza
y guarda las imágenes en diag_frames/.

Uso:
    python diag/diag_field_detections.py
    python diag/diag_field_detections.py --video otro_video.mp4 --conf 0.20
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── rutas relativas al raíz del proyecto ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import VIDEO_PATH, MODELO_CANCHA_PATH

# ── colores por clase (generados automáticamente) ────────────────────────────
_RNG = np.random.default_rng(42)

def _class_color(cls_id: int, n: int = 22):
    colors = (_RNG.integers(60, 240, size=(n, 3))).tolist()
    return tuple(int(c) for c in colors[cls_id % n])


def run(video_path: str, model_path: str, conf_thr: float,
        sample_every_sec: float, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    class_names = model.names
    n_classes = len(class_names)
    print(f"Modelo cargado: {n_classes} clases  |  conf>={conf_thr}")

    colors = {cid: _class_color(cid, n_classes) for cid in range(n_classes)}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir el vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * sample_every_sec))
    n_samples = total_frames // step
    print(f"Vídeo: {total_frames} frames @ {fps:.1f} fps  →  {n_samples} muestras "
          f"(cada {sample_every_sec}s)")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            results = model(frame, verbose=False, conf=conf_thr)[0]

            annotated = frame.copy()
            detections_this_frame = []

            if results.boxes is not None and len(results.boxes) > 0:
                boxes  = results.boxes.xyxy.cpu().numpy()
                cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                confs  = results.boxes.conf.cpu().numpy()

                # un solo bbox por clase (el de mayor confianza)
                best = {}
                for box, cid, cf in zip(boxes, cls_ids, confs):
                    if cid not in best or cf > best[cid][0]:
                        best[cid] = (cf, box)

                for cid, (cf, box) in sorted(best.items()):
                    x1, y1, x2, y2 = box.astype(int)
                    color = colors[cid]
                    label = f"{class_names[cid]} {cf:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    # fondo del texto
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 1, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    detections_this_frame.append(class_names[cid])

            # leyenda con clases NO detectadas (para ver qué falta)
            missing = [class_names[cid] for cid in range(n_classes)
                       if class_names[cid] not in detections_this_frame]
            t_sec = frame_idx / fps
            info = (f"t={t_sec:.1f}s  frame={frame_idx}  "
                    f"detectadas={len(detections_this_frame)}/{n_classes}")
            cv2.putText(annotated, info, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # lista de clases ausentes (columna derecha)
            x_miss = annotated.shape[1] - 220
            cv2.putText(annotated, "SIN DETECTAR:", (x_miss, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
            for i, name in enumerate(missing[:20]):
                cv2.putText(annotated, f"  {name}", (x_miss, 44 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1)

            out_path = out_dir / f"frame_{frame_idx:06d}_t{t_sec:.0f}s.jpg"
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"\nGuardadas {saved} imágenes en: {out_dir.resolve()}")
    print("Abre las imágenes y comprueba si los puntos débiles están visibles pero sin bbox.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnóstico visual modelo de campo")
    parser.add_argument("--video",  default=str(ROOT / VIDEO_PATH))
    parser.add_argument("--model",  default=str(ROOT / MODELO_CANCHA_PATH))
    parser.add_argument("--conf",   type=float, default=0.20,
                        help="Umbral de confianza (default 0.20, más bajo que producción)")
    parser.add_argument("--every",  type=float, default=5.0,
                        help="Segundos entre muestras (default 5)")
    parser.add_argument("--out",    default=str(ROOT / "diag_frames"))
    args = parser.parse_args()

    run(
        video_path=args.video,
        model_path=args.model,
        conf_thr=args.conf,
        sample_every_sec=args.every,
        out_dir=Path(args.out),
    )
