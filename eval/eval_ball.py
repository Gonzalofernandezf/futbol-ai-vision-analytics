"""
Evalúa la calidad de detección de balón de BallDetector (best_ball_v1.pt)
comparando contra anotaciones COCO Detection exportadas desde Roboflow
(clase única "ball", una bbox por frame donde el balón sea visible; los
frames donde el balón está oculto/fuera de cuadro no llevan anotación).

A diferencia de eval_keypoints.py, acá también importa medir falsos
positivos: frames sin balón en el GT donde el detector igual dispara
(manchas de césped, medias blancas — ver Gate 2/3 en ball_detector.py).

Usa el mismo BallDetector de producción (ball_detection/ball_detector.py),
así que respeta automáticamente todos los umbrales activos en config.py
(YOLO_BALL_CONF, BALL_MIN_CONF, BALL_MIN/MAX_BBOX_PX, BALL_MIN/MAX_ASPECT,
CROWD_MASK_Y_PX). Para probar otros umbrales, sobreescribilos por env var
antes de correr (ej. YOLO_BALL_CONF=0.5 python eval/eval_ball.py ...).

Nota — alcance: el set de frames usado acá es una muestra dispersa del
partido (no una secuencia continua), así que este script solo mide
detección cruda por frame. NO mide continuidad de tracking ni el efecto
de interpolación / filtros de velocidad-estático (Trackers.tracker),
que requieren una secuencia de frames consecutivos para tener sentido.
Si más adelante se anota un tramo continuo, eso se evalúa aparte.

Uso:
    python eval/eval_ball.py
        --annotations ruta/coco_annotations.json
        --frames-dir  ruta/frames/
        --output futbol-ai-dashboard/public/eval_ball_report.json
        --pck-thresholds 5 10 20
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as _cfg


def load_coco_gt(coco_path: str):
    """
    Lee JSON COCO Detection de Roboflow y devuelve:
      gt_dict: {filename → (cx_px, cy_px) | None}  None = balón no anotado (ausente/oculto)
      fname_to_wh: {filename → (width, height)}  resolución con la que Roboflow anotó

    A diferencia del eval de cancha, acá se listan TODAS las imágenes del
    export (con o sin anotación) porque la ausencia de balón es una señal
    válida (frame sin balón visible) usada para medir falsos positivos.
    """
    with open(coco_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    fname_to_wh = {
        img["file_name"]: (img["width"], img["height"]) for img in coco["images"]
    }

    gt_dict = {fname: None for fname in id_to_filename.values()}

    seen_multi = 0
    for ann in coco["annotations"]:
        fname = id_to_filename[ann["image_id"]]
        x, y, w, h = ann["bbox"]  # COCO: top-left + w,h
        cx, cy = x + w / 2, y + h / 2
        if gt_dict.get(fname) is not None:
            seen_multi += 1
            continue  # una sola bola por frame — nos quedamos con la primera anotación
        gt_dict[fname] = (cx, cy)

    if seen_multi:
        print(f"  [WARN] {seen_multi} frame(s) con más de una anotación de balón — se ignoraron las extra.")

    return gt_dict, fname_to_wh


def _get_git_info() -> dict:
    try:
        raw = subprocess.check_output(
            ["git", "log", "-1", "--format=%H|%h|%s|%ci"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        full_hash, short_hash, message, date = raw.split("|", 3)
        return {"hash": full_hash, "short_hash": short_hash,
                "message": message, "date": date}
    except Exception:
        return {"hash": None, "short_hash": None, "message": None, "date": None}


def _save_archive(report: dict, output_path: str) -> None:
    git = report["metadata"].get("git_commit") or {}
    short_hash = git.get("short_hash") or "nogit"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports_dir = Path(output_path).parent / "reports_ball"
    reports_dir.mkdir(exist_ok=True)

    filename = f"{short_hash}_{ts}.json"
    dest = reports_dir / filename
    with open(dest, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Archivo archivado       → {dest}")

    manifest_path = reports_dir / "manifest.json"
    manifest = []
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = []

    s = report["summary"]
    m = report["metadata"]
    manifest.append({
        "filename":            filename,
        "git_hash":            git.get("short_hash"),
        "git_message":         git.get("message"),
        "git_date":            git.get("date"),
        "timestamp":           m["timestamp"],
        "model":               m["model"],
        "recall":              s.get("recall"),
        "precision":           s.get("precision"),
        "overall_pck_10px":    s.get("overall_pck_10px"),
        "false_positive_rate": s.get("false_positive_rate"),
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    Manifest actualizado    → {manifest_path}  ({len(manifest)} entradas)")


def _append_history(report: dict, main_output_path: str) -> None:
    s   = report["summary"]
    m   = report["metadata"]
    git = m.get("git_commit") or {}

    entry = {
        "timestamp":            m["timestamp"],
        "git_hash":             git.get("short_hash"),
        "git_message":          git.get("message"),
        "git_date":             git.get("date"),
        "model":                m["model"],
        "n_frames":             m["n_frames"],
        "n_gt_positive":        m["n_gt_positive"],
        "n_gt_negative":        m["n_gt_negative"],
        "recall":               s.get("recall"),
        "precision":            s.get("precision"),
        "f1":                   s.get("f1"),
        "overall_pck_5px":      s.get("overall_pck_5px"),
        "overall_pck_10px":     s.get("overall_pck_10px"),
        "overall_pck_20px":     s.get("overall_pck_20px"),
        "mean_error_px":        s.get("mean_error_px"),
        "false_positive_rate":  s.get("false_positive_rate"),
    }

    history_path = Path(main_output_path).parent / "eval_ball_history.json"
    history = []
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(entry)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"    Historial actualizado   → {history_path}  ({len(history)} entradas)")


def evaluate(coco_path: str, frames_dir: str, pck_thresholds: list[int],
             output_path: str) -> dict:

    from ball_detection.ball_detector import BallDetector

    detector = BallDetector(model_path=_cfg.BALL_MODEL_PATH)

    print(f"Modelo:            {_cfg.BALL_MODEL_PATH}")
    print(f"Anotaciones GT:    {coco_path}")
    print(f"Carpeta frames:    {frames_dir}")
    print(f"Umbrales activos:  YOLO_BALL_CONF={_cfg.YOLO_BALL_CONF}  BALL_MIN_CONF={_cfg.BALL_MIN_CONF}  "
          f"BBOX=[{_cfg.BALL_MIN_BBOX_PX},{_cfg.BALL_MAX_BBOX_PX}]px  "
          f"ASPECT=[{_cfg.BALL_MIN_ASPECT},{_cfg.BALL_MAX_ASPECT}]")
    print(f"PCK thresholds:    {pck_thresholds} px\n")

    gt_dict, fname_to_wh = load_coco_gt(coco_path)

    fnames, frames = [], []
    for fname in gt_dict:
        img_path = os.path.join(frames_dir, fname)
        if not os.path.exists(img_path):
            img_path = os.path.join(frames_dir, os.path.basename(fname))
        if not os.path.exists(img_path):
            print(f"  [WARN] No encontrado: {fname}")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARN] Error al leer: {img_path}")
            continue

        fnames.append(fname)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No se pudo cargar ningún frame del set de evaluación.")

    ball_tracks = detector.get_ball_tracks(frames)

    per_frame = []
    tp = fp = fn = tn = 0
    errors = []
    pck_hits = {t: 0 for t in pck_thresholds}
    confidences_tp = []

    for fname, frame, frame_ball in zip(fnames, frames, ball_tracks):
        gt = gt_dict[fname]

        # Reescala el GT si Roboflow lo anotó a otra resolución que la del
        # frame real evaluado (mismo problema que en eval_keypoints.py).
        if gt is not None:
            ann_w, ann_h = fname_to_wh.get(fname, (frame.shape[1], frame.shape[0]))
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / ann_w
            scale_y = frame_h / ann_h
            if scale_x != 1.0 or scale_y != 1.0:
                gt = (gt[0] * scale_x, gt[1] * scale_y)

        pred = frame_ball.get(1) if frame_ball else None
        outcome = None
        error_px = None
        conf = pred["confidence"] if pred else None

        if gt is not None and pred is not None:
            bbox = pred["bbox"]
            pred_cx, pred_cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            error_px = float(np.hypot(pred_cx - gt[0], pred_cy - gt[1]))
            errors.append(error_px)
            confidences_tp.append(conf)
            tp += 1
            outcome = "TP"
            for t in pck_thresholds:
                if error_px <= t:
                    pck_hits[t] += 1
        elif gt is not None and pred is None:
            fn += 1
            outcome = "FN"
        elif gt is None and pred is not None:
            fp += 1
            outcome = "FP"
        else:
            tn += 1
            outcome = "TN"

        per_frame.append({
            "filename":   os.path.basename(fname),
            "gt_present": gt is not None,
            "pred_present": pred is not None,
            "outcome":    outcome,
            "error_px":   round(error_px, 2) if error_px is not None else None,
            "confidence": round(conf, 3) if conf is not None else None,
        })

    n_frames = len(fnames)
    n_gt_positive = tp + fn
    n_gt_negative = fp + tn

    print(f"\n{'='*60}")
    print(f"Frames evaluados        : {n_frames}  (con balón: {n_gt_positive}, sin balón: {n_gt_negative})")
    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"{'='*60}\n")

    recall    = round(tp / n_gt_positive, 4) if n_gt_positive else None
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    f1 = round(2 * precision * recall / (precision + recall), 4) if (recall and precision) else None
    fp_rate = round(fp / n_gt_negative, 4) if n_gt_negative else None

    pck_global = {
        f"overall_pck_{t}px": round(pck_hits[t] / n_gt_positive, 4) if n_gt_positive else None
        for t in pck_thresholds
    }

    report = {
        "metadata": {
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "n_frames":          n_frames,
            "n_gt_positive":     n_gt_positive,
            "n_gt_negative":     n_gt_negative,
            "model":             os.path.basename(_cfg.BALL_MODEL_PATH),
            "model_task":        "detect",
            "pck_thresholds_px": pck_thresholds,
            "thresholds": {
                "yolo_ball_conf": _cfg.YOLO_BALL_CONF,
                "ball_min_conf":  _cfg.BALL_MIN_CONF,
                "bbox_px":        [_cfg.BALL_MIN_BBOX_PX, _cfg.BALL_MAX_BBOX_PX],
                "aspect":         [_cfg.BALL_MIN_ASPECT, _cfg.BALL_MAX_ASPECT],
            },
        },
        "summary": {
            "recall":              recall,
            "precision":           precision,
            "f1":                  f1,
            "false_positive_rate": fp_rate,
            **pck_global,
            "mean_error_px":        round(float(np.mean(errors)), 2) if errors else None,
            "median_error_px":      round(float(np.median(errors)), 2) if errors else None,
            "mean_confidence_tp":   round(float(np.mean(confidences_tp)), 3) if confidences_tp else None,
        },
        "per_frame": per_frame,
    }

    report["metadata"]["git_commit"] = _get_git_info()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    _save_archive(report, output_path)
    _append_history(report, output_path)

    s = report["summary"]
    print(f"✅  Reporte guardado → {output_path}")
    print(f"    Recall               : {s['recall']}")
    print(f"    Precision            : {s['precision']}")
    print(f"    F1                   : {s['f1']}")
    print(f"    Falsos positivos     : {s['false_positive_rate']}  ({fp}/{n_gt_negative} frames sin balón)")
    print(f"    PCK@10px             : {s.get('overall_pck_10px')}")
    print(f"    Error medio (px)     : {s['mean_error_px']}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalúa detección de balón (BallDetector) vs ground truth Roboflow (COCO Detection)"
    )
    parser.add_argument("--annotations", required=True,
                        help="Ruta al JSON COCO Detection exportado desde Roboflow (clase única 'ball')")
    parser.add_argument("--frames-dir",  required=True,
                        help="Carpeta con los frames JPG a evaluar")
    parser.add_argument("--output", default="futbol-ai-dashboard/public/eval_ball_report.json",
                        help="Ruta del reporte JSON de salida")
    parser.add_argument("--pck-thresholds", type=int, nargs="+", default=[5, 10, 20],
                        help="Tolerancias PCK en píxeles (default: 5 10 20)")
    args = parser.parse_args()

    evaluate(
        coco_path      = args.annotations,
        frames_dir     = args.frames_dir,
        pck_thresholds = args.pck_thresholds,
        output_path    = args.output,
    )
