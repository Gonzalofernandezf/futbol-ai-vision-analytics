"""
Evalúa la calidad de detección de keypoints de cancha de modelo_cancha.pt
comparando contra anotaciones COCO Detection exportadas desde Roboflow.

El modelo nuevo es YOLOv8 detect con 22 clases (una por punto del campo).
El GT se exporta desde Roboflow en formato COCO Detection (no pose).

Métricas calculadas:
  - Recall por clase  (¿detectó el keypoint en el frame donde estaba anotado?)
  - PCK@5px / @10px / @20px  (¿el centro predicho está a ≤N px del centro GT?)
  - Error medio en píxeles por clase
  - Error de reproyección de homografía en metros

Uso:
    python eval/eval_keypoints.py
        --annotations ruta/coco_annotations.json
        --frames-dir  ruta/frames/
        --output futbol-ai-dashboard/public/eval_report.json
        --conf 0.25
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

# Nombres de las 22 clases en orden de Roboflow (alfabético)
KEYPOINT_NAMES = [
    "big_l_far_bot",    # 0
    "big_l_far_top",    # 1
    "big_l_goal_bot",   # 2
    "big_l_goal_top",   # 3
    "big_r_far_bot",    # 4
    "big_r_far_top",    # 5
    "big_r_goal_bot",   # 6
    "big_r_goal_top",   # 7
    "center_spot",      # 8
    "circle_bot",       # 9
    "circle_top",       # 10
    "corner_bl",        # 11
    "corner_br",        # 12
    "corner_tl",        # 13
    "corner_tr",        # 14
    "halfline_top",     # 15
    "penalty_l",        # 16
    "penalty_r",        # 17
    "small_l_far_bot",  # 18
    "small_l_far_top",  # 19
    "small_r_far_bot",  # 20
    "small_r_far_top",  # 21
]

TARGET_VERTICES = {
    0:  [16.5,  52.16],  # big_l_far_bot
    1:  [16.5,  11.84],  # big_l_far_top
    2:  [0.0,   52.16],  # big_l_goal_bot
    3:  [0.0,   11.84],  # big_l_goal_top
    4:  [83.5,  52.16],  # big_r_far_bot
    5:  [83.5,  11.84],  # big_r_far_top
    6:  [100.0, 52.16],  # big_r_goal_bot
    7:  [100.0, 11.84],  # big_r_goal_top
    8:  [50.0,  32.0],   # center_spot
    9:  [50.0,  41.15],  # circle_bot
    10: [50.0,  22.85],  # circle_top
    11: [0.0,   64.0],   # corner_bl
    12: [100.0, 64.0],   # corner_br
    13: [0.0,   0.0],    # corner_tl
    14: [100.0, 0.0],    # corner_tr
    15: [50.0,  0.0],    # halfline_top
    16: [11.0,  32.0],   # penalty_l
    17: [89.0,  32.0],   # penalty_r
    18: [5.5,   41.16],  # small_l_far_bot
    19: [5.5,   22.84],  # small_l_far_top
    20: [94.5,  41.16],  # small_r_far_bot
    21: [94.5,  22.84],  # small_r_far_top
}

N_CLASSES = len(KEYPOINT_NAMES)


def load_coco_gt(coco_path: str):
    """
    Lee JSON COCO Detection de Roboflow y devuelve:
      gt_dict: {filename → {class_id: (cx_px, cy_px)}}  (en la resolución del JSON, no la real)
      cat_id_to_class_id: {category_id_coco → class_id (0-21)}
      fname_to_wh: {filename → (width, height)}  resolución con la que Roboflow anotó

    Formato real de export de Roboflow para este proyecto: una anotación
    (bbox) por punto de cancha, con una categoría por punto (22 categorías
    con nombre + 1 supercategoría genérica "football-pitch-detection").
    Mapeamos categoría → class_id por NOMBRE (no por índice posicional),
    para no depender de que Roboflow numere las category_id en el mismo
    orden que KEYPOINT_NAMES.
    """
    with open(coco_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    fname_to_wh = {
        img["file_name"]: (img["width"], img["height"]) for img in coco["images"]
    }

    name_to_class_id = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    cat_id_to_class_id = {}
    for cat in coco["categories"]:
        cid = name_to_class_id.get(cat["name"])
        if cid is not None:
            cat_id_to_class_id[cat["id"]] = cid

    gt_dict = {}
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in cat_id_to_class_id:
            continue
        class_id = cat_id_to_class_id[cat_id]
        fname = id_to_filename[ann["image_id"]]
        x, y, w, h = ann["bbox"]  # COCO: top-left + w,h
        cx, cy = x + w / 2, y + h / 2
        if fname not in gt_dict:
            gt_dict[fname] = {}
        gt_dict[fname][class_id] = (cx, cy)

    return gt_dict, cat_id_to_class_id, fname_to_wh


def run_model(model, frame: np.ndarray, conf_threshold: float) -> dict:
    """
    Corre modelo_cancha.pt (detect) sobre un frame.
    Devuelve {class_id: (cx, cy, conf)} con el mejor bbox por clase.
    """
    results = model(frame, verbose=False)[0]
    best = {}

    if results.boxes is None or len(results.boxes) == 0:
        return best

    xyxy   = results.boxes.xyxy.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    confs  = results.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
        if conf < conf_threshold:
            continue
        if cls_id not in best or conf > best[cls_id][2]:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            best[cls_id] = (cx, cy, float(conf))

    return best


def compute_homography_error(gt_pts: dict, pred_pts: dict) -> float | None:
    """
    Construye homografía con puntos GT y mide error de reproyección
    de las predicciones en metros.
    """
    gt_px, gt_world = [], []
    for cls_id, (cx, cy) in gt_pts.items():
        gt_px.append([cx, cy])
        gt_world.append(TARGET_VERTICES[cls_id])

    if len(gt_px) < 4:
        return None

    H, _ = cv2.findHomography(
        np.array(gt_px, dtype=np.float32),
        np.array(gt_world, dtype=np.float32),
        cv2.RANSAC, 5.0,
    )
    if H is None:
        return None

    errors = []
    for cls_id, (cx, cy, _) in pred_pts.items():
        if cls_id not in TARGET_VERTICES:
            continue
        p_px    = np.array([[[cx, cy]]], dtype=np.float32)
        p_world = cv2.perspectiveTransform(p_px, H)[0][0]
        canon   = np.array(TARGET_VERTICES[cls_id], dtype=np.float32)
        errors.append(float(np.linalg.norm(p_world - canon)))

    return float(np.mean(errors)) if errors else None


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

    reports_dir = Path(output_path).parent / "reports"
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
        "filename":               filename,
        "git_hash":               git.get("short_hash"),
        "git_message":            git.get("message"),
        "git_date":               git.get("date"),
        "timestamp":              m["timestamp"],
        "model":                  m["model"],
        "overall_recall":         s.get("overall_recall"),
        "overall_pck_10px":       s.get("overall_pck_10px"),
        "homography_failure_rate": s.get("homography_failure_rate"),
    })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    Manifest actualizado    → {manifest_path}  ({len(manifest)} entradas)")


def _append_history(report: dict, main_output_path: str) -> None:
    s   = report["summary"]
    m   = report["metadata"]
    git = m.get("git_commit") or {}

    entry = {
        "timestamp":                   m["timestamp"],
        "git_hash":                    git.get("short_hash"),
        "git_message":                 git.get("message"),
        "git_date":                    git.get("date"),
        "model":                       m["model"],
        "conf_threshold":              m["conf_threshold"],
        "n_frames":                    m["n_frames"],
        "overall_recall":              s.get("overall_recall"),
        "overall_pck_5px":             s.get("overall_pck_5px"),
        "overall_pck_10px":            s.get("overall_pck_10px"),
        "overall_pck_20px":            s.get("overall_pck_20px"),
        "mean_error_px":               s.get("mean_error_px"),
        "mean_homography_error_m":     s.get("mean_homography_error_m"),
        "homography_failure_rate":     s.get("homography_failure_rate"),
        "mean_kps_detected_per_frame": s.get("mean_kps_detected_per_frame"),
    }

    history_path = Path(main_output_path).parent / "eval_history.json"
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


def evaluate(coco_path: str, frames_dir: str, conf_threshold: float,
             pck_thresholds: list[int], output_path: str) -> dict:

    from ultralytics import YOLO

    model_path = str(Path(__file__).parent.parent / "modelo_cancha.pt")
    model = YOLO(model_path)

    print(f"Modelo:            {model_path}  (task={model.task})")
    print(f"Clases:            {model.names}")
    print(f"Anotaciones GT:    {coco_path}")
    print(f"Carpeta frames:    {frames_dir}")
    print(f"Conf threshold:    {conf_threshold}")
    print(f"PCK thresholds:    {pck_thresholds} px\n")

    gt_dict, _, fname_to_wh = load_coco_gt(coco_path)

    kp_stats = {
        name: {
            "n_gt": 0,
            "n_detected": 0,
            "errors": [],
            **{f"pck_{t}px": 0 for t in pck_thresholds},
        }
        for name in KEYPOINT_NAMES
    }

    frame_results = []
    n_homography_ok = 0

    for fname, gt_pts in gt_dict.items():
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

        # Las anotaciones de Roboflow están en la resolución con la que se
        # subió el frame al proyecto, que puede diferir de la resolución real
        # del JPG evaluado (p.ej. si Roboflow reescaló al importar). Sin este
        # reescalado, un frame anotado a 640x640 y evaluado a 1920x1080 arroja
        # errores en píxeles ~3x mayores de los reales.
        ann_w, ann_h = fname_to_wh.get(fname, (frame.shape[1], frame.shape[0]))
        frame_h, frame_w = frame.shape[:2]
        scale_x = frame_w / ann_w
        scale_y = frame_h / ann_h
        if scale_x != 1.0 or scale_y != 1.0:
            gt_pts = {
                cls_id: (cx * scale_x, cy * scale_y)
                for cls_id, (cx, cy) in gt_pts.items()
            }

        pred_pts = run_model(model, frame, conf_threshold)

        n_gt      = len(gt_pts)
        n_detected = len(pred_pts)
        matched    = 0
        frame_errors = []

        for cls_id, (gt_cx, gt_cy) in gt_pts.items():
            name = KEYPOINT_NAMES[cls_id]
            kp_stats[name]["n_gt"] += 1

            if cls_id not in pred_pts:
                continue

            kp_stats[name]["n_detected"] += 1
            pred_cx, pred_cy, _ = pred_pts[cls_id]
            dist = float(np.hypot(pred_cx - gt_cx, pred_cy - gt_cy))
            kp_stats[name]["errors"].append(dist)
            frame_errors.append(dist)
            matched += 1

            for t in pck_thresholds:
                if dist <= t:
                    kp_stats[name][f"pck_{t}px"] += 1

        h_error = compute_homography_error(gt_pts, pred_pts)
        if h_error is not None:
            n_homography_ok += 1

        frame_results.append({
            "filename":           os.path.basename(fname),
            "n_gt":               n_gt,
            "n_detected":         n_detected,
            "n_matched":          matched,
            "mean_error_px":      round(float(np.mean(frame_errors)), 2) if frame_errors else None,
            "homography_ok":      h_error is not None,
            "homography_error_m": round(h_error, 3) if h_error is not None else None,
        })

        print(f"  {os.path.basename(fname):40s}  "
              f"GT={n_gt:2d}  Pred={n_detected:2d}  Match={matched:2d}  "
              f"HomErr={f'{h_error:.2f}m' if h_error else 'N/A':>7s}")

    n_frames = len(frame_results)

    per_kp = {}
    all_errors = []
    for cls_id, name in enumerate(KEYPOINT_NAMES):
        s     = kp_stats[name]
        n_gt  = s["n_gt"]
        errs  = s["errors"]
        all_errors.extend(errs)
        per_kp[name] = {
            "class_id":      cls_id,
            "n_gt":          n_gt,
            "n_detected":    s["n_detected"],
            "recall":        round(s["n_detected"] / n_gt, 4) if n_gt else None,
            "mean_error_px": round(float(np.mean(errs)), 2) if errs else None,
            **{
                f"pck_{t}px": round(s[f"pck_{t}px"] / n_gt, 4) if n_gt else None
                for t in pck_thresholds
            },
        }

    recalls = [v["recall"] for v in per_kp.values() if v["recall"] is not None]
    pck_global = {
        f"overall_pck_{t}px": round(float(np.mean(
            [v for v in [per_kp[n][f"pck_{t}px"] for n in KEYPOINT_NAMES] if v is not None]
        )), 4)
        for t in pck_thresholds
    }

    hom_errors = [r["homography_error_m"]
                  for r in frame_results if r["homography_error_m"] is not None]

    report = {
        "metadata": {
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "n_frames":          n_frames,
            "model":             "modelo_cancha.pt",
            "model_task":        "detect",
            "n_classes":         N_CLASSES,
            "conf_threshold":    conf_threshold,
            "pck_thresholds_px": pck_thresholds,
        },
        "summary": {
            "overall_recall":              round(float(np.mean(recalls)), 4) if recalls else None,
            **pck_global,
            "mean_error_px":               round(float(np.mean(all_errors)), 2) if all_errors else None,
            "median_error_px":             round(float(np.median(all_errors)), 2) if all_errors else None,
            "homography_failure_rate":     round(1 - n_homography_ok / n_frames, 4) if n_frames else None,
            "mean_homography_error_m":     round(float(np.mean(hom_errors)), 3) if hom_errors else None,
            "mean_kps_detected_per_frame": round(float(np.mean([r["n_detected"] for r in frame_results])), 1),
            "mean_kps_gt_per_frame":       round(float(np.mean([r["n_gt"] for r in frame_results])), 1),
        },
        "per_keypoint": per_kp,
        "per_frame":    frame_results,
    }

    report["metadata"]["git_commit"] = _get_git_info()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    _save_archive(report, output_path)
    _append_history(report, output_path)

    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"✅  Reporte guardado → {output_path}")
    print(f"    Frames evaluados        : {n_frames}")
    print(f"    Recall global           : {s.get('overall_recall')}")
    print(f"    PCK@10px (global)       : {s.get('overall_pck_10px')}")
    print(f"    Error medio (px)        : {s['mean_error_px']}")
    print(f"    Fallo homografía        : {s['homography_failure_rate']} ({n_frames - n_homography_ok}/{n_frames} frames)")
    print(f"    Error homografía (m)    : {s['mean_homography_error_m']}")
    print(f"{'='*60}")

    # Tabla de recall por clase (ordenada de peor a mejor)
    print("\n  Recall por clase (peor → mejor):")
    sorted_kp = sorted(per_kp.items(), key=lambda x: (x[1]["recall"] or 0))
    for name, stats in sorted_kp:
        r = stats["recall"]
        bar = "█" * int((r or 0) * 20) + "░" * (20 - int((r or 0) * 20))
        print(f"  {stats['class_id']:2d} {name:20s} {bar} {(r or 0)*100:5.1f}%  (GT={stats['n_gt']})")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalúa detección de keypoints de cancha vs ground truth Roboflow (COCO Detection)"
    )
    parser.add_argument("--annotations", required=True,
                        help="Ruta al JSON COCO Detection exportado desde Roboflow")
    parser.add_argument("--frames-dir",  required=True,
                        help="Carpeta con los frames JPG a evaluar")
    parser.add_argument("--output", default="futbol-ai-dashboard/public/eval_report.json",
                        help="Ruta del reporte JSON de salida")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral de confianza del modelo (default: 0.25)")
    parser.add_argument("--pck-thresholds", type=int, nargs="+", default=[5, 10, 20],
                        help="Tolerancias PCK en píxeles (default: 5 10 20)")
    args = parser.parse_args()

    evaluate(
        coco_path       = args.annotations,
        frames_dir      = args.frames_dir,
        conf_threshold  = args.conf,
        pck_thresholds  = args.pck_thresholds,
        output_path     = args.output,
    )
