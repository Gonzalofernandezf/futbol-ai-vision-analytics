"""
Evalúa la calidad de detección de keypoints de cancha de modelo_cancha.pt
comparando contra anotaciones COCO exportadas desde Roboflow.

Métricas calculadas:
  - PCK (Percentage of Correct Keypoints) @ 5px, 10px, 20px
  - Error medio en píxeles por keypoint
  - Recall por keypoint (detección vs anotación GT)
  - Error de reproyección de homografía en metros

Uso:
    python eval/eval_keypoints.py \
        --annotations ruta/keypoints_coco.json \
        --frames-dir  ruta/frames/ \
        [--output demo_dashboard/eval_report.json] \
        [--conf 0.35] \
        [--pck-thresholds 5 10 20]

El JSON de salida es leído por la sección "Métricas ML" del dashboard.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Permite importar config desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Nombres canónicos de los 25 keypoints (mismos índices que target_vertices_dict) ──
KEYPOINT_NAMES = [
    "L_corner_tl",   # 0   esquina sup-izq
    "L_areas_tl",    # 1   área chica sup-izq
    "L_areas_tr",    # 2   área chica sup-der (izq del campo)
    "L_areab_tl",    # 3   área grande sup-izq
    "L_areab_tr",    # 4   área grande sup-der (izq del campo)
    "Center_top",    # 5   línea de medio campo, borde sup
    "R_areab_tl",    # 6   área grande sup-izq (der del campo)
    "R_areab_tr",    # 7   área grande sup-der
    "R_areas_tl",    # 8   área chica sup-izq (der del campo)
    "R_areas_tr",    # 9   área chica sup-der
    "R_corner_tr",   # 10  esquina sup-der
    "L_penalspot",   # 11  punto penal izq
    "Center",        # 12  centro del campo
    "R_penalspot",   # 13  punto penal der
    "L_corner_bl",   # 14  esquina inf-izq
    "L_areas_bl",    # 15  área chica inf-izq
    "L_areas_br",    # 16  área chica inf-der (izq del campo)
    "L_areab_bl",    # 17  área grande inf-izq
    "L_areab_br",    # 18  área grande inf-der (izq del campo)
    "Center_bottom", # 19  línea de medio campo, borde inf
    "R_areab_bl",    # 20  área grande inf-izq (der del campo)
    "R_areab_br",    # 21  área grande inf-der
    "R_areas_bl",    # 22  área chica inf-izq (der del campo)
    "R_areas_br",    # 23  área chica inf-der
    "R_corner_br",   # 24  esquina inf-der
]

# Coordenadas reales en metros — cancha sub-18/20 (100 m × 64 m)
TARGET_VERTICES = {
    0:  [0.0,   0.0],    1:  [0.0,  22.84],   2:  [5.5,  22.84],
    3:  [0.0,  11.84],   4:  [16.5, 11.84],   5:  [50.0,  0.0],
    6:  [83.5, 11.84],   7:  [100.0, 11.84],  8:  [94.5, 22.84],
    9:  [100.0, 22.84],  10: [100.0,  0.0],   11: [11.0, 32.0],
    12: [50.0, 32.0],    13: [89.0, 32.0],    14: [0.0,  64.0],
    15: [0.0,  41.16],   16: [5.5,  41.16],   17: [0.0,  52.16],
    18: [16.5, 52.16],   19: [50.0, 64.0],    20: [83.5, 52.16],
    21: [100.0, 52.16],  22: [94.5, 41.16],   23: [100.0, 41.16],
    24: [100.0, 64.0],
}


# ── Parser del formato COCO de Roboflow ────────────────────────────────────────

def load_coco_gt(coco_path: str):
    """
    Lee el JSON COCO exportado por Roboflow y devuelve:
      - gt_dict: {filename → lista de 25 dicts {x,y,v,name}}
      - kp_names: lista de 25 nombres (desde el schema del JSON o por defecto)

    Formato COCO de keypoints:
      annotation.keypoints = [x0,y0,v0, x1,y1,v1, ..., x24,y24,v24]
      v=0: no anotado  |  v=1: anotado no visible  |  v=2: anotado y visible
    """
    with open(coco_path) as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # Usa nombres del schema si están definidos y son 25
    kp_names = KEYPOINT_NAMES
    if coco.get("categories"):
        schema_names = coco["categories"][0].get("keypoints", [])
        if len(schema_names) == 25:
            kp_names = schema_names

    gt_dict = {}
    for ann in coco["annotations"]:
        fname = id_to_filename[ann["image_id"]]
        raw = ann["keypoints"]  # 75 valores: [x0,y0,v0, x1,y1,v1, ...]
        kps = []
        for i in range(25):
            x, y, v = raw[3 * i], raw[3 * i + 1], int(raw[3 * i + 2])
            kps.append({"x": float(x), "y": float(y), "v": v, "name": kp_names[i]})
        gt_dict[fname] = kps

    return gt_dict, kp_names


# ── Inferencia del modelo ──────────────────────────────────────────────────────

def run_model(model, frame: np.ndarray, conf_threshold: float) -> list[dict]:
    """
    Corre modelo_cancha.pt sobre un frame y devuelve 25 predicciones ordenadas.
    Si no hay detección, devuelve 25 entradas con detected=False.
    """
    not_detected = [{"x": 0.0, "y": 0.0, "conf": 0.0, "detected": False}
                    for _ in range(25)]

    results = model(frame, verbose=False)[0]

    if results.keypoints is None or len(results.keypoints.xy) == 0:
        return not_detected

    xy   = results.keypoints.xy[0].cpu().numpy()    # (25, 2)
    conf = results.keypoints.conf[0].cpu().numpy()  # (25,)

    if xy.shape[0] != 25:
        return not_detected

    preds = []
    for i in range(25):
        c = float(conf[i])
        preds.append({
            "x": float(xy[i][0]),
            "y": float(xy[i][1]),
            "conf": c,
            "detected": c >= conf_threshold,
        })
    return preds


# ── Calidad de homografía ──────────────────────────────────────────────────────

def compute_homography_error(gt_kps: list, pred_kps: list) -> float | None:
    """
    Construye la homografía con los keypoints GT y mide el error de reproyección
    de las predicciones del modelo en metros.

    Devuelve el error medio en metros, o None si no hay suficientes puntos.
    """
    # Puntos GT para construir la homografía de referencia
    gt_px, gt_world = [], []
    for i, kp in enumerate(gt_kps):
        if kp["v"] >= 1:  # solo keypoints anotados
            gt_px.append([kp["x"], kp["y"]])
            gt_world.append(TARGET_VERTICES[i])

    if len(gt_px) < 4:
        return None

    H, mask = cv2.findHomography(
        np.array(gt_px, dtype=np.float32),
        np.array(gt_world, dtype=np.float32),
        cv2.RANSAC, 5.0,
    )
    if H is None:
        return None

    # Proyectar predicciones con H y comparar con coordenadas canónicas
    errors = []
    for i, pred in enumerate(pred_kps):
        if not pred["detected"]:
            continue
        p_px = np.array([[pred["x"], pred["y"]]], dtype=np.float32).reshape(-1, 1, 2)
        p_world = cv2.perspectiveTransform(p_px, H)[0][0]
        canon   = np.array(TARGET_VERTICES[i], dtype=np.float32)
        errors.append(float(np.linalg.norm(p_world - canon)))

    return float(np.mean(errors)) if errors else None


# ── Evaluación principal ───────────────────────────────────────────────────────

def evaluate(coco_path: str, frames_dir: str, conf_threshold: float,
             pck_thresholds: list[int], output_path: str) -> dict:

    from ultralytics import YOLO

    model_path = str(Path(__file__).parent.parent / "modelo_cancha.pt")
    model = YOLO(model_path)

    print(f"Modelo:            {model_path}")
    print(f"Anotaciones GT:    {coco_path}")
    print(f"Carpeta frames:    {frames_dir}")
    print(f"Conf threshold:    {conf_threshold}")
    print(f"PCK thresholds:    {pck_thresholds} px\n")

    gt_dict, kp_names = load_coco_gt(coco_path)

    # Acumuladores por keypoint
    kp_stats = {
        name: {
            "n_gt_visible": 0,
            "n_detected":   0,
            "errors":       [],
            **{f"pck_{t}px": 0 for t in pck_thresholds},
        }
        for name in kp_names
    }

    frame_results = []
    n_homography_ok = 0

    for fname, gt_kps in gt_dict.items():
        # Buscar imagen en disco (con o sin subcarpeta)
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

        pred_kps = run_model(model, frame, conf_threshold)

        n_gt_visible = sum(1 for kp in gt_kps if kp["v"] >= 1)
        n_detected   = sum(1 for p  in pred_kps if p["detected"])
        matched      = 0
        frame_errors = []

        for i, (gt, pred) in enumerate(zip(gt_kps, pred_kps)):
            name = gt["name"]

            if gt["v"] < 1:
                continue  # keypoint no anotado en este frame, se omite

            kp_stats[name]["n_gt_visible"] += 1

            if not pred["detected"]:
                continue  # modelo no lo detectó → FN

            kp_stats[name]["n_detected"] += 1
            dist = np.hypot(pred["x"] - gt["x"], pred["y"] - gt["y"])
            kp_stats[name]["errors"].append(dist)
            frame_errors.append(dist)
            matched += 1

            for t in pck_thresholds:
                if dist <= t:
                    kp_stats[name][f"pck_{t}px"] += 1

        h_error = compute_homography_error(gt_kps, pred_kps)
        if h_error is not None:
            n_homography_ok += 1

        frame_results.append({
            "filename":           os.path.basename(fname),
            "n_gt_visible":       n_gt_visible,
            "n_detected":         n_detected,
            "n_matched":          matched,
            "mean_error_px":      round(float(np.mean(frame_errors)), 2) if frame_errors else None,
            "homography_ok":      h_error is not None,
            "homography_error_m": round(h_error, 3) if h_error is not None else None,
        })

        print(f"  {os.path.basename(fname):40s}  "
              f"GT={n_gt_visible:2d}  Pred={n_detected:2d}  Match={matched:2d}  "
              f"HomErr={f'{h_error:.2f}m' if h_error else 'N/A':>7s}")

    n_frames = len(frame_results)

    # Agregar por keypoint
    per_kp = {}
    all_errors = []
    for name in kp_names:
        s    = kp_stats[name]
        n_vis = s["n_gt_visible"]
        errs  = s["errors"]
        all_errors.extend(errs)
        per_kp[name] = {
            "n_gt_visible": n_vis,
            "n_detected":   s["n_detected"],
            "recall":       round(s["n_detected"] / n_vis, 4) if n_vis else None,
            "mean_error_px": round(float(np.mean(errs)), 2) if errs else None,
            **{
                f"pck_{t}px": round(s[f"pck_{t}px"] / n_vis, 4) if n_vis else None
                for t in pck_thresholds
            },
        }

    # Resumen global
    pck_global = {
        f"overall_pck_{t}px": round(float(np.mean(
            [v for v in [per_kp[n][f"pck_{t}px"] for n in kp_names] if v is not None]
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
            "conf_threshold":    conf_threshold,
            "pck_thresholds_px": pck_thresholds,
        },
        "summary": {
            **pck_global,
            "mean_error_px":             round(float(np.mean(all_errors)), 2) if all_errors else None,
            "median_error_px":           round(float(np.median(all_errors)), 2) if all_errors else None,
            "homography_failure_rate":   round(1 - n_homography_ok / n_frames, 4) if n_frames else None,
            "mean_homography_error_m":   round(float(np.mean(hom_errors)), 3) if hom_errors else None,
            "mean_kps_detected_per_frame": round(float(np.mean([r["n_detected"] for r in frame_results])), 1),
            "mean_kps_gt_per_frame":       round(float(np.mean([r["n_gt_visible"] for r in frame_results])), 1),
        },
        "per_keypoint": per_kp,
        "per_frame":    frame_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    pck_primary = pck_thresholds[1] if len(pck_thresholds) > 1 else pck_thresholds[0]
    print(f"\n{'='*60}")
    print(f"✅  Reporte guardado → {output_path}")
    print(f"    Frames evaluados        : {n_frames}")
    print(f"    PCK@{pck_primary}px (global)      : {s.get(f'overall_pck_{pck_primary}px')}")
    print(f"    Error medio (px)        : {s['mean_error_px']}")
    print(f"    Fallo homografía        : {s['homography_failure_rate']} ({n_frames - n_homography_ok}/{n_frames} frames)")
    print(f"    Error homografía (m)    : {s['mean_homography_error_m']}")
    print(f"{'='*60}")

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalúa detección de keypoints de cancha vs ground truth Roboflow"
    )
    parser.add_argument("--annotations", required=True,
                        help="Ruta al JSON COCO exportado desde Roboflow")
    parser.add_argument("--frames-dir",  required=True,
                        help="Carpeta con los frames JPG a evaluar")
    parser.add_argument("--output", default="demo_dashboard/eval_report.json",
                        help="Ruta del reporte JSON de salida (default: demo_dashboard/eval_report.json)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Umbral de confianza del modelo (default: 0.35)")
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
