#!/usr/bin/env python3
"""
Reconstruye eval_history.json a partir de todos los reportes individuales
guardados en eval/results/ (o eval/results_ball/ con --kind ball) y
actualiza los artefactos del dashboard:

  - futbol-ai-dashboard/public/eval_report.json        (el más reciente)
  - futbol-ai-dashboard/public/eval_history.json        (todos los runs ordenados)
  - futbol-ai-dashboard/public/eval_ball_report.json    (--kind ball)
  - futbol-ai-dashboard/public/eval_ball_history.json   (--kind ball)

Flujo de uso:
  1. Descarga eval_report.json (eval_keypoints.py) o eval_ball_report.json
     (eval_ball.py) de Kaggle.
  2. Renómbralo con fecha/hash y cópialo a eval/results/ o eval/results_ball/
     (ej: eval/results/2026-07-01_abc1234.json)
  3. Ejecuta:  python eval/build_history.py [--kind keypoints|ball]
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

KIND_CONFIG = {
    "keypoints": {
        "results_dir":  REPO_ROOT / "eval" / "results",
        "report_name":  "eval_report.json",
        "history_name": "eval_history.json",
    },
    "ball": {
        "results_dir":  REPO_ROOT / "eval" / "results_ball",
        "report_name":  "eval_ball_report.json",
        "history_name": "eval_ball_history.json",
    },
}
DEFAULT_OUT = REPO_ROOT / "futbol-ai-dashboard" / "public"


def _extract_entry_keypoints(report: dict) -> dict:
    s = report["summary"]
    m = report["metadata"]
    git = m.get("git_commit") or {}
    return {
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


def _extract_entry_ball(report: dict) -> dict:
    s = report["summary"]
    m = report["metadata"]
    git = m.get("git_commit") or {}
    return {
        "timestamp":           m["timestamp"],
        "git_hash":            git.get("short_hash"),
        "git_message":         git.get("message"),
        "git_date":            git.get("date"),
        "model":               m["model"],
        "n_frames":            m["n_frames"],
        "n_gt_positive":       m.get("n_gt_positive"),
        "n_gt_negative":       m.get("n_gt_negative"),
        "recall":              s.get("recall"),
        "precision":           s.get("precision"),
        "f1":                  s.get("f1"),
        "overall_pck_5px":     s.get("overall_pck_5px"),
        "overall_pck_10px":    s.get("overall_pck_10px"),
        "overall_pck_20px":    s.get("overall_pck_20px"),
        "mean_error_px":       s.get("mean_error_px"),
        "false_positive_rate": s.get("false_positive_rate"),
    }


_EXTRACT_ENTRY = {
    "keypoints": _extract_entry_keypoints,
    "ball":      _extract_entry_ball,
}


def _print_summary_keypoints(s: dict) -> None:
    print(f"  Recall  : {(s.get('overall_recall') or 0) * 100:.1f}%")
    print(f"  PCK@10  : {(s.get('overall_pck_10px') or 0) * 100:.1f}%")
    print(f"  Error   : {s.get('mean_error_px')} px")
    print(f"  Hom.err : {s.get('mean_homography_error_m')} m")


def _print_summary_ball(s: dict) -> None:
    print(f"  Recall     : {(s.get('recall') or 0) * 100:.1f}%")
    print(f"  Precision  : {(s.get('precision') or 0) * 100:.1f}%")
    print(f"  PCK@10     : {(s.get('overall_pck_10px') or 0) * 100:.1f}%")
    print(f"  Error      : {s.get('mean_error_px')} px")
    print(f"  Falsos pos.: {(s.get('false_positive_rate') or 0) * 100:.1f}%")


_PRINT_SUMMARY = {
    "keypoints": _print_summary_keypoints,
    "ball":      _print_summary_ball,
}


def build(kind: str, results_dir: Path, out_dir: Path) -> None:
    cfg = KIND_CONFIG[kind]
    results_dir = results_dir or cfg["results_dir"]

    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in results_dir.glob("*.json") if p.name != ".gitkeep")
    if not paths:
        print(f"No hay reportes en {results_dir}.")
        print(f"Descarga {cfg['report_name']}, renómbralo y ponlo ahí.")
        print(f"  ej: {results_dir}/2026-07-01_abc1234.json")
        return

    loaded = []
    for path in paths:
        try:
            report = json.loads(path.read_text())
            loaded.append((path, report))
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")

    if not loaded:
        print("No se pudo cargar ningún reporte.")
        return

    loaded.sort(key=lambda x: x[1]["metadata"]["timestamp"])

    extract_entry = _EXTRACT_ENTRY[kind]
    history = [extract_entry(report) for _, report in loaded]
    _, latest = loaded[-1]

    report_out = out_dir / cfg["report_name"]
    history_out = out_dir / cfg["history_name"]

    report_out.write_text(json.dumps(latest, indent=2))
    history_out.write_text(json.dumps(history, indent=2))

    s = latest["summary"]
    print(f"✅  {len(loaded)} reporte(s) procesado(s).  (kind={kind})")
    print(f"    {cfg['report_name']}  → {report_out}")
    print(f"    {cfg['history_name']} → {history_out}  ({len(history)} entradas)")
    print()
    print("Último run:")
    _PRINT_SUMMARY[kind](s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["keypoints", "ball"], default="keypoints",
                        help="Tipo de evaluación a consolidar (default: keypoints)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Carpeta con los reportes descargados (default según --kind)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="Carpeta de salida (public/ del dashboard)")
    args = parser.parse_args()
    build(args.kind, args.results_dir, args.out_dir)
