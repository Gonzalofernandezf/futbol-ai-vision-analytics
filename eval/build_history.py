#!/usr/bin/env python3
"""
Reconstruye eval_history.json a partir de todos los reportes individuales
guardados en eval/results/ y actualiza los artefactos del dashboard:

  - futbol-ai-dashboard/public/eval_report.json   (el más reciente)
  - futbol-ai-dashboard/public/eval_history.json  (todos los runs ordenados)

Flujo de uso:
  1. Descarga eval_report.json de Kaggle.
  2. Renómbralo con fecha/hash y cópialo a eval/results/
     (ej: eval/results/2026-07-01_abc1234.json)
  3. Ejecuta:  python eval/build_history.py
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS = REPO_ROOT / "eval" / "results"
DEFAULT_OUT = REPO_ROOT / "futbol-ai-dashboard" / "public"


def _extract_entry(report: dict) -> dict:
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


def build(results_dir: Path, out_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in results_dir.glob("*.json") if p.name != ".gitkeep")
    if not paths:
        print(f"No hay reportes en {results_dir}.")
        print("Descarga eval_report.json de Kaggle, renómbralo y ponlo ahí.")
        print("  ej: eval/results/2026-07-01_abc1234.json")
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

    history = [_extract_entry(report) for _, report in loaded]
    _, latest = loaded[-1]

    report_out = out_dir / "eval_report.json"
    history_out = out_dir / "eval_history.json"

    report_out.write_text(json.dumps(latest, indent=2))
    history_out.write_text(json.dumps(history, indent=2))

    s = latest["summary"]
    print(f"✅  {len(loaded)} reporte(s) procesado(s).")
    print(f"    eval_report.json  → {report_out}")
    print(f"    eval_history.json → {history_out}  ({len(history)} entradas)")
    print()
    print("Último run:")
    print(f"  Recall  : {(s.get('overall_recall') or 0) * 100:.1f}%")
    print(f"  PCK@10  : {(s.get('overall_pck_10px') or 0) * 100:.1f}%")
    print(f"  Error   : {s.get('mean_error_px')} px")
    print(f"  Hom.err : {s.get('mean_homography_error_m')} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Carpeta con los eval_report.json descargados")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="Carpeta de salida (public/ del dashboard)")
    args = parser.parse_args()
    build(args.results_dir, args.out_dir)
