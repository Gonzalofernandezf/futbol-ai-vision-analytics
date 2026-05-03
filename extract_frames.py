"""
Extrae N frames distribuidos uniformemente a lo largo del vídeo y los guarda como
imágenes JPEG en una carpeta de salida.

Uso:
    python extract_frames.py
    python extract_frames.py --video mi_video.mp4 --n 200 --output frames_tagging

    # Partido completo con descanso y pre/post partido:
    python extract_frames.py --start 31 --end 6793 --skip-start 2791 --skip-end 3726
"""

import argparse
import os
import cv2
import numpy as np


def extract_frames(
    video_path: str,
    n_frames: int,
    output_dir: str,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    skip_start_sec: float = 0.0,
    skip_end_sec: float = 0.0,
    offset_frac: float = 0.0,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        raise ValueError("El vídeo no tiene frames o no se pudo leer su duración.")

    start_f      = int(start_sec      * fps) if start_sec > 0 else 0
    end_f        = int(end_sec        * fps) if end_sec   > 0 else total_frames
    skip_start_f = int(skip_start_sec * fps)
    skip_end_f   = int(skip_end_sec   * fps)

    # Pool de índices válidos: dentro del rango del partido, fuera del descanso
    all_indices = np.arange(start_f, end_f)
    if skip_start_f < skip_end_f:
        valid_indices = all_indices[
            (all_indices < skip_start_f) | (all_indices >= skip_end_f)
        ]
    else:
        valid_indices = all_indices

    n_frames = min(n_frames, len(valid_indices))
    step = len(valid_indices) / n_frames
    sample_positions = (np.arange(n_frames) * step + step * offset_frac).astype(int)
    sample_positions = np.clip(sample_positions, 0, len(valid_indices) - 1)
    indices = valid_indices[sample_positions]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Vídeo:            {video_path}")
    print(f"Total frames:     {total_frames}  ({total_frames / fps:.1f}s a {fps:.2f} fps)")
    print(f"Rango partido:    {start_sec:.0f}s – {end_sec if end_sec else total_frames/fps:.0f}s")
    if skip_start_f < skip_end_f:
        print(f"Descanso omitido: {skip_start_sec:.0f}s – {skip_end_sec:.0f}s  "
              f"({skip_end_f - skip_start_f} frames excluidos)")
    print(f"Frames útiles:    {len(valid_indices)}")
    print(f"Frames a extraer: {n_frames}")
    print(f"Carpeta salida:   {output_dir}\n")

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            print(f"  [WARN] No se pudo leer el frame {idx}, saltando.")
            continue

        seconds = idx / fps
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        filename = f"frame_{idx:06d}_t{minutes:02d}m{secs:02d}s.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    print(f"Extraccion completada: {saved}/{n_frames} frames guardados en '{output_dir}'")


if __name__ == "__main__":
    from config import VIDEO_PATH

    parser = argparse.ArgumentParser(description="Extrae frames del vídeo para tagging")
    parser.add_argument("--video", default=VIDEO_PATH, help="Ruta al vídeo (default: config.VIDEO_PATH)")
    parser.add_argument("--n", type=int, default=200, help="Número de frames a extraer (default: 200)")
    parser.add_argument("--output", default="frames_tagging", help="Carpeta de salida (default: frames_tagging/)")
    parser.add_argument("--start", type=float, default=0.0, metavar="SEG",
                        help="Segundo de inicio del partido (default: 0)")
    parser.add_argument("--end", type=float, default=0.0, metavar="SEG",
                        help="Segundo de fin del partido (default: fin del vídeo)")
    parser.add_argument("--skip-start", type=float, default=0.0, metavar="SEG",
                        help="Segundo de inicio del descanso a omitir")
    parser.add_argument("--skip-end", type=float, default=0.0, metavar="SEG",
                        help="Segundo de fin del descanso a omitir")
    parser.add_argument("--offset-frac", type=float, default=0.0, metavar="FRAC",
                        help="Desplazamiento dentro de cada bucket (0.0-1.0). "
                             "Útil para extraer frames distintos a una extracción previa "
                             "(ej. --offset-frac 0.5 para no solapar con la primera).")
    args = parser.parse_args()

    extract_frames(args.video, args.n, args.output, args.start, args.end,
                   args.skip_start, args.skip_end, args.offset_frac)
