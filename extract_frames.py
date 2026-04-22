"""
Extrae N frames distribuidos uniformemente a lo largo del vídeo y los guarda como
imágenes JPEG en una carpeta de salida.

Uso:
    python extract_frames.py
    python extract_frames.py --video mi_video.mp4 --n 200 --output frames_tagging
"""

import argparse
import os
import cv2
import numpy as np


def extract_frames(video_path: str, n_frames: int, output_dir: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        raise ValueError("El vídeo no tiene frames o no se pudo leer su duración.")

    n_frames = min(n_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Vídeo:          {video_path}")
    print(f"Total frames:   {total_frames}  ({total_frames / fps:.1f}s a {fps:.2f} fps)")
    print(f"Frames a extraer: {n_frames}")
    print(f"Carpeta salida: {output_dir}\n")

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
    args = parser.parse_args()

    extract_frames(args.video, args.n, args.output)
