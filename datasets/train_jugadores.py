"""
DÍA 3 — Reentrenamiento de best_100e.pt (detección de jugadores)

ANTES DE EJECUTAR:
1. Tener descargados los datasets (download_datasets.py)
2. Comprobar que datasets/football-players-detection/data.yaml existe

DÓNDE EJECUTAR:
- Con GPU local (NVIDIA): ejecutar directamente aquí
- Sin GPU:               abrir train_jugadores.ipynb en Google Colab y ejecutar ahí

TIEMPO ESTIMADO:
- GPU local (RTX 3060+): ~30-45 min (solo 50 épocas, parte de base ya entrenada)
- Google Colab T4 (gratis): ~1 hora
- CPU solo:              NO recomendado
"""
from ultralytics import YOLO

# Partimos de tu modelo actual como base
model = YOLO("best_100e.pt")

results = model.train(
    data="datasets/football-players-detection/data.yaml",
    epochs=50,
    imgsz=1280,
    batch=8,          # reducir a 4 si hay error de memoria
    freeze=10,        # congela el backbone, solo reentrena la cabeza de detección
    device=0,         # 0 = primera GPU, 'cpu' si no tienes GPU
    project="models/runs",
    name="jugadores_v2",
    patience=20,
    save=True,
    exist_ok=True,
)

print(f"\nModelo guardado en: models/runs/jugadores_v2/weights/best.pt")
print("Copia ese archivo a la raíz como best_v2.pt para probarlo")
