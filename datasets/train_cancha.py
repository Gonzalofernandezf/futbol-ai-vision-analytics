"""
DÍA 2 — Reentrenamiento de modelo_cancha.pt (detección keypoints de cancha)

ANTES DE EJECUTAR:
1. Tener descargados los datasets (download_datasets.py)
2. Comprobar que datasets/soccer-field-keypoint/data.yaml existe

DÓNDE EJECUTAR:
- Con GPU local (NVIDIA): ejecutar directamente aquí
- Sin GPU:               abrir train_cancha.ipynb en Google Colab y ejecutar ahí

TIEMPO ESTIMADO:
- GPU local (RTX 3060+): ~1-2 horas
- Google Colab T4 (gratis): ~2-3 horas
- CPU solo:              NO recomendado (>12h)
"""
from ultralytics import YOLO

# Partimos de tu modelo actual como base (no desde cero)
model = YOLO("modelo_cancha.pt")

results = model.train(
    data="datasets/soccer-field-keypoint/data.yaml",
    epochs=150,
    imgsz=1280,
    batch=8,          # reducir a 4 si hay error de memoria
    device=0,         # 0 = primera GPU, 'cpu' si no tienes GPU
    project="models/runs",
    name="cancha_v2",
    patience=30,      # para si deja de mejorar
    save=True,
    exist_ok=True,
)

print(f"\nModelo guardado en: models/runs/cancha_v2/weights/best.pt")
print("Copia ese archivo a la raíz como modelo_cancha_v2.pt para probarlo")
