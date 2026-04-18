"""
Ejecutar en tu máquina local:
  pip install roboflow
  python datasets/download_datasets.py
"""
from roboflow import Roboflow

API_KEY = "TU_API_KEY_AQUI"  # Pega tu nueva key aquí tras regenerarla

rf = Roboflow(api_key=API_KEY)

# 1. Keypoints de cancha — para reentrenar modelo_cancha.pt
print("Descargando soccer-field-keypoint...")
project = rf.workspace("roboflow-100").project("soccer-field-keypoint")
project.version(1).download("yolov8", location="datasets/soccer-field-keypoint")
print("✓ Cancha OK")

# 2. Jugadores — para reentrenar best_100e.pt
print("Descargando football-players-detection...")
project2 = rf.workspace("roboflow-jvnbs").project("football-players-detection-3zvbc")
project2.version(4).download("yolov8", location="datasets/football-players-detection")
print("✓ Jugadores OK")

print("\nDatasets listos en datasets/")
