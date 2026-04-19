"""
Ejecutar en tu máquina local:
  pip install roboflow
  python datasets/download_datasets.py
"""
from roboflow import Roboflow

API_KEY = "TU_API_KEY_AQUI"  # Pega tu nueva key aquí tras regenerarla en roboflow.com

rf = Roboflow(api_key=API_KEY)

# --- CANCHA (para reentrenar modelo_cancha.pt) ---

print("1/4 Descargando soccer-field-keypoint...")
project = rf.workspace("roboflow-100").project("soccer-field-keypoint")
project.version(1).download("yolov8", location="datasets/soccer-field-keypoint")
print("  ✓ OK")

print("2/4 Descargando football-field-detection (SoccerNet)...")
project = rf.workspace("soccernet").project("football-field-detection")
project.version(1).download("yolov8", location="datasets/football-field-detection")
print("  ✓ OK")

# --- JUGADORES (para reentrenar best_100e.pt) ---

print("3/4 Descargando football-players-detection...")
project = rf.workspace("roboflow-jvnbs").project("football-players-detection-3zvbc")
project.version(4).download("yolov8", location="datasets/football-players-detection")
print("  ✓ OK")

print("4/4 Descargando soccer-players-detection...")
project = rf.workspace("david-lee-d0rhs").project("soccer-players-detection")
project.version(1).download("yolov8", location="datasets/soccer-players-detection")
print("  ✓ OK")

print("\nTodos los datasets listos en datasets/")
