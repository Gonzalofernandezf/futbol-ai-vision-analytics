from ultralytics import YOLO
import os

# 1. Cargar modelo
print("🧠 Cargando modelo...")
model = YOLO('best_100e.pt') 

# 2. RUTAS ACTUALIZADAS (Rutas relativas)
# Como el video y el código están en la misma carpeta, solo ponemos el nombre.
video_path = r'C:\Users\gafer\Desktop\24. Scouting_Project_Local\01_Inputs\football video analysis_1.mp4'  # <--- Asegúrate que tu video se llama así
ruta_salida = r'C:\Users\gafer\Desktop\24. Scouting_Project_Local\03_Resultados_Futbol\01_Local_Project'

# Verificación de seguridad para que no te vuelvas loco si no lo encuentra
if not os.path.exists(video_path):
    print(f"❌ ERROR: No encuentro el video: {video_path}")
    print("Asegúrate de que el archivo .mp4 está en la misma carpeta que este script.")
    exit()

print(f"🚀 Iniciando procesamiento en CPU (Ryzen 7)...")

# 3. Ejecución
results = model.track(
    source=video_path,
    save=True,
    project=ruta_salida,
    name='analisis_v3_agresivo',
    exist_ok=True,
    
    # CAMBIOS:
    conf=0.10,       # Bajamos la confianza al 10% (captará al arquero, pero quizás también algo de basura)
    iou=0.5,         # Intersection Over Union: ayuda a manejar oclusiones
    imgsz=1280,      
    persist=True,
    
    line_width=1,
    show_conf=False,
    show_labels=True,
    device='cpu',
    
    # IMPORTANTE: Esto le dice a YOLO que intente rastrear aunque pierda al jugador unos frames
    tracker="bytetrack.yaml" 
)

print(f"✅ ¡Terminado!")