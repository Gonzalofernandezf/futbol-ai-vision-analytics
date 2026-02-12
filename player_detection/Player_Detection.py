from ultralytics import YOLO
import os

# 1. Load model
print("🧠 Loading model...")
model = YOLO('best_100e.pt') 

# 2. UPDATED PATHS (Relative paths)
# Since the video and code are in the same folder, we just use the name.
video_path = r'C:\Users\gafer\Desktop\24. Scouting_Project_Local\01_Inputs\football video analysis_1.mp4'  # <--- Make sure your video is called like that
ruta_salida = r'C:\Users\gafer\Desktop\24. Scouting_Project_Local\03_Resultados_Futbol\01_Local_Project'

# Safety check so you don't go crazy if it's not found
if not os.path.exists(video_path):
    print(f"❌ ERROR: Cannot find video: {video_path}")
    print("Make sure the .mp4 file is in the same folder as this script.")
    exit()

print(f"🚀 Starting processing on CPU...")

# 3. Execution
results = model.track(
    source=video_path,
    save=True,
    project=ruta_salida,
    name='analisis_v3_agresivo',
    exist_ok=True,
    
    conf=0.10,       # We lower confidence to 10% (it will catch the goalkeeper, but perhaps also some junk).
    iou=0.5,         # Intersection Over Union: ayuda a manejar oclusiones
    imgsz=1280,      
    persist=True,
    
    line_width=1,
    show_conf=False,
    show_labels=True,
    device='cpu',
    
    # IMPORTANT: This tells YOLO to try to track even if it loses the player for a few frames
    tracker="bytetrack.yaml" 
)

print(f"✅ Done!")