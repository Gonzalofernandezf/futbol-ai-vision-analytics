import cv2
import numpy as np

# ⚠️ ASEGÚRATE DE QUE EL NOMBRE DEL VIDEO ES CORRECTO
VIDEO_PATH = "football video analysis_1.mp4" 
SEGUNDO_EXACTO = 23   # El segundo que dijiste que se ve bien
ANCHO_PANTALLA = 1080 # Forzamos que la ventana no pase de este ancho (puedes bajarlo a 900 si sigue grande)

points_real = [] # Aquí guardaremos las coordenadas REALES (video original)
points_visual = [] # Aquí las del clic (para dibujar)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Calcular coordenada REAL (deshaciendo el zoom)
        scale_factor = param
        real_x = int(x / scale_factor)
        real_y = int(y / scale_factor)
        
        points_real.append([real_x, real_y])
        points_visual.append([x, y])
        
        print(f"✅ Click {len(points_real)}: Visual[{x},{y}] -> Real[{real_x},{real_y}]")
        
        # Dibujar en la imagen reducida (visual)
        cv2.circle(frame_resized, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame_resized, str(len(points_real)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow(f"CALIBRADOR - Frame {SEGUNDO_EXACTO}s", frame_resized)

        if len(points_real) == 4:
            print("\n" + "="*50)
            print("🚀 ¡PUNTOS LISTOS! COPIA ESTO EN view_transformer.py:")
            print("="*50)
            # Imprimimos el array de coordenadas REALES
            print(f"self.pixel_vertices = np.array({points_real}).astype(np.float32)")
            print("\nPulsa 'q' para salir.")

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, SEGUNDO_EXACTO * 1000)
ret, frame = cap.read()
cap.release()

if ret:
    # --- LOGICA DE REDIMENSIONADO ---
    alto_original, ancho_original = frame.shape[:2]
    scale = ANCHO_PANTALLA / ancho_original
    
    # Redimensionamos solo para ver
    frame_resized = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    
    print(f"\n--- INSTRUCCIONES (ESCALA {scale:.2f}x) ---")
    print("Haz clic en las 4 esquinas del ÁREA GRANDE (Caja del Portero):")
    print("1. PICO ARRIBA (Mira al medio campo)")
    print("2. FONDO ARRIBA (Línea de gol)")
    print("3. FONDO ABAJO  (Línea de gol)")
    print("4. PICO ABAJO  (Mira al medio campo)")
    
    cv2.imshow(f"CALIBRADOR - Frame {SEGUNDO_EXACTO}s", frame_resized)
    # Pasamos el factor de escala como parámetro para poder calcular la inversa al hacer clic
    cv2.setMouseCallback(f"CALIBRADOR - Frame {SEGUNDO_EXACTO}s", mouse_callback, scale)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Error: No se pudo leer el video.")