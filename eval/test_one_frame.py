from ultralytics import YOLO
import cv2

model = YOLO("modelo_cancha.pt")
frame = cv2.imread(r"C:\Users\gafer\Desktop\Git\futbol-ai-vision-analytics\eval\frames\frame_002684_t01m29s_jpg.rf.KtggtKBlhLGyn2jQQWRu.jpg")

if frame is None:
    print("ERROR: no se pudo leer el frame. Verifica la ruta.")
else:
    print(f"Frame leido: {frame.shape}")
    results = model(frame, verbose=True)[0]
    print("Keypoints:", results.keypoints)
    print("Boxes:", results.boxes)
