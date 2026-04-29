from ultralytics import YOLO
import cv2

model = YOLO("modelo_cancha.pt")
frame = cv2.imread(r"C:\Users\gafer\Desktop\Git\futbol-ai-vision-analytics\eval\frames\frame_002684_t01m29s_jpg.rf.KtggtKBlhLGyn2jQQWRu.jpg")

results = model(frame, verbose=False)[0]
xy   = results.keypoints.xy[0].cpu().numpy()
conf = results.keypoints.conf[0].cpu().numpy()

for i, (x, y) in enumerate(xy):
    x, y = int(x), int(y)
    c = float(conf[i])
    color = (0, 255, 0) if c >= 0.1 else (0, 0, 255)
    cv2.circle(frame, (x, y), 8, color, -1)
    cv2.putText(frame, f"{i}({c:.2f})", (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

out_path = r"C:\Users\gafer\Desktop\Git\futbol-ai-vision-analytics\eval\keypoints_visualization.jpg"
cv2.imwrite(out_path, frame)
print(f"Guardado en: {out_path}")
