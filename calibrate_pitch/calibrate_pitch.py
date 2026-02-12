import cv2
import numpy as np

# ⚠️ MAKE SURE THE VIDEO NAME IS CORRECT
VIDEO_PATH = "football video analysis_1.mp4" 
SEGUNDO_EXACTO = 23   # Exact second where we can see the right side of the pitch better
ANCHO_PANTALLA = 1080 # Force window width (900 if still too large)

points_real = [] # Here we save the real coordinates (original video)
points_visual = [] # Here we save the visual coordinates (from clicks)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Calculate REAL coordinate (undoing the zoom)
        scale_factor = param
        real_x = int(x / scale_factor)
        real_y = int(y / scale_factor)
        
        points_real.append([real_x, real_y])
        points_visual.append([x, y])
        
        print(f"✅ Click {len(points_real)}: Visual[{x},{y}] -> Real[{real_x},{real_y}]")
        
        # Draw on the reduced image (visual)
        cv2.circle(frame_resized, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame_resized, str(len(points_real)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow(f"CALIBRADOR - Frame {SEGUNDO_EXACTO}s", frame_resized)

        if len(points_real) == 4:
            print("\n" + "="*50)
            print("🚀 POINTS READY! COPY THIS TO view_transformer.py:")
            print("="*50)
            # Print the array of REAL coordinates
            print(f"self.pixel_vertices = np.array({points_real}).astype(np.float32)")
            print("\nPress 'q' to exit.")

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC, SEGUNDO_EXACTO * 1000)
ret, frame = cap.read()
cap.release()

if ret:
    # Resizing logic
    alto_original, ancho_original = frame.shape[:2]
    scale = ANCHO_PANTALLA / ancho_original
    
    # Resize only for viewing
    frame_resized = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    
    print(f"\n--- INSTRUCTIONS (SCALE {scale:.2f}x) ---")
    print("Click on the 4 corners of the PENALTY BOX (Goalkeeper area):")
    print("1. TOP CORNER (Looking at midfield)")
    print("2. TOP END (Goal line)")
    print("3. BOTTOM END (Goal line)")
    print("4. BOTTOM CORNER (Looking at midfield)")
    
    cv2.imshow(f"CALIBRADOR - Frame {SEGUNDO_EXACTO}s", frame_resized)
    # Pass the scale factor as a parameter to compute the inverse when clicking
    cv2.setMouseCallback(f"CALIBRADOR - Frame {SEGUNDO_EXACTO}s", mouse_callback, scale)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Error: Could not read the video.")