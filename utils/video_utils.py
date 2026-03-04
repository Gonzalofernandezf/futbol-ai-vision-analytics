import cv2

def read_video(video_path, segments=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    if not segments:
        print("🎬 Leyendo el video completo...")
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
    else:
        for i, (start_sec, end_sec) in enumerate(segments):
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            
            print(f"🎬 Extrayendo Tiempo {i+1}: Desde el segundo {start_sec} hasta el {end_sec}...")
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret: 
                    break 
                frames.append(frame)
                current_frame += 1
                
    cap.release()
    print(f"✅ Extracción completada. Total de fotogramas a analizar: {len(frames)}")
    return frames, fps

# (Tu función save_video sigue igual abajo)

def save_video(output_video_frames, output_video_path, fps=25):
    if not output_video_frames:
        print("❌ Error: No frames to save.")
        return
   
    # Force the filename to end with .mp4
    if not output_video_path.endswith('.mp4'):
        output_video_path = output_video_path.replace('.webm', '.mp4')
    
    height, width, _ = output_video_frames[0].shape
    
    # We use 'vp09'. Allows to be presented on the HTML MVP 
    fourcc = cv2.VideoWriter_fourcc(*'vp09') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"💾 Video saved successfully to: {output_video_path}")

   