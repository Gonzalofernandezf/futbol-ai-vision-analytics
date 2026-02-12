import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # --- NUEVO: Leemos los FPS originales del video ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Ahora devolvemos DOS cosas: la lista de fotos Y la velocidad (fps)
    return frames, fps

def save_video(output_video_frames, output_video_path, fps=25):
    if not output_video_frames:
        print("❌ Error: No hay frames para guardar.")
        return
   
    # Forzamos que el nombre termine en .mp4
    if not output_video_path.endswith('.mp4'):
        output_video_path = output_video_path.replace('.webm', '.mp4')
    
    height, width, _ = output_video_frames[0].shape
    
    # Usamos 'mp4v'. Es el códec más universal y menos problemático que existe.
    fourcc = cv2.VideoWriter_fourcc(*'vp09') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"💾 Video guardado limpiamente en: {output_video_path}")

   