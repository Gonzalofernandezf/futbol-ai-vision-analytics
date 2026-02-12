import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Read the original FPS from the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Now we return TWO things: the list of frames AND the fps
    return frames, fps

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

   