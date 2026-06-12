import { createContext, useCallback, useContext, useMemo, useRef, type ReactNode } from "react";

interface VideoState {
  registerVideo: (el: HTMLVideoElement | null) => void;
  seekToMinute: (minute: number) => void;
}

const VideoContext = createContext<VideoState | null>(null);

export function VideoProvider({ children }: { children: ReactNode }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const registerVideo = useCallback((el: HTMLVideoElement | null) => {
    videoRef.current = el;
  }, []);

  const seekToMinute = useCallback((minute: number) => {
    const v = videoRef.current;
    if (!v) return;
    const target = minute * 60;
    const max = Number.isFinite(v.duration) ? v.duration : target;
    v.currentTime = Math.max(0, Math.min(target, max));
    v.pause();
  }, []);

  const value = useMemo<VideoState>(
    () => ({ registerVideo, seekToMinute }),
    [registerVideo, seekToMinute],
  );

  return <VideoContext.Provider value={value}>{children}</VideoContext.Provider>;
}

export function useVideo(): VideoState {
  const ctx = useContext(VideoContext);
  if (!ctx) throw new Error("useVideo must be used within VideoProvider");
  return ctx;
}
