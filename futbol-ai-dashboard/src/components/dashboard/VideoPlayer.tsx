import type { MatchMeta } from "@/types/match";
import { PossessionTimeline } from "./PossessionTimeline";
import { useVideo } from "@/contexts/VideoContext";

export function VideoPlayer({ meta }: { meta: MatchMeta }) {
  const { registerVideo } = useVideo();
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
      <PossessionTimeline meta={meta} />
      <div className="relative aspect-video w-full overflow-hidden rounded-md bg-black">
        <video
          ref={registerVideo}
          src="/demo_video.mp4"
          controls
          className="h-full w-full object-contain"
        />
      </div>
    </div>
  );
}