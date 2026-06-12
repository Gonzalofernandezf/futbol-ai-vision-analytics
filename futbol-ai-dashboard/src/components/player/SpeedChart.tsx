import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { useVideo } from "@/contexts/VideoContext";

interface Props {
  speedOverTime: Array<number | null>;
  /** Range in minutes to render */
  rangeMin: [number, number];
}

export function SpeedChart({ speedOverTime, rangeMin }: Props) {
  const { seekToMinute } = useVideo();
  const startSec = rangeMin[0] * 60;
  const endSec = Math.min(speedOverTime.length, rangeMin[1] * 60);
  // Downsample to ~120 points and smooth with a moving average so the
  // chart looks like a continuous line rather than a noisy bar-like signal.
  // `t` is kept as exact minutes (float) so clicks map to the precise second.
  const raw = speedOverTime.slice(startSec, endSec);
  const targetPoints = 120;
  const bucketSize = Math.max(1, Math.floor(raw.length / targetPoints));
  const buckets: Array<{ t: number; v: number | null }> = [];
  for (let i = 0; i < raw.length; i += bucketSize) {
    let sum = 0;
    let count = 0;
    for (let j = i; j < Math.min(i + bucketSize, raw.length); j += 1) {
      const x = raw[j];
      if (x != null) {
        sum += x;
        count += 1;
      }
    }
    const centerSec = startSec + i + Math.min(bucketSize, raw.length - i) / 2;
    buckets.push({
      t: centerSec / 60,
      v: count > 0 ? sum / count : null,
    });
  }
  const slice = buckets;

  const fmtMinSec = (mins: number) => {
    const totalSec = Math.max(0, Math.round(mins * 60));
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="h-32 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={slice}
          margin={{ top: 4, right: 8, left: -20, bottom: 0 }}
          style={{ cursor: "pointer" }}
          onClick={(e) => {
            const label = e?.activeLabel;
            const minute = typeof label === "number" ? label : Number(label);
            if (Number.isFinite(minute)) seekToMinute(minute);
          }}
        >
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={[rangeMin[0], rangeMin[1]]}
            tick={{ fontSize: 10, fill: "rgba(255,255,255,0.5)" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `${Math.round(v)}′`}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "rgba(255,255,255,0.5)" }}
            tickLine={false}
            axisLine={false}
            width={36}
          />
          <Tooltip
            contentStyle={{
              background: "var(--popover)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
            }}
            labelFormatter={(l) => fmtMinSec(Number(l))}
            formatter={(v) => {
              const num = typeof v === "number" ? v : null;
              return [num == null ? "—" : `${num.toFixed(1)} km/h`, "Vel"];
            }}
          />
          <Line
            type="monotone"
            dataKey="v"
            stroke="var(--primary)"
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}