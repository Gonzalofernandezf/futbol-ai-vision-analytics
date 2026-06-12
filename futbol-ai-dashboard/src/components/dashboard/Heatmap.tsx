import { useMemo } from "react";

interface Props {
  positions: Array<[number, number] | null>;
  width?: number;
  height?: number;
  /** Pitch dimensions in meters */
  pitchWidth?: number;
  pitchHeight?: number;
}

const GRID_X = 24;
const GRID_Y = 16;

function heatColor(t: number): string {
  // t in [0,1] -> blue -> yellow -> red
  if (t <= 0) return "rgba(0,0,0,0)";
  if (t < 0.5) {
    const k = t / 0.5;
    const r = Math.round(40 + k * (240 - 40));
    const g = Math.round(120 + k * (220 - 120));
    const b = Math.round(220 - k * 220);
    return `rgba(${r},${g},${b},${0.55 * t + 0.2})`;
  }
  const k = (t - 0.5) / 0.5;
  const r = 240;
  const g = Math.round(220 - k * 180);
  const b = Math.round(40 - k * 40);
  return `rgba(${r},${g},${b},${0.55 * t + 0.3})`;
}

export function Heatmap({
  positions,
  width = 720,
  height = 460,
  pitchWidth = 100,
  pitchHeight = 64,
}: Props) {
  const cells = useMemo(() => {
    const grid: number[][] = Array.from({ length: GRID_Y }, () =>
      Array.from({ length: GRID_X }, () => 0),
    );
    let max = 0;
    for (const p of positions) {
      if (!p) continue;
      const [x, y] = p;
      const gx = Math.min(GRID_X - 1, Math.max(0, Math.floor((x / pitchWidth) * GRID_X)));
      const gy = Math.min(GRID_Y - 1, Math.max(0, Math.floor((y / pitchHeight) * GRID_Y)));
      grid[gy][gx] += 1;
      if (grid[gy][gx] > max) max = grid[gy][gx];
    }
    return { grid, max };
  }, [positions, pitchWidth, pitchHeight]);

  const cellW = width / GRID_X;
  const cellH = height / GRID_Y;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="h-auto w-full rounded-lg border border-border bg-[#0a4d2e]"
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <filter id="heatBlur" x="-10%" y="-10%" width="120%" height="120%">
          <feGaussianBlur stdDeviation="14" />
        </filter>
      </defs>

      {/* Pitch markings */}
      <g stroke="rgba(255,255,255,0.5)" strokeWidth="2" fill="none">
        <rect x="4" y="4" width={width - 8} height={height - 8} />
        <line x1={width / 2} y1="4" x2={width / 2} y2={height - 4} />
        <circle cx={width / 2} cy={height / 2} r={height * 0.12} />
        <circle cx={width / 2} cy={height / 2} r="3" fill="rgba(255,255,255,0.6)" />
        {/* Penalty areas */}
        <rect x="4" y={height * 0.2} width={width * 0.12} height={height * 0.6} />
        <rect
          x={width - 4 - width * 0.12}
          y={height * 0.2}
          width={width * 0.12}
          height={height * 0.6}
        />
      </g>

      {/* Heat cells */}
      <g filter="url(#heatBlur)">
        {cells.grid.map((row, gy) =>
          row.map((v, gx) => {
            if (v === 0 || cells.max === 0) return null;
            const t = v / cells.max;
            return (
              <rect
                key={`${gx}-${gy}`}
                x={gx * cellW}
                y={gy * cellH}
                width={cellW}
                height={cellH}
                fill={heatColor(t)}
              />
            );
          }),
        )}
      </g>
    </svg>
  );
}