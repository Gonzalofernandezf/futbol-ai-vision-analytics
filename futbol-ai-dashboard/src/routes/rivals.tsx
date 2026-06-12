import { createFileRoute } from "@tanstack/react-router";
import { useMatchData } from "@/hooks/useMatchData";
import type { Player } from "@/types/match";
import { LoadingSkeleton, ErrorPanel } from "@/components/LoadingSkeleton";
import { Gauge, Route as RouteIcon, Zap, Users, Activity } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";

export const Route = createFileRoute("/rivals")({
  head: () => ({
    meta: [
      { title: "Rivals — Futbol AI" },
      { name: "description", content: "Comparativa equipo vs equipo." },
    ],
  }),
  component: RivalsPage,
});

interface TeamAgg {
  team: 1 | 2;
  count: number;
  total_distance_m: number;
  avg_distance_m: number;
  max_speed_kmh: number;
  avg_max_speed_kmh: number;
  avg_max_accel_ms2: number;
}

function aggregate(players: Player[], team: 1 | 2): TeamAgg {
  const subset = players.filter((p) => p.team === team);
  const n = subset.length || 1;
  const totalDist = subset.reduce((s, p) => s + p.total_distance_m, 0);
  const maxSpeed = subset.reduce((m, p) => Math.max(m, p.max_speed_kmh), 0);
  const avgMaxSpeed = subset.reduce((s, p) => s + p.max_speed_kmh, 0) / n;
  const avgMaxAccel = subset.reduce((s, p) => s + p.max_acceleration_ms2, 0) / n;
  return {
    team,
    count: subset.length,
    total_distance_m: totalDist,
    avg_distance_m: totalDist / n,
    max_speed_kmh: maxSpeed,
    avg_max_speed_kmh: avgMaxSpeed,
    avg_max_accel_ms2: avgMaxAccel,
  };
}

function RivalsPage() {
  const { meta, players, loading, error } = useMatchData();
  if (loading) return <LoadingSkeleton />;
  if (error || !meta) return <ErrorPanel message={error ?? "Datos no disponibles"} />;

  const t1 = aggregate(players, 1);
  const t2 = aggregate(players, 2);

  const possessionRows = [
    { key: "Posesión", t1: meta.home_possession, t2: meta.away_possession, unit: "%" },
  ];

  const numericRows: Array<{
    key: string;
    icon: React.ReactNode;
    t1: number;
    t2: number;
    fmt: (v: number) => string;
  }> = [
    {
      key: "Distancia total",
      icon: <RouteIcon className="h-3.5 w-3.5" />,
      t1: t1.total_distance_m,
      t2: t2.total_distance_m,
      fmt: fmtDist,
    },
    {
      key: "Distancia media / jugador",
      icon: <RouteIcon className="h-3.5 w-3.5" />,
      t1: t1.avg_distance_m,
      t2: t2.avg_distance_m,
      fmt: fmtDist,
    },
    {
      key: "Vel. máx. del equipo",
      icon: <Gauge className="h-3.5 w-3.5" />,
      t1: t1.max_speed_kmh,
      t2: t2.max_speed_kmh,
      fmt: (v) => `${v.toFixed(1)} km/h`,
    },
    {
      key: "Vel. máx. media",
      icon: <Gauge className="h-3.5 w-3.5" />,
      t1: t1.avg_max_speed_kmh,
      t2: t2.avg_max_speed_kmh,
      fmt: (v) => `${v.toFixed(1)} km/h`,
    },
    {
      key: "Acel. máx. media",
      icon: <Zap className="h-3.5 w-3.5" />,
      t1: t1.avg_max_accel_ms2,
      t2: t2.avg_max_accel_ms2,
      fmt: (v) => `${v.toFixed(2)} m/s²`,
    },
    {
      key: "Jugadores detectados",
      icon: <Users className="h-3.5 w-3.5" />,
      t1: t1.count,
      t2: t2.count,
      fmt: (v) => `${v}`,
    },
  ];

  const barData = [
    {
      metric: "Distancia tot. (km)",
      Equipo1: +(t1.total_distance_m / 1000).toFixed(1),
      Equipo2: +(t2.total_distance_m / 1000).toFixed(1),
    },
    {
      metric: "Vel. máx (km/h)",
      Equipo1: +t1.max_speed_kmh.toFixed(1),
      Equipo2: +t2.max_speed_kmh.toFixed(1),
    },
    {
      metric: "Vel. máx media",
      Equipo1: +t1.avg_max_speed_kmh.toFixed(1),
      Equipo2: +t2.avg_max_speed_kmh.toFixed(1),
    },
    {
      metric: "Acel. máx media",
      Equipo1: +t1.avg_max_accel_ms2.toFixed(2),
      Equipo2: +t2.avg_max_accel_ms2.toFixed(2),
    },
  ];

  return (
    <div className="flex w-full flex-col gap-5 p-6">
      <header>
        <h1 className="text-2xl font-bold">Rivals — Equipo vs Equipo</h1>
        <p className="text-sm text-muted-foreground">
          Agregados del partido cargado · {Math.round(meta.duration_seconds / 60)}′
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <TeamCard team={1} agg={t1} possession={meta.home_possession} />
        <TeamCard team={2} agg={t2} possession={meta.away_possession} />
      </div>

      <section className="rounded-lg border border-border bg-card p-4">
        <h2 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          <Activity className="h-3.5 w-3.5" /> Comparativa
        </h2>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis dataKey="metric" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={40} />
              <Tooltip
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="Equipo1" fill="var(--team-1)" />
              <Bar dataKey="Equipo2" fill="var(--team-2)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-lg border border-border bg-card p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Tabla detallada
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-xs text-muted-foreground">
                <th className="py-2 text-left font-medium">Métrica</th>
                <th className="py-2 text-right font-medium" style={{ color: "var(--team-1)" }}>
                  Equipo 1
                </th>
                <th className="py-2 text-right font-medium" style={{ color: "var(--team-2)" }}>
                  Equipo 2
                </th>
                <th className="py-2 text-right font-medium">Δ</th>
              </tr>
            </thead>
            <tbody>
              {possessionRows.map((r) => (
                <tr key={r.key} className="border-b border-border/50">
                  <td className="py-2 text-muted-foreground">{r.key}</td>
                  <td className="py-2 text-right font-mono">{r.t1.toFixed(1)}{r.unit}</td>
                  <td className="py-2 text-right font-mono">{r.t2.toFixed(1)}{r.unit}</td>
                  <td className="py-2 text-right font-mono text-xs text-muted-foreground">
                    {(r.t1 - r.t2 > 0 ? "+" : "")}
                    {(r.t1 - r.t2).toFixed(1)}{r.unit}
                  </td>
                </tr>
              ))}
              {numericRows.map((r) => {
                const delta = r.t1 - r.t2;
                return (
                  <tr key={r.key} className="border-b border-border/50 last:border-0">
                    <td className="py-2 text-muted-foreground">
                      <span className="inline-flex items-center gap-1.5">
                        {r.icon}
                        {r.key}
                      </span>
                    </td>
                    <td className="py-2 text-right font-mono">{r.fmt(r.t1)}</td>
                    <td className="py-2 text-right font-mono">{r.fmt(r.t2)}</td>
                    <td
                      className={`py-2 text-right font-mono text-xs ${
                        delta > 0
                          ? "text-emerald-400"
                          : delta < 0
                            ? "text-red-400"
                            : "text-muted-foreground"
                      }`}
                    >
                      {delta > 0 ? "+" : ""}
                      {r.fmt(delta).replace(/[a-zA-Z²/ ]+$/, "")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function TeamCard({
  team,
  agg,
  possession,
}: {
  team: 1 | 2;
  agg: TeamAgg;
  possession: number;
}) {
  const color = team === 1 ? "var(--team-1)" : "var(--team-2)";
  return (
    <div
      className="relative overflow-hidden rounded-lg border border-border bg-card p-4"
      style={{
        background: `linear-gradient(135deg, color-mix(in oklab, ${color} 14%, transparent) 0%, transparent 70%)`,
      }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold" style={{ color }}>
          Equipo {team}
        </h3>
        <span className="text-3xl font-black tabular-nums" style={{ color }}>
          {possession.toFixed(1)}%
        </span>
      </div>
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">posesión</div>
      <div className="mt-3 grid grid-cols-3 gap-2">
        <Mini label="Distancia" value={fmtDist(agg.total_distance_m)} />
        <Mini label="Vel. máx" value={`${agg.max_speed_kmh.toFixed(1)} km/h`} />
        <Mini label="Jugadores" value={`${agg.count}`} />
      </div>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 p-2">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-sm font-bold">{value}</div>
    </div>
  );
}

function fmtDist(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(0)} m`;
}
