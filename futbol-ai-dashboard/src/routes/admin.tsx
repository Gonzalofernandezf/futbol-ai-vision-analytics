// TODO: add auth before production — this admin panel is currently open.
import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  ArrowLeft,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Cpu,
  DollarSign,
  Gauge,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ProcessingMeta } from "@/types/processing";

export const Route = createFileRoute("/admin")({
  head: () => ({
    meta: [
      { title: "Admin · Procesamiento — Futbol AI" },
      { name: "description", content: "Panel interno de monitoreo de procesamiento." },
      { name: "robots", content: "noindex,nofollow" },
    ],
  }),
  component: AdminPage,
});

function AdminPage() {
  const [meta, setMeta] = useState<ProcessingMeta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/processing_meta.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<ProcessingMeta>;
      })
      .then(setMeta)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Error"));
  }, []);

  if (error) {
    return (
      <div className="p-6 text-sm text-destructive">No se pudo cargar processing_meta.json: {error}</div>
    );
  }
  if (!meta) {
    return <div className="p-6 text-sm text-muted-foreground">Cargando…</div>;
  }

  const warnCount = meta.sanity_checks.filter((c) => c.status === "warn").length;
  const errorCount = meta.sanity_checks.filter((c) => c.status === "error").length;
  const allOk = warnCount === 0 && errorCount === 0;

  return (
    <div className="flex min-h-screen w-full flex-col gap-4 bg-background p-4 font-mono text-[13px]">
      {/* HEADER */}
      <header className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-border bg-card px-4 py-3">
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">
              admin
            </span>
            <h1 className="font-sans text-base font-semibold tracking-tight">
              {meta.match_id}
            </h1>
          </div>
          <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
            <span>processed_at: {new Date(meta.processed_at).toLocaleString()}</span>
            <span>pipeline: {meta.pipeline_version}</span>
          </div>
        </div>
        <Link
          to="/"
          className="flex items-center gap-2 rounded-md border border-border bg-background px-3 py-1.5 font-sans text-xs hover:bg-muted"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Volver al dashboard cliente
        </Link>
      </header>

      {/* Banner */}
      <div
        className={`flex items-center gap-2 rounded-md border px-4 py-2 text-sm ${
          allOk
            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
            : errorCount > 0
              ? "border-red-500/40 bg-red-500/10 text-red-300"
              : "border-amber-500/40 bg-amber-500/10 text-amber-300"
        }`}
      >
        {allOk ? (
          <>
            <CheckCircle2 className="h-4 w-4" />
            Todos los checks dentro de rango
          </>
        ) : (
          <>
            <AlertTriangle className="h-4 w-4" />
            {warnCount + errorCount} sanity check{warnCount + errorCount === 1 ? "" : "s"} fuera de rango — revisar output
          </>
        )}
      </div>

      {/* SECCIÓN 1: Sanity Checks */}
      <section className="flex flex-col gap-2">
        <SectionTitle>Sanity checks</SectionTitle>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
          {meta.sanity_checks.map((c) => (
            <SanityCard key={c.label} check={c} />
          ))}
        </div>
      </section>

      {/* SECCIÓN 2: Timing breakdown */}
      <section className="grid grid-cols-1 gap-2 lg:grid-cols-[1fr_280px]">
        <div className="flex flex-col gap-3 rounded-md border border-border bg-card p-4">
          <div className="flex items-baseline justify-between">
            <SectionTitle>Timing breakdown</SectionTitle>
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                total
              </div>
              <div className="font-sans text-3xl font-bold tabular-nums text-foreground">
                {formatMMSS(meta.timing.total_seconds)}
              </div>
            </div>
          </div>
          <PhasesChart phases={meta.timing.phases} total={meta.timing.total_seconds} />
        </div>
        <div className="flex flex-col gap-2 rounded-md border border-border bg-card p-4">
          <SectionTitle>Chunks</SectionTitle>
          <ChunkStat label="Total" value={`${meta.timing.chunks_processed}`} />
          <ChunkStat label="Avg" value={`${meta.timing.per_chunk_avg_seconds}s`} />
          <ChunkStat label="Slowest" value={`${meta.timing.slowest_chunk_seconds}s`} />
          <ChunkStat label="Fastest" value={`${meta.timing.fastest_chunk_seconds}s`} />
        </div>
      </section>

      {/* SECCIÓN 3: Models */}
      <section className="flex flex-col gap-2 rounded-md border border-border bg-card p-4">
        <SectionTitle>Models in use</SectionTitle>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-32">Model type</TableHead>
              <TableHead>Filename</TableHead>
              <TableHead className="w-32">mAP</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow>
              <TableCell>Players</TableCell>
              <TableCell className="font-mono">{meta.models.players}</TableCell>
              <TableCell className="text-muted-foreground">—</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Ball</TableCell>
              <TableCell className="font-mono">{meta.models.ball}</TableCell>
              <TableCell className="text-muted-foreground">—</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Field</TableCell>
              <TableCell className="font-mono">{meta.models.field}</TableCell>
              <TableCell className="text-muted-foreground">—</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </section>

      {/* SECCIÓN 4: Infra */}
      <section className="flex flex-col gap-2 rounded-md border border-border bg-card p-4">
        <SectionTitle>Infrastructure</SectionTitle>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
          <InfraStat
            icon={<Cpu className="h-4 w-4" />}
            label="GPU model"
            value={meta.infra.gpu_model}
          />
          <InfraStat
            icon={<Gauge className="h-4 w-4" />}
            label="GPU util avg"
            value={`${meta.infra.gpu_util_avg_pct}%`}
          />
          <InfraStat
            icon={<DollarSign className="h-4 w-4" />}
            label="Estimated cost"
            value={`$${meta.infra.estimated_cost_usd.toFixed(2)}`}
          />
        </div>
      </section>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="font-sans text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
      {children}
    </h2>
  );
}

function SanityCard({
  check,
}: {
  check: ProcessingMeta["sanity_checks"][number];
}) {
  const colors = {
    ok: { border: "#10b981", icon: <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> },
    warn: { border: "#f59e0b", icon: <AlertTriangle className="h-3.5 w-3.5 text-amber-400" /> },
    error: { border: "#ef4444", icon: <XCircle className="h-3.5 w-3.5 text-red-400" /> },
  } as const;
  const c = colors[check.status];
  return (
    <div
      className="flex flex-col gap-1 rounded-md border border-border bg-card p-3"
      style={{ borderLeft: `3px solid ${c.border}` }}
    >
      <div className="flex items-center justify-between gap-1">
        <span className="truncate text-[10px] uppercase tracking-wider text-muted-foreground">
          {check.label}
        </span>
        {c.icon}
      </div>
      <span className="font-sans text-2xl font-bold tabular-nums text-foreground">
        {check.value}
      </span>
      <span className="text-[10px] text-muted-foreground">expected: {check.expected}</span>
    </div>
  );
}

function PhasesChart({
  phases,
  total,
}: {
  phases: Record<string, number>;
  total: number;
}) {
  const data = Object.entries(phases)
    .map(([name, seconds]) => ({
      name,
      seconds,
      pct: (seconds / total) * 100,
    }))
    .sort((a, b) => b.seconds - a.seconds);

  return (
    <div style={{ width: "100%", height: Math.max(220, data.length * 26) }}>
      <ResponsiveContainer>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 80, left: 10, bottom: 4 }}
        >
          <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 10, fill: "rgba(255,255,255,0.5)" }}
            tickLine={false}
            axisLine={false}
            unit="s"
          />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            tick={{ fontSize: 11, fill: "rgba(255,255,255,0.75)", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              background: "var(--popover)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v: number, _n, item) => [
              `${v}s (${(item.payload as { pct: number }).pct.toFixed(1)}%)`,
              "duración",
            ]}
          />
          <Bar
            dataKey="seconds"
            radius={[2, 2, 2, 2]}
            label={{
              position: "right",
              fontSize: 10,
              fill: "rgba(255,255,255,0.7)",
              formatter: (label: unknown) => {
                const v = typeof label === "number" ? label : 0;
                return `${v}s (${((v / total) * 100).toFixed(1)}%)`;
              },
            }}
          >
            {data.map((d) => (
              <Cell
                key={d.name}
                fill={
                  d.pct > 50
                    ? "var(--primary)"
                    : d.pct > 10
                      ? "color-mix(in oklab, var(--primary) 70%, transparent)"
                      : "color-mix(in oklab, var(--primary) 40%, transparent)"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ChunkStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-border/50 py-1.5 last:border-0">
      <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className="font-sans text-lg font-bold tabular-nums text-foreground">{value}</span>
    </div>
  );
}

function InfraStat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-border bg-background/40 p-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary">
        {icon}
      </div>
      <div className="flex flex-col">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</span>
        <span className="font-sans text-lg font-bold tabular-nums text-foreground">{value}</span>
      </div>
    </div>
  );
}

function formatMMSS(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}