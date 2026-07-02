import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import { AlertTriangle, CheckCircle2, Target, Crosshair } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { EvalHistoryEntry, EvalReport } from "@/types/eval";

export const Route = createFileRoute("/metrics/")({
  component: FieldMetricsPage,
});

function FieldMetricsPage() {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [history, setHistory] = useState<EvalHistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/eval_report.json").then((r) =>
        r.ok ? (r.json() as Promise<EvalReport>) : Promise.reject(new Error(`report ${r.status}`)),
      ),
      fetch("/eval_history.json").then((r) =>
        r.ok
          ? (r.json() as Promise<EvalHistoryEntry[]>)
          : Promise.resolve([] as EvalHistoryEntry[]),
      ),
    ])
      .then(([rep, hist]) => {
        setReport(rep);
        setHistory(hist);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Error"));
  }, []);

  if (error) {
    return <div className="text-sm text-destructive">No se pudo cargar la evaluación: {error}</div>;
  }
  if (!report) {
    return <div className="text-sm text-muted-foreground">Cargando evaluación…</div>;
  }

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          {report.metadata.model} · {report.metadata.n_frames} frames · conf{" "}
          {report.metadata.conf_threshold} · {fmtDate(report.metadata.timestamp)}
        </p>
        <code className="rounded bg-muted px-2 py-1 text-xs text-muted-foreground">
          {report.metadata.git_commit.short_hash} · {report.metadata.git_commit.message}
        </code>
      </header>

      <SummaryCards report={report} />

      <Section title="Histórico de evaluaciones">
        {history && history.length > 1 ? (
          <HistoryChart history={history} />
        ) : (
          <p className="text-xs text-muted-foreground">
            Solo hay {history?.length ?? 0} run{history?.length === 1 ? "" : "s"} registrada
            {history?.length === 1 ? "" : "s"}. La curva aparecerá con más historial.
          </p>
        )}
      </Section>

      <Section title="PCK por keypoint">
        <PerKeypointChart report={report} />
      </Section>

      <Section title="Detalle por keypoint">
        <PerKeypointTable report={report} />
      </Section>
    </div>
  );
}

function SummaryCards({ report }: { report: EvalReport }) {
  const s = report.summary;
  const cards = [
    {
      icon: <Target className="h-4 w-4" />,
      label: "Recall global",
      value: s.overall_recall == null ? "—" : pct(s.overall_recall),
      tone: tonePct(s.overall_recall ?? 0, 0.5, 0.8),
    },
    {
      icon: <Target className="h-4 w-4" />,
      label: "PCK @ 5 px",
      value: s.overall_pck_5px == null ? "—" : pct(s.overall_pck_5px),
      tone: tonePct(s.overall_pck_5px ?? 0, 0.5, 0.8),
    },
    {
      icon: <Target className="h-4 w-4" />,
      label: "PCK @ 10 px",
      value: s.overall_pck_10px == null ? "—" : pct(s.overall_pck_10px),
      tone: tonePct(s.overall_pck_10px ?? 0, 0.6, 0.85),
    },
    {
      icon: <Target className="h-4 w-4" />,
      label: "PCK @ 20 px",
      value: s.overall_pck_20px == null ? "—" : pct(s.overall_pck_20px),
      tone: tonePct(s.overall_pck_20px ?? 0, 0.7, 0.9),
    },
    {
      icon: <Crosshair className="h-4 w-4" />,
      label: "Error medio (px)",
      value: s.mean_error_px == null ? "—" : s.mean_error_px.toFixed(1),
      tone: "neutral" as const,
    },
    {
      icon: <Crosshair className="h-4 w-4" />,
      label: "Error homografía (m)",
      value: s.mean_homography_error_m == null ? "—" : s.mean_homography_error_m.toFixed(2),
      tone: "neutral" as const,
    },
    {
      icon:
        (s.homography_failure_rate ?? 0) > 0.2 ? (
          <AlertTriangle className="h-4 w-4" />
        ) : (
          <CheckCircle2 className="h-4 w-4" />
        ),
      label: "Fallos homografía",
      value: s.homography_failure_rate == null ? "—" : pct(s.homography_failure_rate),
      tone: tonePct(1 - (s.homography_failure_rate ?? 0), 0.7, 0.9),
    },
    {
      icon: <Target className="h-4 w-4" />,
      label: "KPs detectados / frame",
      value: s.mean_kps_detected_per_frame.toFixed(1),
      tone: "neutral" as const,
    },
    {
      icon: <Target className="h-4 w-4" />,
      label: "KPs GT / frame",
      value: s.mean_kps_gt_per_frame.toFixed(1),
      tone: "neutral" as const,
    },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {cards.map((c) => (
        <div key={c.label} className="rounded-lg border border-border bg-card p-3">
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
            {c.icon}
            {c.label}
          </div>
          <div className={`mt-1 text-xl font-bold ${toneClass(c.tone)}`}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}

function HistoryChart({ history }: { history: EvalHistoryEntry[] }) {
  const data = [...history]
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    .map((e) => ({
      t: e.timestamp.slice(5, 16).replace("T", " "),
      hash: e.git_hash,
      pck5: e.overall_pck_5px != null ? e.overall_pck_5px * 100 : null,
      pck10: e.overall_pck_10px != null ? e.overall_pck_10px * 100 : null,
      pck20: e.overall_pck_20px != null ? e.overall_pck_20px * 100 : null,
    }));
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis dataKey="t" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
          <YAxis
            tick={{ fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            domain={[0, 100]}
            unit="%"
            width={40}
          />
          <Tooltip
            contentStyle={{
              background: "var(--popover)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v: number) => `${v.toFixed(1)}%`}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="pck5" name="PCK@5" stroke="#ef4444" dot strokeWidth={2} />
          <Line
            type="monotone"
            dataKey="pck10"
            name="PCK@10"
            stroke="#f59e0b"
            dot
            strokeWidth={2}
          />
          <Line
            type="monotone"
            dataKey="pck20"
            name="PCK@20"
            stroke="#10b981"
            dot
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function PerKeypointChart({ report }: { report: EvalReport }) {
  const data = useMemo(
    () =>
      Object.entries(report.per_keypoint).map(([name, k]) => ({
        name,
        pck10: k.pck_10px != null ? k.pck_10px * 100 : null,
        recall: k.recall != null ? k.recall * 100 : null,
      })),
    [report],
  );
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 16, left: -10, bottom: 60 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 9 }}
            tickLine={false}
            axisLine={false}
            angle={-45}
            textAnchor="end"
            interval={0}
            height={60}
          />
          <YAxis
            tick={{ fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            domain={[0, 100]}
            unit="%"
            width={40}
          />
          <Tooltip
            contentStyle={{
              background: "var(--popover)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v: number) => `${v.toFixed(1)}%`}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="pck10" name="PCK@10" fill="#f59e0b" />
          <Bar dataKey="recall" name="Recall" fill="#3b82f6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function PerKeypointTable({ report }: { report: EvalReport }) {
  const rows = Object.entries(report.per_keypoint).sort(
    (a, b) =>
      (b[1].recall ?? 0) - (a[1].recall ?? 0) || (b[1].pck_10px ?? 0) - (a[1].pck_10px ?? 0),
  );
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Keypoint</TableHead>
            <TableHead className="text-right">GT vis.</TableHead>
            <TableHead className="text-right">Detect.</TableHead>
            <TableHead className="text-right">Recall</TableHead>
            <TableHead className="text-right">PCK@5</TableHead>
            <TableHead className="text-right">PCK@10</TableHead>
            <TableHead className="text-right">PCK@20</TableHead>
            <TableHead className="text-right">Err px</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(([name, k]) => (
            <TableRow key={name}>
              <TableCell className="font-mono text-xs">{name}</TableCell>
              <TableCell className="text-right">{k.n_gt}</TableCell>
              <TableCell className="text-right">{k.n_detected}</TableCell>
              <TableCell className="text-right">{k.recall == null ? "—" : pct(k.recall)}</TableCell>
              <TableCell className="text-right">
                {k.pck_5px == null ? "—" : pct(k.pck_5px)}
              </TableCell>
              <TableCell className="text-right">
                {k.pck_10px == null ? "—" : pct(k.pck_10px)}
              </TableCell>
              <TableCell className="text-right">
                {k.pck_20px == null ? "—" : pct(k.pck_20px)}
              </TableCell>
              <TableCell className="text-right">
                {k.mean_error_px == null ? "—" : k.mean_error_px.toFixed(1)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h2>
      <div className="rounded-lg border border-border bg-card p-3">{children}</div>
    </section>
  );
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function fmtDate(iso: string): string {
  return iso.replace("T", " ").slice(0, 16);
}

function tonePct(v: number, low: number, high: number): "bad" | "ok" | "good" {
  if (v >= high) return "good";
  if (v >= low) return "ok";
  return "bad";
}

function toneClass(t: "bad" | "ok" | "good" | "neutral"): string {
  switch (t) {
    case "good":
      return "text-emerald-400";
    case "ok":
      return "text-amber-400";
    case "bad":
      return "text-red-400";
    default:
      return "";
  }
}
