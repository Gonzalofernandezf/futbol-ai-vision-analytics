import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import { AlertTriangle, Crosshair, Target } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { EvalBallHistoryEntry, EvalBallPerFrame, EvalBallReport } from "@/types/eval-ball";

export const Route = createFileRoute("/metrics/ball")({
  component: BallMetricsPage,
});

function BallMetricsPage() {
  const [report, setReport] = useState<EvalBallReport | null>(null);
  const [history, setHistory] = useState<EvalBallHistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/eval_ball_report.json").then((r) =>
        r.ok
          ? (r.json() as Promise<EvalBallReport>)
          : Promise.reject(new Error(`report ${r.status}`)),
      ),
      fetch("/eval_ball_history.json").then((r) =>
        r.ok
          ? (r.json() as Promise<EvalBallHistoryEntry[]>)
          : Promise.resolve([] as EvalBallHistoryEntry[]),
      ),
    ])
      .then(([rep, hist]) => {
        setReport(rep);
        setHistory(hist);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Error"));
  }, []);

  if (error) {
    return (
      <div className="text-sm text-destructive">
        No se pudo cargar la evaluación de balón: {error}
      </div>
    );
  }
  if (!report) {
    return <div className="text-sm text-muted-foreground">Cargando evaluación…</div>;
  }

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          {report.metadata.model} · {report.metadata.n_frames} frames (
          {report.metadata.n_gt_positive} con balón, {report.metadata.n_gt_negative} sin balón) ·{" "}
          {fmtDate(report.metadata.timestamp)}
        </p>
        <code className="rounded bg-muted px-2 py-1 text-xs text-muted-foreground">
          {report.metadata.git_commit.short_hash ?? "—"} ·{" "}
          {report.metadata.git_commit.message ?? "sin commit"}
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

      <Section title="Frames problemáticos (falsos positivos / falsos negativos)">
        <ProblemFramesTable report={report} />
      </Section>
    </div>
  );
}

function SummaryCards({ report }: { report: EvalBallReport }) {
  const s = report.summary;
  const cards = [
    {
      icon: <Target className="h-4 w-4" />,
      label: "Recall",
      value: s.recall == null ? "—" : pct(s.recall),
      tone: tonePct(s.recall ?? 0, 0.6, 0.85),
    },
    {
      icon: <Target className="h-4 w-4" />,
      label: "Precision",
      value: s.precision == null ? "—" : pct(s.precision),
      tone: tonePct(s.precision ?? 0, 0.7, 0.9),
    },
    {
      icon: <Target className="h-4 w-4" />,
      label: "F1",
      value: s.f1 == null ? "—" : pct(s.f1),
      tone: tonePct(s.f1 ?? 0, 0.6, 0.85),
    },
    {
      icon: <AlertTriangle className="h-4 w-4" />,
      label: "Falsos positivos",
      value: s.false_positive_rate == null ? "—" : pct(s.false_positive_rate),
      tone: tonePct(1 - (s.false_positive_rate ?? 0), 0.8, 0.95),
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
      icon: <Crosshair className="h-4 w-4" />,
      label: "Error medio (px)",
      value: s.mean_error_px == null ? "—" : s.mean_error_px.toFixed(1),
      tone: "neutral" as const,
    },
    {
      icon: <Target className="h-4 w-4" />,
      label: "Confianza media (TP)",
      value: s.mean_confidence_tp == null ? "—" : s.mean_confidence_tp.toFixed(2),
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

function HistoryChart({ history }: { history: EvalBallHistoryEntry[] }) {
  const data = [...history]
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    .map((e) => ({
      t: e.timestamp.slice(5, 16).replace("T", " "),
      hash: e.git_hash,
      recall: e.recall != null ? e.recall * 100 : null,
      precision: e.precision != null ? e.precision * 100 : null,
      pck10: e.overall_pck_10px != null ? e.overall_pck_10px * 100 : null,
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
          <Line
            type="monotone"
            dataKey="recall"
            name="Recall"
            stroke="#3b82f6"
            dot
            strokeWidth={2}
          />
          <Line
            type="monotone"
            dataKey="precision"
            name="Precision"
            stroke="#10b981"
            dot
            strokeWidth={2}
          />
          <Line
            type="monotone"
            dataKey="pck10"
            name="PCK@10"
            stroke="#f59e0b"
            dot
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ProblemFramesTable({ report }: { report: EvalBallReport }) {
  const rows = useMemo(
    () =>
      report.per_frame
        .filter((f) => f.outcome === "FP" || f.outcome === "FN")
        .sort((a, b) => (a.outcome === b.outcome ? 0 : a.outcome === "FP" ? -1 : 1)),
    [report],
  );
  const okCount = report.per_frame.length - rows.length;

  if (rows.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Sin falsos positivos ni negativos — todo limpio.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-muted-foreground">
        Mostrando solo los {rows.length} frames con error, de {report.per_frame.length} evaluados en
        total ({okCount} correctos — aciertos o "sin balón" bien detectados — no se listan acá).
      </p>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Frame</TableHead>
              <TableHead>Tipo</TableHead>
              <TableHead className="text-right">Confianza</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((f: EvalBallPerFrame) => (
              <TableRow key={f.filename}>
                <TableCell className="font-mono text-xs">{f.filename}</TableCell>
                <TableCell>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                      f.outcome === "FP"
                        ? "bg-red-500/15 text-red-400"
                        : "bg-amber-500/15 text-amber-400"
                    }`}
                  >
                    {f.outcome === "FP" ? "Falso positivo" : "Falso negativo"}
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  {f.confidence == null ? "—" : f.confidence.toFixed(2)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
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
