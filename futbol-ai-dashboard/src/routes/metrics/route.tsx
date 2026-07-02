import { Link, Outlet, createFileRoute, useRouterState } from "@tanstack/react-router";

export const Route = createFileRoute("/metrics")({
  head: () => ({
    meta: [
      { title: "Metrics — Futbol AI" },
      { name: "description", content: "Evaluación técnica de los modelos del pipeline." },
      { name: "robots", content: "noindex,nofollow" },
    ],
  }),
  component: MetricsLayout,
});

const TABS = [
  { to: "/metrics", label: "Cancha", exact: true },
  { to: "/metrics/ball", label: "Balón", exact: false },
] as const;

function MetricsLayout() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="flex w-full flex-col gap-5 p-6">
      <div>
        <h1 className="text-2xl font-bold">Evaluación de modelos</h1>
        <p className="text-sm text-muted-foreground">
          Métricas técnicas por modelo del pipeline de visión.
        </p>
      </div>

      <nav className="flex gap-1 border-b border-border">
        {TABS.map((tab) => {
          const active = tab.exact ? pathname === tab.to : pathname.startsWith(tab.to);
          return (
            <Link
              key={tab.to}
              to={tab.to}
              className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                active
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>

      <Outlet />
    </div>
  );
}
