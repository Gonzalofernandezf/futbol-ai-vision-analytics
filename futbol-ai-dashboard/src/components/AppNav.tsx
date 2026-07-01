import { Link, useRouterState } from "@tanstack/react-router";
import { House, GitCompare, Activity, Users, FlaskConical } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";

const items = [
  { to: "/", label: "Dashboard", Icon: House },
  { to: "/team", label: "Equipo", Icon: Users },
  { to: "/compare", label: "Comparar", Icon: GitCompare },
  { to: "/metrics", label: "Métricas modelo", Icon: FlaskConical },
  { to: "/admin", label: "Admin", Icon: Activity },
] as const;

export function AppNav() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <nav className="flex h-screen w-16 flex-col items-center gap-2 border-r border-border bg-sidebar py-4">
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-md bg-primary/15 text-primary font-bold">
        F
      </div>
      {items.map(({ to, label, Icon }) => {
        const active = pathname === to;
        return (
          <Link
            key={to}
            to={to}
            title={label}
            className={`group relative flex h-10 w-10 items-center justify-center rounded-md transition-colors ${
              active
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            }`}
          >
            <Icon className="h-5 w-5" />
            {active && (
              <span className="absolute -bottom-1 h-0.5 w-6 rounded-full bg-primary" />
            )}
          </Link>
        );
      })}
      <div className="mt-auto">
        <ThemeToggle />
      </div>
    </nav>
  );
}