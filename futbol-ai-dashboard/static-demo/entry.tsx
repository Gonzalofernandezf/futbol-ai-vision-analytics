import { createRoot } from "react-dom/client";
import { QueryClient } from "@tanstack/react-query";
import { createRouter, createHashHistory, RouterProvider } from "@tanstack/react-router";
import { routeTree } from "../src/routeTree.gen";
import { generateMockMatchData } from "./mock-data";
import "../src/styles.css";

// Under file:// some browsers make real localStorage unreliable (slow/blocked
// storage-partition IPC, or a hard SecurityError). This is a throwaway demo
// snapshot anyway, so swap in an in-memory Storage — fast and deterministic.
class MemoryStorage implements Storage {
  private map = new Map<string, string>();
  get length() {
    return this.map.size;
  }
  clear(): void {
    this.map.clear();
  }
  getItem(key: string): string | null {
    return this.map.has(key) ? this.map.get(key)! : null;
  }
  key(index: number): string | null {
    return [...this.map.keys()][index] ?? null;
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}
const memoryLocalStorage = new MemoryStorage();
Object.defineProperty(window, "localStorage", { value: memoryLocalStorage, configurable: true });
Object.defineProperty(window, "sessionStorage", { value: new MemoryStorage(), configurable: true });

const mockMatchData = generateMockMatchData();
const mockJson = JSON.stringify(mockMatchData);

// useTeam() mounts multiple times per page (sidebar list, player card, team
// route) and each instance independently bootstraps "missing" roster entries
// from match_data.json on first render. Pre-seeding the roster means every
// instance sees zero missing players from its very first render, so none of
// them ever call setTeam — sidesteps a render race between instances that
// otherwise pegs the tab (reproduced consistently when navigating from the
// dashboard into /team or /compare in this single-bundle CSR build).
const seededRoster = {
  players: Object.entries(mockMatchData.players).map(([id, p]) => ({
    id: `seed-${id}`,
    name: "",
    jerseyNumber: Number(id),
    position: "",
    team: p.team,
    status: "fit" as const,
    trackerId: Number(id),
  })),
  lastUpdated: new Date().toISOString(),
};
memoryLocalStorage.setItem("futbol-ai-team-v3", JSON.stringify(seededRoster));

const realFetch = window.fetch.bind(window);
window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  if (url.endsWith("/match_data.json")) {
    return Promise.resolve(
      new Response(mockJson, { status: 200, headers: { "content-type": "application/json" } }),
    );
  }
  return realFetch(input, init);
}) as typeof window.fetch;

const queryClient = new QueryClient();
const router = createRouter({
  routeTree,
  context: { queryClient },
  scrollRestoration: false,
  defaultPreloadStaleTime: 0,
  history: createHashHistory(),
});

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("#root not found");
createRoot(rootEl).render(<RouterProvider router={router} />);
