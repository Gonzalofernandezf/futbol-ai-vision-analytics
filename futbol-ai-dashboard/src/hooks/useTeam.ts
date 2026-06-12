import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import type { Team, TeamPlayer } from "@/types/team";
import { useMatchData } from "@/hooks/useMatchData";

const STORAGE_KEY = "futbol-ai-team-v3";
const LEGACY_STORAGE_KEYS = ["futbol-ai-team", "futbol-ai-team-v2"];

function genId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `p-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function emptyTeam(): Team {
  return { players: [], lastUpdated: new Date().toISOString() };
}

function readStorage(): Team {
  if (typeof window === "undefined") return emptyTeam();
  // Drop seeded data from older versions so hardcoded names don't linger.
  for (const k of LEGACY_STORAGE_KEYS) {
    try { window.localStorage.removeItem(k); } catch { /* ignore */ }
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as Team;
  } catch {
    /* ignore */
  }
  return emptyTeam();
}

async function compressImage(file: File): Promise<string> {
  const dataUrl: string = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("No se pudo leer el archivo"));
    reader.readAsDataURL(file);
  });
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = () => reject(new Error("Imagen inválida"));
    i.src = dataUrl;
  });
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas no soportado");
  ctx.fillStyle = "#1f2937";
  ctx.fillRect(0, 0, size, size);
  const scale = Math.min(size / img.width, size / img.height);
  const w = img.width * scale;
  const h = img.height * scale;
  ctx.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);
  return canvas.toDataURL("image/jpeg", 0.85);
}

export function useTeam() {
  const [team, setTeam] = useState<Team>(() => ({ players: [], lastUpdated: new Date().toISOString() }));
  const [hydrated, setHydrated] = useState(false);
  const { players: jsonPlayers } = useMatchData();

  useEffect(() => {
    setTeam(readStorage());
    setHydrated(true);
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        try {
          setTeam(JSON.parse(e.newValue) as Team);
        } catch {
          /* ignore */
        }
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // Bootstrap roster from JSON tracker IDs once data is available. Preserves
  // any user-edited entries (matched by trackerId) and only adds missing ones.
  useEffect(() => {
    if (!hydrated || jsonPlayers.length === 0) return;
    const existingTrackers = new Set(
      team.players.map((p) => p.trackerId).filter((t): t is number => t != null),
    );
    const missing = jsonPlayers.filter((jp) => {
      const tid = Number(jp.id);
      return Number.isFinite(tid) && !existingTrackers.has(tid);
    });
    if (missing.length === 0) return;
    const additions: TeamPlayer[] = missing.map((jp) => {
      const tid = Number(jp.id);
      return {
        id: genId(),
        name: "",
        jerseyNumber: tid,
        position: "",
        team: jp.team,
        status: "fit",
        trackerId: tid,
      };
    });
    const next: Team = {
      players: [...team.players, ...additions],
      lastUpdated: new Date().toISOString(),
    };
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
    setTeam(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated, jsonPlayers]);

  const persist = useCallback((next: Team): boolean => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      setTeam(next);
      return true;
    } catch (e) {
      const isQuota =
        e instanceof DOMException &&
        (e.name === "QuotaExceededError" || e.code === 22);
      toast.error(
        isQuota
          ? "Almacenamiento lleno — borra avatares viejos antes de subir nuevos"
          : "No se pudo guardar el equipo",
      );
      return false;
    }
  }, []);

  const addPlayer = useCallback(
    (data: Omit<TeamPlayer, "id">) => {
      const next: Team = {
        players: [...team.players, { ...data, id: genId() }],
        lastUpdated: new Date().toISOString(),
      };
      if (persist(next)) toast.success("Jugador añadido");
    },
    [team, persist],
  );

  const updatePlayer = useCallback(
    (id: string, patch: Partial<Omit<TeamPlayer, "id">>) => {
      const next: Team = {
        players: team.players.map((p) => (p.id === id ? { ...p, ...patch } : p)),
        lastUpdated: new Date().toISOString(),
      };
      if (persist(next)) toast.success("Jugador actualizado");
    },
    [team, persist],
  );

  const deletePlayer = useCallback(
    (id: string) => {
      const next: Team = {
        players: team.players.filter((p) => p.id !== id),
        lastUpdated: new Date().toISOString(),
      };
      if (persist(next)) toast.success("Jugador eliminado");
    },
    [team, persist],
  );

  const uploadAvatar = useCallback(
    async (playerId: string, file: File) => {
      const allowed = ["image/jpeg", "image/png", "image/webp"];
      if (!allowed.includes(file.type)) {
        toast.error("Formato no permitido (usa JPG, PNG o WEBP)");
        return;
      }
      if (file.size > 2 * 1024 * 1024) {
        toast.error("La imagen supera 2 MB");
        return;
      }
      try {
        const base64 = await compressImage(file);
        const next: Team = {
          players: team.players.map((p) =>
            p.id === playerId ? { ...p, avatarBase64: base64 } : p,
          ),
          lastUpdated: new Date().toISOString(),
        };
        if (persist(next)) toast.success("Avatar actualizado");
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Error al procesar la imagen");
      }
    },
    [team, persist],
  );

  const getPlayerById = useCallback(
    (id: string) => team.players.find((p) => p.id === id),
    [team],
  );
  const getPlayerByTrackerId = useCallback(
    (trackerId: number) => team.players.find((p) => p.trackerId === trackerId),
    [team],
  );

  return {
    team,
    hydrated,
    addPlayer,
    updatePlayer,
    deletePlayer,
    uploadAvatar,
    getPlayerById,
    getPlayerByTrackerId,
  };
}