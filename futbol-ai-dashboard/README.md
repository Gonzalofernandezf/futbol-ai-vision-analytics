# Futbol AI — Dashboard de Análisis Táctico

Aplicación frontend (React 19 + TanStack Start + Vite + Tailwind v4) que
visualiza datos tácticos generados por nuestro backend Python a partir de
vídeo de partidos grabados con una sola cámara.

## Cómo correr

```bash
npm install
npm run dev
```

El dashboard carga datos desde `/public/match_data.json`. Reemplazá ese
archivo con la salida real del pipeline Python — la estructura está
documentada en `src/types/match.ts`.

## Estructura del proyecto

```
public/
  match_data.json          # JSON de ejemplo (12 jugadores, 2 equipos)
  demo_video.mp4           # Placeholder de vídeo (poner el real aquí)
src/
  routes/
    __root.tsx             # Layout global + nav lateral
    index.tsx              # Dashboard principal (/)
    compare.tsx            # Comparativa de 2 jugadores (/compare)
  components/
    AppNav.tsx             # Nav vertical persistente
    LoadingSkeleton.tsx
    dashboard/             # Sidebar, heatmap, video, header del partido
    player/                # FIFA-card, gráfico de velocidad, insights
  hooks/
    useMatchData.ts        # fetch del JSON
  contexts/
    SelectionContext.tsx   # Equipo / jugador / rango de minutos
  types/
    match.ts               # MatchMeta, PlayerStats, MatchData
```

## Notas

- Tema oscuro por defecto; acento verde césped (`--primary`).
- El heatmap normaliza posiciones desde metros (cancha 100×64) al tamaño
  del SVG; usa props `width` / `height` para configurarlo.
- La posición del jugador (`DEF`, `MID`, etc.) está hard-codeada como
  placeholder hasta que el backend la exponga.
- Si necesitás añadir más jugadores, editá `public/match_data.json`
  respetando la estructura de `MatchData`.

_Nota técnica:_ el proyecto usa **TanStack Router** (file-based) en lugar
de `react-router-dom`; la API de navegación es equivalente (`<Link to>`,
`useNavigate`) y los archivos en `src/routes/` definen las URLs.
_(El brief pedía React Router; usamos TanStack Router porque es el
ruteador del stack base de este proyecto.)_

## Gestión del equipo

La ruta `/team` permite al cuerpo técnico mantener el plantel (alta/edición/
baja de jugadores, dorsal, posición, estado físico y avatar). Los datos
viven en `localStorage` bajo la key `futbol-ai-team` — no hay backend.

- Los avatares se comprimen a 256×256 JPEG (q=0.85) usando un canvas antes
  de guardarse como base64. Aun así, `localStorage` está limitado a ~5 MB
  por origen, por lo que conviene mantener pocas docenas de avatares; si
  aparece el toast "Almacenamiento lleno" hay que borrar avatares viejos.
- El enlace con las estadísticas del partido se hace por `trackerId`: si
  un jugador del plantel tiene un `trackerId` que coincide con un ID en
  `match_data.json`, su nombre, dorsal y avatar reemplazan al placeholder
  en el dashboard principal y en la comparativa.
- La página vive en `src/routes/team.tsx` (no `src/pages/`, porque
  TanStack Router usa file-based routing bajo `src/routes/`).