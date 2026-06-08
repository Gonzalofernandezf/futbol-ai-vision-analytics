# MatchAnalytics v2 — JSON Schema

`match_data.json` emitido por el pipeline desde la versión 2.

---

## Estructura raíz

```json
{
  "schema_version": 2,
  "match_meta": { ... },
  "players": { "1": { ... }, "2": { ... } },
  "team_stats": { "1": { ... }, "2": { ... } }
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `schema_version` | `int` | Siempre `2` para este formato |
| `match_meta` | objeto | Metadatos del partido (sin cambios respecto a v1) |
| `players` | objeto | Estadísticas por jugador, clave = ID string |
| `team_stats` | objeto | Agregados por equipo, claves `"1"` y `"2"` |

---

## match_meta

```json
"match_meta": {
  "duration_seconds": 35.00,
  "fps": 29.97,
  "home_possession": 41.9,
  "away_possession": 58.1
}
```

---

## players[id]

Cada jugador incluye los **campos crudos** (sin cambios desde v1) más un bloque `derived`.

### Campos crudos (v1, sin cambios)

| Campo | Tipo | Unidad |
|---|---|---|
| `team` | `int` | `1` o `2` |
| `max_speed_kmh` | `float` | km/h |
| `total_distance_m` | `float` | metros |
| `max_acceleration_ms2` | `float` | m/s² (ventana 0.5 s sobre frames crudos) |
| `speed_over_time` | `[float\|null]` | km/h, 1 valor/s, null = sin dato |
| `position_history` | `[[x,y]\|null]` | metros en cancha 100 × 64 m, 1 valor/frame |

### derived (v2, nuevo)

```json
"derived": {
  "active_time_sec": 28.0,
  "missing_data_pct": 20.0,
  "sprints": {
    "count": 3,
    "longest_sec": 4,
    "total_distance_m": 62.5
  },
  "speed_zones_sec": {
    "walk": 8,
    "jog": 10,
    "run": 6,
    "sprint": 4
  },
  "high_intensity_pct": 35.71,
  "accelerations": {
    "high_count": 2,
    "max_ms2": 4.2
  },
  "halves": {
    "first":  { "distance_m": 210.5, "avg_speed_kmh": 9.8, "sprints": 2 },
    "second": { "distance_m": 185.0, "avg_speed_kmh": 8.1, "sprints": 1 }
  },
  "drop_off_pct": -17.35,
  "peak_window": {
    "start_sec": 420,
    "end_sec": 720,
    "avg_speed_kmh": 13.4
  },
  "position": {
    "centroid_m": [62.3, 18.7],
    "dispersion_m": 14.2,
    "coverage_pct": 22.0,
    "dominant_cell": [6, 2],
    "lateral_band": "left"
  }
}
```

#### Definiciones detalladas

**active_time_sec**
Segundos en `speed_over_time` con valor no nulo.

**missing_data_pct**
`null_seconds / total_seconds * 100`. Si el array está vacío, `null`.

**sprints**
Tramo contiguo de ≥ 1 s con velocidad ≥ `SPRINT_THRESHOLD_KMH` (default: **21 km/h**).
Un null interrumpe el tramo.

- `count`: número de tramos
- `longest_sec`: duración del tramo más largo (segundos)
- `total_distance_m`: distancia acumulada en todos los tramos, estimada como `Σ v/3.6` (cada segundo contribuye `v km/h ÷ 3.6`)

**speed_zones_sec**
Segundos (sin nulos) en cada zona. Solo cuenta instantes con dato.

| Zona | Rango |
|---|---|
| `walk` | v < 7 km/h |
| `jog` | 7 ≤ v < 15 km/h |
| `run` | 15 ≤ v < 21 km/h |
| `sprint` | v ≥ 21 km/h |

Umbrales configurables vía `.env`: `SPEED_ZONE_WALK_KMH`, `SPEED_ZONE_JOG_KMH`, `SPEED_ZONE_RUN_KMH`.

**high_intensity_pct**
`count(v ≥ HI_THRESHOLD_KMH) / active_time_sec * 100`.
Default: `HI_THRESHOLD_KMH` = **15 km/h**. `null` si no hay datos activos.

**accelerations**
Aceleraciones discretas calculadas sobre pares de segundos consecutivos válidos:
`a = (v[i] - v[i-1]) / 3.6` m/s² (Δt = 1 s, velocidades convertidas de km/h).
Un null reinicia la cadena.

- `high_count`: número de **cruces ascendentes** del umbral `HIGH_ACCEL_THRESHOLD_MS2` (default: **3 m/s²**). Un cruce ascendente = transición de `a < umbral` a `a ≥ umbral`.
- `max_ms2`: máximo valor de aceleración calculado sobre los pares válidos. `null` si no hay pares.

**halves**
Split en `floor(duration_seconds / 2)` segundos.

Por cada mitad:
- `distance_m`: `Σ v/3.6` sobre segundos válidos de esa mitad
- `avg_speed_kmh`: media de velocidades válidas
- `sprints`: número de tramos sprint en esa mitad

Cualquier campo es `null` si la mitad no tiene datos.

**drop_off_pct**
`(avg_2H - avg_1H) / avg_1H * 100`.
Negativo = el jugador bajó el ritmo en la segunda mitad.
`null` si `avg_1H` es `null` o cero.

**peak_window**
Ventana deslizante de `PEAK_WINDOW_SEC` segundos (default: **300 s = 5 min**) con mayor velocidad media (ignorando nulos).
Si el vídeo es más corto que la ventana, la ventana se recorta al tamaño del vídeo.
`null` si no hay ningún dato válido.

- `start_sec` / `end_sec`: límites de la ventana (en segundos desde el inicio)
- `avg_speed_kmh`: velocidad media en la ventana (solo valores no nulos)

**position**
Derivado de `position_history` (coordenadas en metros, cancha 100 × 64 m).
Requiere al menos 2 puntos válidos; en caso contrario todos los campos son `null`.

- `centroid_m`: `[x, y]` — posición media del jugador
- `dispersion_m`: `sqrt(var_x + var_y)` — dispersión 2D alrededor del centroide
- `coverage_pct`: porcentaje de celdas de la rejilla `10 × 10` visitadas al menos una vez
- `dominant_cell`: `[fila, columna]` (0-indexed) de la celda con más apariciones
- `lateral_band`: banda lateral basada en la coordenada y del centroide

| Banda | Condición (pitch_width = 64 m) |
|---|---|
| `left` | y < 64/3 ≈ 21.3 m |
| `center` | 21.3 ≤ y ≤ 42.7 m |
| `right` | y > 64×2/3 ≈ 42.7 m |

---

## team_stats[1\|2]

```json
"team_stats": {
  "1": {
    "players_count": 6,
    "percentiles": {
      "total_distance_m":   { "p5": 280.0, "p25": 320.0, "p50": 380.0, "p75": 430.0, "p95": 490.0 },
      "max_speed_kmh":      { "p5": 16.2,  "p25": 18.4,  "p50": 21.0,  "p75": 23.5,  "p95": 26.1  },
      "sprints_count":      { "p5": 1.0,   "p25": 2.0,   "p50": 3.0,   "p75": 5.0,   "p95": 8.0   },
      "high_intensity_pct": { "p5": 10.0,  "p25": 18.0,  "p50": 25.0,  "p75": 35.0,  "p95": 50.0  },
      "drop_off_pct":       { "p5": -30.0, "p25": -15.0, "p50": -5.0,  "p75": 5.0,   "p95": 20.0  },
      "active_time_sec":    { "p5": 1200,  "p25": 1500,  "p50": 1800,  "p75": 2100,  "p95": 2500  }
    },
    "leaders": {
      "total_distance_m": { "player_id": "7",  "value": 11240.0 },
      "max_speed_kmh":    { "player_id": "11", "value": 29.3    },
      "sprints_count":    { "player_id": "9",  "value": 14      }
    }
  },
  "2": { ... }
}
```

Los percentiles se calculan mediante interpolación lineal (equivalente a `numpy.percentile` con `method='linear'`) sobre los jugadores válidos de cada equipo.
Si un equipo no tiene jugadores detectados, su valor es `null`.

---

## Reglas generales

- **Unidades**: metros, segundos, km/h, m/s². Sin mezclar.
- **Nulos**: si una métrica no se puede calcular, el valor es `null`. La clave siempre está presente.
- **Determinismo**: mismo input → mismo output. Sin aleatoriedad.
- **Compatibilidad hacia atrás**: todos los campos v1 se mantienen intactos. El dashboard v1 sigue funcionando si ignora los nuevos campos.

---

## Umbrales por defecto (todos sobreescribibles vía `.env`)

| Variable | Default | Descripción |
|---|---|---|
| `SPRINT_THRESHOLD_KMH` | 21.0 | Velocidad mínima para considerar sprint |
| `HI_THRESHOLD_KMH` | 15.0 | Umbral de alta intensidad |
| `HIGH_ACCEL_THRESHOLD_MS2` | 3.0 | Umbral de aceleración alta |
| `PEAK_WINDOW_SEC` | 300 | Duración de la ventana pico (segundos) |
| `ANALYTICS_GRID_ROWS` | 10 | Filas del grid de cobertura |
| `ANALYTICS_GRID_COLS` | 10 | Columnas del grid de cobertura |
| `SPEED_ZONE_WALK_KMH` | 7.0 | Límite superior zona walk |
| `SPEED_ZONE_JOG_KMH` | 15.0 | Límite superior zona jog |
| `SPEED_ZONE_RUN_KMH` | 21.0 | Límite superior zona run |
