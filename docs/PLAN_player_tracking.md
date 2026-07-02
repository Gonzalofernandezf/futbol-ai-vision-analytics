# Plan — Mejora de detección y persistencia de IDs de jugadores

**Contexto:** después de fix del bug de balón + integración del nuevo modelo de balón, el detector de jugadores y la persistencia de tracking quedaban como cuello de botella principal. Este documento recoge el plan original y su estado de ejecución actualizado a 2026-07-02.

**Regla general:** un PR por Tier. Antes/después medible con `utils/perf_monitor.py`.

---

## Estado de ejecución

| Tier | Descripción | Estado |
|------|-------------|--------|
| T0 + T5 | Quick wins de configuración + observabilidad | ✅ Completo |
| T1 | Migración ByteTrack → BoT-SORT con ReID | ✅ Completo |
| T4 | Team assignment robusto | ✅ Completo |
| — | Reentrenamiento `modelo_balon.pt` | ✅ Completo |
| — | Reentrenamiento `modelo_cancha.pt` | ✅ Completo (PCK@5px 93%, error homografía 5.5m) |
| — | Heatmaps precisos por jugador | ✅ Completo (desbloqueado por nuevo modelo_cancha) |
| T2 | Cross-chunk ReID | 🔲 Pendiente |
| T3 | Reentrenamiento `best_100e.pt` (cámara baja) | 🔲 Pendiente |
| — | Análisis de balón parado | 🔲 Pendiente |

---

## Tiers completados

### ✅ Tier 0 — Quick wins de configuración

Corregidos errores de configuración del detector principal:
- `Trackers/tracker.py` usa `YOLO_CONF`/`YOLO_IOU` en vez de los del balón.
- `imgsz` pasado explícitamente (antes caía a 640, pérdida de recall en jugadores pequeños).
- `agnostic_nms=True` para que jugador y portero no se dupliquen en NMS.
- Parámetros de ByteTrack relajados: `lost_track_buffer=200`, `match_thresh=0.7`.
- Todos los parámetros expuestos en `config.py` y leídos desde `.env`.

### ✅ Tier 5 — Observabilidad

Métricas de tracking añadidas a `utils/perf_monitor.py`:
- `num_unique_player_ids`, `num_long_tracks`, `avg_track_duration_s`, `id_churn_ratio`, `team_flip_rate`.
- Se imprimen al final de cada run para comparar entre PRs.

### ✅ Tier 1 — BoT-SORT con ReID

Migración de `supervision.ByteTrack` a BoT-SORT con ReID vía `ultralytics.YOLO.track()`:
- ReID con OSNet (embeddings de apariencia de 512 dimensiones).
- Persistencia de IDs entre batches con `persist=True`.
- Config en `Trackers/botsort.yaml`.
- `get_object_tracks` adaptado para consumir `Boxes.id` directamente.

### ✅ Tier 4 — Team assignment robusto

- Segmentación HSV para descartar césped y fondo antes del clustering.
- Re-cluster periódico (resiste cambios de iluminación y sombras).
- Votación móvil por ID en vez de cacheo permanente.
- Portero tratado como clase separada (no contamina los clusters de equipo).

### ✅ Modelo de campo reentrenado (`modelo_cancha.pt`)

Entrenado en Kaggle con datasets de Roboflow. Resultados en eval válido (47 frames, 2026-07-02):

| Métrica | Antes | Después |
|---------|-------|---------|
| Recall | 39.9% | 94.9% |
| PCK@5px | 5.9% | 92.8% |
| PCK@10px | 8.2% | 94.5% |
| Error medio (px) | 557 px | 1.3 px |
| Fallo homografía | 22.2% | 2.1% |
| Error homografía (m) | 134.7 m | 5.5 m |

### ✅ Modelo de balón reentrenado (`modelo_balon.pt`)

Detector dedicado con pesos y thresholds propios, separado del detector de jugadores. Integrado en pipeline. KPIs evaluados con `eval/eval_ball.py` sobre 106 frames de ground truth.

### ✅ Heatmaps precisos por jugador

`position_history` (lista de `[x, y]` en metros) ya existía en el schema v2. La homografía ahora es fiable (error ~5.5m vs ~135m antes), lo que hace los heatmaps coherentes con el movimiento real. Validado visualmente en el dashboard. **Diferenciador #1 para el cliente.**

---

## Tiers pendientes

### 🔲 Tier 2 — Cross-chunk ReID

**Objetivo:** que un jugador mantenga el mismo `track_id` a lo largo de todo el partido, independientemente de en qué chunk fue procesado.

**Problema actual:** `run_chunked.py` divide el partido en chunks. Cada chunk asigna IDs locales desde cero — el jugador que era ID 5 en el chunk 1 puede ser ID 23 en el chunk 2. El JSON final puede tener 200+ IDs únicos para lo que en realidad son 22 jugadores, rompiendo las estadísticas por jugador (distancia total, velocidad, heatmap) que cruzan chunks.

**Decisión de arquitectura acordada:** la reconciliación de IDs se hace como **paso de post-procesado** al final del merge, no de forma secuencial chunk a chunk. Esto permite que los chunks sigan corriendo en paralelo en Kaggle (múltiples CPUs) sin coordinación entre ellos.

**Diseño**

Cada chunk, al terminar, guarda un fichero de estado con los embeddings de sus tracks:

```python
# stubs/chunk_state_{N}.pkl — generado al final de cada chunk
chunk_state = {
    track_id: {
        "team_id": int,
        "embeddings": List[np.ndarray],   # embeddings OSNet del track (512-d)
        "mean_embedding": np.ndarray,     # media normalizada para el matching
        "last_position_m": (x, y),        # última posición en metros (campo)
        "frame_start_global": int,
        "frame_end_global": int,
        "track_duration_frames": int,
    }
}
```

El paso de merge en `run_chunked.py` añade una etapa de relabeling:

1. Cargar todos los `chunk_state_{N}.pkl` en orden cronológico.
2. Para cada chunk N > 0, hacer Hungarian assignment entre sus tracks y los del chunk N-1 usando una matriz de costes:
   - **Coste de apariencia:** distancia coseno entre `mean_embedding`.
   - **Coste espacial:** distancia euclídea en metros entre `last_position_m` del chunk N-1 y la primera posición del chunk N.
   - **Coste temporal:** penalización si el gap entre chunks es > 30s (jugador puede haber salido del campo).
   - **Coste de equipo:** ∞ si `team_id` distinto (nunca asignar IDs entre equipos distintos).
3. Renombrar IDs locales de cada chunk → IDs globales del partido.
4. Re-escribir los JSONs de chunks con los IDs globales antes del merge final.

**Nuevo módulo:**

```
Trackers/
  cross_chunk_reid.py   # ChunkBridge class
```

**Interface:**

```python
bridge = ChunkBridge()
bridge.load_chunk_states(stubs_dir)          # carga todos los .pkl
global_id_maps = bridge.compute_global_ids() # Hungarian assignment entre chunks
bridge.relabel_json(chunk_json, chunk_idx, global_id_maps)  # reescribe IDs
```

**Validación:**
- Añadir `match_unique_player_ids` al RUN SUMMARY de `run_chunked.py`.
- Para un partido completo de 90 min: idealmente ≤ 30 IDs únicos globales.
- `id_churn_ratio` (de `perf_monitor`) debería bajar significativamente vs. sin T2.

**Esfuerzo:** 2-3 días.
**Riesgo:** medio. Pre-requisito: T1 completo ✅.

---

### 🔲 Tier 3 — Reentrenamiento `best_100e.pt` (cámara baja)

**Objetivo:** mejorar el recall del detector de jugadores en las condiciones reales de Dinamó: cámara fija a 3-4 metros de altura en posición lateral. A esa altura los jugadores del fondo aparecen más pequeños, con más oclusiones entre sí, y la perspectiva es diferente a la cenital.

**Decisión acordada sobre etiquetado manual:** para la primera iteración **no se etiqueta nada manualmente**. Los datasets de Roboflow ya configurados en `datasets/download_datasets.py` incluyen imágenes con cámara lateral de otros partidos. Si tras el reentrenamiento persisten fallos sistemáticos con el vídeo de Dinamó, se abriría una segunda iteración con 200-300 frames suyos etiquetados.

**Sub-tareas:**

**T3.a — Dataset**
3 datasets de Roboflow Universe configurados en `datasets/download_datasets.py` (footage de cámara lateral/tribuna, coherente con el caso Dinamó). No se añaden datos propios en esta iteración.

**Decisión:** el fine-tuning excluye la clase `ball`. `best_100e.pt` ya solo se usa para jugador/portero/árbitro — el balón lo cubre `modelo_balon.pt` por separado — así que las cajas de balón se descartan al fusionar los datasets (ruido de objeto pequeño sin aportar valor a este modelo).

**T3.b — Entrenamiento**
**Decisión revisada:** `best_100e.pt` es YOLOv8 — no es posible cargarlo como base YOLOv11m (arquitecturas incompatibles, no hay transfer de pesos entre ellas). Se entrena YOLOv11m desde `yolo11m.pt` (COCO-pretrained de Ultralytics), no desde `best_100e.pt`. Se acepta perder el warm-start de "cámara alta" porque (a) uno de los 3 datasets de T3.a ya formaba parte del entrenamiento original de `best_100e.pt`, así que ese conocimiento se reintroduce de todos modos, y (b) YOLOv11m aporta bloques de atención (C2PSA) con mejor desempeño documentado en objetos pequeños/ocluidos — exactamente el problema de cámara baja. Fallback si no converge bien: fine-tuning YOLOv8 clásico desde `best_100e.pt`.

Script listo en `datasets/train_jugadores.py`. Parámetros objetivo:
- Base: YOLOv11m (`yolo11m.pt`, COCO-pretrained)
- `imgsz=1280`
- 100 epochs
- Augmentation geométrico agresivo (perspectiva, rotación, shear) para cubrir variaciones de ángulo.
- Ejecutar en Kaggle con GPU.

**T3.c — Validación**
- mAP50 ≥ 0.85 en el holdout de validación del dataset.
- Correr `Main.py` con `video_OG.mp4` y comparar visualmente recall en momentos con jugadores pequeños/ocluidos.
- Grabar 2-3 minutos con la cámara de Dinamó (3-4m de altura) y validar antes de la próxima reunión con el cliente.

**T3.d — Rollout**
- Nuevo modelo como `best_jugadores_v2.pt` en la raíz del repo (no sobreescribir `best_100e.pt`).
- Apuntar con `MODEL_PATH` en `.env` al nuevo fichero.
- Mantener `best_100e.pt` como fallback durante 1-2 semanas.

**Esfuerzo:** 1-2 días de setup + tiempo de entrenamiento en Kaggle (~2-4h de GPU).
**Riesgo:** bajo-medio. Los scripts ya están listos; el riesgo es que los datasets de Roboflow no cubran suficientemente bien el ángulo específico de Dinamó — en ese caso, segunda iteración con datos propios.

---

### 🔲 Análisis de balón parado

**Objetivo:** detectar automáticamente los momentos de balón parado del partido (córners, faltas, saques de banda) y presentarlos al entrenador como una línea de tiempo navegable desde el dashboard. El cliente lo pidió explícitamente — es una feature de producto, no solo una mejora de modelo.

**Enfoque acordado: regla-based como MVP**

No se entrena ningún modelo nuevo en esta iteración. Se usan los datos que ya exporta el pipeline en `match_data.json`:
- Velocidad del balón frame a frame (`speed_over_time` del balón).
- Posición del balón en metros (`position_history` del balón).
- Número de jugadores agrupados cerca del balón.

**Lógica de detección:**

Un evento de balón parado se detecta cuando:
1. La velocidad del balón cae por debajo de un umbral (p.ej. < 1 km/h) durante más de N frames consecutivos (p.ej. 30 frames = ~1s).
2. La posición del balón está dentro del campo (descarta pérdidas de detección fuera de campo).

Clasificación del tipo (heurística por posición):
- **Córner:** balón en las 4 esquinas del campo (radio < 3m de cada esquina).
- **Saque de banda:** balón en la línea lateral (< 2m del borde).
- **Falta / balón parado en juego:** resto de posiciones.

**Dónde va el código:**
- Lógica de detección → `analytics/set_piece_detector.py` (nuevo módulo).
- Output → campo `set_pieces` en `match_data.json` (lista de eventos con `type`, `frame_start`, `frame_end`, `position_m`).
- Frontend → componente de línea de tiempo en el dashboard que permita saltar al minuto del vídeo correspondiente. **Esto requiere PR conjunto backend + dashboard.**

**Validación:**
- Correr sobre `video_OG.mp4` y revisar manualmente que los eventos detectados corresponden a balones parados reales.
- Falsos positivos esperados: jugador que controla el balón quieto durante >1s. Ajustar el umbral de frames hasta que la tasa sea aceptable.

**Esfuerzo:** 2-3 días (lógica analítica + componente dashboard).
**Riesgo:** bajo. Si la detección rule-based tiene demasiados falsos positivos, la siguiente iteración añadiría un clasificador ligero de eventos — pero eso es post-MVP.

---

## Orden de ejecución pendiente

| Orden | Tier | Motivo |
|-------|------|--------|
| 1 | **T3 — Reentrenar `best_100e.pt`** | Lanzar en Kaggle; corre en paralelo mientras se trabaja en lo demás. Sin dependencias de código. |
| 2 | **Balón parado (MVP rule-based)** | Feature nueva de alto impacto para Dinamó, sin bloqueadores técnicos. |
| 3 | **T2 — Cross-chunk ReID** | El más complejo; crítico para que las stats de partido completo (heatmaps, distancia) sean correctas a lo largo de 90 min. |

---

## Lo que este plan NO incluye (a propósito)

- Análisis táctico (formaciones, pressing, líneas defensivas) — fuera de scope.
- Tracking en tiempo real durante el partido — fuera de scope corto plazo.
- Segunda iteración de T3 con datos propios de Dinamó — solo si la primera iteración muestra fallos sistemáticos.
- Clasificación avanzada de tipo de balón parado con modelo de eventos — solo si el MVP rule-based no satisface al cliente.
