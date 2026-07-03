# Plan — Mejora de detección y persistencia de IDs de jugadores

**Contexto:** después de fix del bug de balón + integración del nuevo modelo de balón, el detector de jugadores y la persistencia de tracking quedaban como cuello de botella principal. Este documento recoge el plan original y su estado de ejecución actualizado a 2026-07-02.

**Actualización 2026-07-03:** se incorpora la auditoría de un deep research externo (Gemini) sobre el estado del arte 2023-2025 en visión por computador para fútbol, encargado con una restricción explícita de ROI para equipo de 2 personas. Ver sección "Roadmap 2026" más abajo — incluye una corrección importante al estado de Tier 4 (team assignment), encontrada al verificar el research contra el código real.

**Regla general:** un PR por Tier. Antes/después medible con `utils/perf_monitor.py`.

---

## Estado de ejecución

| Tier | Descripción | Estado |
|------|-------------|--------|
| T0 + T5 | Quick wins de configuración + observabilidad | ✅ Completo |
| T1 | Migración ByteTrack → BoT-SORT con ReID | ✅ Completo |
| T4 | Team assignment robusto | ⚠️ Completo solo parcialmente — la descripción original no coincide con el código real. Ver corrección 2026-07-03 en la sección T4 y T7 en "Roadmap 2026" |
| — | Reentrenamiento `modelo_balon.pt` | ✅ Completo |
| — | Reentrenamiento `modelo_cancha.pt` | ✅ Completo (PCK@5px 93%, error homografía 5.5m) |
| — | Heatmaps precisos por jugador | ✅ Completo (desbloqueado por nuevo modelo_cancha) |
| T2 | Cross-chunk ReID | ✅ Completo (código). Pendiente validar contra un partido completo real por chunks en Kaggle. |
| T3 | Reentrenamiento `best_100e.pt` (cámara baja) | ✅ Completo — `best_jugadores_v2.pt` validado sobre `video_OG.mp4` (footage real de Dinamó, cámara 3-4m): mAP50 0.983, id_churn_ratio 3.24→2.46. Fallback a `best_100e.pt` por 1-2 semanas de uso real. |
| — | Análisis de balón parado | ✅ Completo (backend + dashboard). Pendiente validar detección contra `video_OG.mp4` real. |

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

### ⚠️ Tier 4 — Team assignment robusto (corrección 2026-07-03: descripción original no coincide con el código)

Descripción original de este plan (nunca verificada línea a línea contra el código hasta ahora):
- Segmentación HSV para descartar césped y fondo antes del clustering.
- Re-cluster periódico (resiste cambios de iluminación y sombras).
- Votación móvil por ID en vez de cacheo permanente.
- Portero tratado como clase separada (no contamina los clusters de equipo).

**Lo que realmente hay en el código**, verificado al auditar el pipeline contra el deep research externo (ver "Roadmap 2026" más abajo):

| Garantía descrita | Código real |
|---|---|
| Segmentación HSV para descartar césped | No existe. `team_assigner.py::get_player_color` solo recorta el 60% central del ancho y la mitad superior del bbox — un recorte geométrico fijo, no una máscara de color |
| Re-cluster periódico | No existe. `self.kmeans` se entrena una única vez en `assign_team_color`, llamado solo con el frame 0 del vídeo (`Main.py:260`), y nunca se vuelve a entrenar durante el resto del partido |
| Votación móvil por ID | Existe un voto (`Main.py:281-303`), pero es un voto **global de todo el partido calculado una sola vez al final**, no una ventana deslizante que se adapte a lo largo del partido |
| Portero como clase separada | No existe. `Trackers/tracker.py:166-167` remapea `goalkeeper → player` antes de que el team assigner lo vea; el portero entra en el mismo clustering de camiseta que el resto de jugadores |

Es decir: T4 nunca llegó a implementar el diseño robusto que este plan describía — lo que hay hoy es la versión más simple posible (crop de color + K-Means fijo desde frame 0 + voto global al cierre), con exactamente las debilidades (sombras, iluminación cambiante, portero contaminando el clustering) que el diseño original decía haber resuelto. Plan de corrección con dos opciones en el Tier T7 de "Roadmap 2026".

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

### ✅ Tier 3 — Reentrenamiento `best_100e.pt` (cámara baja)

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

**T3.c — Validación** ✅ Completa
- **Nota importante:** `video_OG.mp4` (el vídeo de muestra del repo) **es footage real de Dinamó, grabado con su cámara fija a 3-4m de altura, lateral** — es decir, exactamente el caso objetivo de este Tier, no un proxy genérico. La validación de abajo mide directamente el problema reportado por el cliente.
- mAP50 = 0.983 en el holdout del dataset fusionado (objetivo ≥0.85). Por clase vs. `best_100e.pt` (mismo holdout, normalizado sin balón): `goalkeeper` +2.1pts mAP50 / +6.0pts mAP50-95 (mejora real), `player`/`referee` prácticamente planos (ya estaban cerca del techo).
- Comparación con `debug_player_detection.py` (script nuevo, más liviano que `Main.py` — corre solo tracker+equipos, sin balón/campo/velocidad) sobre 30s de `video_OG.mp4`: `id_churn_ratio` 3.24→2.46 (pasa de incumplir el objetivo ≤3.0 a cumplirlo), long tracks (≥5s) 21→24, avg track duration 5.4s→6.0s. Inspección visual lado a lado: sin diferencias apreciables a simple vista, consistente con que la mejora real es modesta (concentrada en `goalkeeper`).
- Conclusión: mejora real pero modesta, medida en las condiciones reales de Dinamó. No hace falta grabar footage adicional para cerrar este Tier.

**T3.d — Rollout**
- ✅ `best_jugadores_v2.pt` (YOLOv11m) es el candidato — colocar en la raíz del repo (no sobreescribe `best_100e.pt`; `*.pt` sigue en `.gitignore`, no se commitea).
- Apuntar con `MODEL_PATH=best_jugadores_v2.pt` en `.env` (local y Kaggle) para usarlo. `config.py`/`.env.example` mantienen `best_100e.pt` como default — el cambio es opt-in por entorno, no global.
- Mantener `best_100e.pt` como fallback durante 1-2 semanas de uso real antes de promoverlo a default (práctica estándar de rollout, no por falta de validación — T3.c ya está cerrada).

**Esfuerzo:** 1-2 días de setup + tiempo de entrenamiento en Kaggle (~2-4h de GPU).
**Riesgo:** bajo-medio. Los scripts ya están listos; el riesgo es que los datasets de Roboflow no cubran suficientemente bien el ángulo específico de Dinamó — en ese caso, segunda iteración con datos propios.

---

## Tiers pendientes

### ✅ Tier 2 — Cross-chunk ReID

**Objetivo:** que un jugador mantenga el mismo `track_id` a lo largo de todo el partido, independientemente de en qué chunk fue procesado.

**Problema previo:** `run_chunked.py` divide el partido en chunks. Cada chunk asigna IDs locales desde cero. Además, **ya existía** un `_match_ids` en `run_chunked.py` — no partimos de cero: era un matcher ingenuo (solo posición, umbral fijo de 5m, greedy no-óptimo, sin conocer equipo). T2 lo reemplaza por uno mejor, no construye el mecanismo desde cero.

**Decisión de arquitectura:** la reconciliación de IDs sigue siendo un **paso de post-procesado** al final del merge — los chunks corren en paralelo sin coordinación entre ellos, tal como se acordó.

**Verificación técnica previa a implementar:** el diseño depende de leer embeddings de apariencia de BoT-SORT. Se confirmó contra el código fuente real de `ultralytics` (v8.4.84, la versión usada en Kaggle) que es viable: `BOTrack` guarda `curr_feat`/`smooth_feat` por track, y como el pipeline ya usa `persist=True`, el objeto tracker interno sigue vivo y accesible vía `self.model.predictor.trackers[0].tracked_stracks`/`lost_stracks` después de `model.track()`. **Advertencia:** esto es API interna/privada de `ultralytics`, no pública — una actualización futura de la librería podría romperlo en silencio. Por eso `Tracker.get_track_embeddings()` es defensiva (try/except, devuelve `{}` si falla) y todo el pipeline de cross-chunk ReID tiene fallback automático al matching antiguo por posición si los embeddings no están disponibles.

**Diseño implementado:**

1. `Trackers/tracker.py::get_track_embeddings()` — lee los embeddings de los tracks activos/recién perdidos al terminar `get_object_tracks()` de un chunk.
2. `Trackers/cross_chunk_reid.py::build_chunk_state(tracks, embeddings)` — combina esos embeddings con `team`/posición ya calculados en `tracks['players']`. Guarda `first_position_m` **y** `last_position_m` por track (el borrador original solo tenía `last_position_m` — insuficiente, porque cada track necesita servir de "extremo final" al comparar contra el chunk anterior y de "extremo inicial" al comparar contra el siguiente).
3. `Main.py` — tras la votación de equipo, llama a `build_chunk_state()` y guarda `chunk_state.pkl` en el `OUTPUT_DIR` propio del chunk (no en un `stubs/` compartido como decía el borrador — evita colisiones entre chunks corriendo en paralelo).
4. `Trackers/cross_chunk_reid.py::ChunkBridge` — Hungarian assignment (`scipy.optimize.linear_sum_assignment`, dependencia nueva) entre el final de un chunk y el inicio del siguiente, con matriz de costes: apariencia (coseno) + posición espacial, equipo como restricción dura (∞ si difiere), umbral de distancia espacial máxima y de coste total máximo para rechazar matches poco confiables. Encadena 3+ chunks correctamente (verificado con datos sintéticos: un jugador que cambia de ID local en cada chunk mantiene su ID global a lo largo de toda la cadena).
5. `run_chunked.py` — cambio quirúrgico: si todos los chunks aportaron `chunk_state.pkl`, `ChunkBridge` reemplaza a `_match_ids`; si falta alguno, cae automáticamente al matching antiguo por posición (`_match_ids`, ahora documentado como fallback). `_merge_chunks` no cambió su lógica de fusión/recorte, solo recibe un `id_map` mejor.

**Nota sobre el "coste temporal" del borrador original:** no se implementó como término explícito de la matriz de costes — el margen de `CHUNK_OVERLAP_SEC` y el `track_buffer` de BoT-SORT (`BOTSORT_TRACK_BUFFER`) ya limitan naturalmente qué tracks sobreviven hasta el borde del chunk como para ser candidatos a matching; añadir una penalización temporal explícita se dejó fuera del MVP por no aportar señal adicional clara sobre lo que ya filtra el resto del pipeline.

**Validación:**
- Probado end-to-end con datos sintéticos (`ChunkBridge` + `_merge_chunks`): con IDs locales reseteados entre chunks, el merge produce el número correcto de jugadores globales, con `position_history`/distancia acumulados bien.
- 🔲 **Pendiente:** correr sobre un partido completo real en Kaggle con `run_chunked.py` y comparar `match_unique_player_ids`/`id_churn_ratio` antes/después — no se puede probar en este entorno de desarrollo (sin GPU ni `ultralytics` instalado). Objetivo original del plan: para un partido de 90 min, idealmente ≤30 IDs únicos globales.

**Esfuerzo real:** ~1 sesión.
**Riesgo:** medio — la dependencia de API interna de `ultralytics` es el riesgo principal a mediano plazo (mitigado con fallback automático); el ajuste fino de pesos/umbrales de la matriz de costes necesita, como con T3 y balón parado, una validación con datos reales antes de confiar en los números.

---

### ✅ Análisis de balón parado

**Objetivo:** detectar automáticamente los momentos de balón parado del partido (córners, faltas, saques de banda) y presentarlos al entrenador como una línea de tiempo navegable desde el dashboard. El cliente lo pidió explícitamente — es una feature de producto, no solo una mejora de modelo.

**Corrección importante al plan original:** se asumía que el pipeline ya exportaba datos de balón (`speed_over_time`, `position_history`) en `match_data.json`. Verificado en código: **falso**. `data_exporter/data_exporter.py` solo procesaba `tracks['players']`; no había ninguna clave `"ball"` en el JSON exportado. Además, `Trackers/tracker.py::filter_ball_positions_by_speed` colapsaba los rechazos "fuera de campo" (Stage A) al mismo `{}` que una oclusión normal, sin conservar causa ni posición. Ambos se corrigieron como parte de esta implementación.

**Enfoque implementado: dos señales complementarias**

1. **Balón sale de la cancha y reaparece → córner / saque de banda / saque de meta.**
   `filter_ball_positions_by_speed` ahora también devuelve `out_of_bounds_events` (frame + posición exacta del rechazo Stage A). `analytics/set_piece_detector.py::detect_boundary_events` detecta un hueco en `position_history` del balón que contenga al menos un frame de `out_of_bounds_events` (no una oclusión normal), y clasifica por el **punto de SALIDA** — no el de reaparición como decía el borrador original. Corrección hecha durante la implementación: un córner o saque de meta no reaparece donde el balón salió (se saca hacia dentro del campo), así que el punto de salida es la señal fiable; la de reaparición no lo es.
   - Salida cerca de esquina (radio ≤5m, ajustado desde los 3m del borrador — error medio de homografía medido ~5.5m) → córner.
   - Salida cerca de línea de fondo (fuera de las esquinas) → saque de meta.
   - Salida cerca de línea lateral → saque de banda.
2. **Balón casi inmóvil sin haber salido de cancha, durante ≥5s** (`detect_stationary_events`, subido desde 1s del borrador — 1s es indistinguible de un jugador controlando el balón un instante) → falta / tiro libre.

**Dónde vive el código:**
- `config.py` — `SET_PIECE_STATIONARY_SEC`, `SET_PIECE_STATIONARY_KMH`, `SET_PIECE_BOUNDARY_MARGIN_M`, `SET_PIECE_MIN_GAP_FRAMES`.
- `Trackers/tracker.py::filter_ball_positions_by_speed` — ahora devuelve `(ball_tracks, out_of_bounds_events)`.
- `analytics/set_piece_detector.py` (nuevo, funciones puras) — `detect_boundary_events`, `detect_stationary_events`, `detect_set_pieces`.
- `Main.py` — corre la detección tras el filtrado de balón, pasa el resultado al exportador.
- `data_exporter/data_exporter.py` — nuevo `ball` (position_history/speed_over_time, mismo formato que un jugador) y `set_pieces` (con `start_sec`/`end_sec` ya calculados para que el dashboard solo tenga que hacer `currentTime = start_sec`).

**Dashboard (SP.c/d) — implementado según diseño acordado con Gonzalo:**
1. Mapa de calor y vídeo lado a lado (mitad cada uno) en vez de apilados — `src/routes/index.tsx`, sin redimensionar el resto de la página (`MatchHeader` y la barra derecha quedan igual).
2. `SetPiecesCard` nuevo (`src/components/player/SetPiecesCard.tsx`) bajo `PlayerCard` en la barra derecha, mismo estilo visual (`rounded-xl border bg-card p-4 shadow-lg`). 4 pestañas clicables (Falta/Córner/Banda/Meta), Falta como vista por defecto.
3. Click en cualquier evento listado salta el vídeo a ese segundo, reutilizando `VideoContext.seekToMinute` (el mismo mecanismo que ya usa `SpeedChart`).
- Verificado con un `match_data.json` sintético vía Playwright: layout, filtrado por pestaña y click-to-seek funcionan correctamente.

**Validación pendiente (no bloqueante, siguiente paso):**
- Correr sobre `video_OG.mp4` real y revisar manualmente que los eventos detectados corresponden a balones parados reales — la lógica se probó con datos sintéticos, no contra el vídeo real todavía.
- Ajustar `SET_PIECE_STATIONARY_SEC`/`SET_PIECE_BOUNDARY_MARGIN_M` si la tasa de falsos positivos/negativos no es aceptable.

**Límite conocido:** `run_chunked.py` no fusiona `set_pieces` entre chunks todavía (su `_merge_chunks` reconstruye el JSON campo a campo y no conoce esta clave nueva) — solo las corridas de `Main.py` sin chunking exportan `set_pieces` hoy. Fuera de alcance de esta iteración; abordar si/cuando haga falta procesar partidos completos por chunks con esta feature activa.

**Esfuerzo real:** ~1 sesión (backend + frontend juntos, dentro del rango revisado de 3-5 días).
**Riesgo:** bajo-medio — pendiente la validación contra vídeo real antes de considerarlo probado en producción.

---

## Orden de ejecución pendiente

| Orden | Tier | Motivo |
|-------|------|--------|
| ~~1~~ | ~~T3 — Reentrenar `best_100e.pt`~~ | ✅ Completo. |
| ~~2~~ | ~~Balón parado (MVP rule-based)~~ | ✅ Completo (backend + dashboard). Pendiente validar contra `video_OG.mp4` real. |
| ~~3~~ | ~~T2 — Cross-chunk ReID~~ | ✅ Completo (código). Pendiente validar con un partido completo por chunks en Kaggle. |

Los tres tiers priorizados del plan están implementados. Pendiente: validación con datos reales de los tres (Kaggle) antes de darlos por cerrados en producción.

---

## Roadmap 2026 — auditoría del deep research externo (Gemini, 2026-07-03)

**Origen:** Gonzalo encargó a Gemini un deep research sobre el estado del arte 2023-2025 en visión por computador para fútbol, con un prompt explícitamente acotado a ROI: no "qué existe en el estado del arte" sino "dado lo que ya tenemos funcionando (cancha y balón ya reentrenados con buenos resultados, cámara fija no-broadcast, GPU consumer, equipo de 2 personas), qué mejoras se justifican". El documento cubre 5 áreas — calibración de campo, tracking/ReID, team assignment, detección de balón, balón parado — y cierra con una matriz de prioridad ROI.

Lo que sigue no es una transcripción de esa matriz: es la contrastación de cada recomendación contra el código real del repo (verificado archivo por archivo, no contra lo que este plan *decía* que había — ver la corrección de T4 arriba, que es exactamente el tipo de desajuste que esta auditoría buscaba evitar en el resto del roadmap).

### Tabla resumen por área

| Área del research | Recomendación | Estado real en nuestro código (auditado 2026-07-03) | Veredicto |
|---|---|---|---|
| 1. Calibración de campo (PnLCalib) | Adoptar pipeline completo de optimización 3D puntos+líneas | `view_transformer.py` ya reentrenado (`modelo_cancha.pt`, PCK@5px 92.8%, error homográfico 5.5m), con RANSAC + fallback a última matriz válida. Recalcula la homografía completa por frame; no aprovecha que la cámara es fija (K y posición constantes durante todo el partido) | Postergar adopción completa — coincide con la propia conclusión del research. Evaluar una versión mínima aparte (ver "Ideas futuras") |
| 2. Tracking/ReID (GTATrack: Deep-EIoU + GTA-Link) | Asociación global de tracklets: grafo con splitter (DBSCAN) + connector (Hungarian) | T2 ya implementa Hungarian + apariencia (embeddings OSNet de BoT-SORT) + posición + equipo como restricción dura — pero **solo en la frontera entre chunks consecutivos** (`Trackers/cross_chunk_reid.py::ChunkBridge._match_pair`). No existe splitter ni reconciliación de tracklets fragmentados dentro de un mismo chunk. Además, `BOTSORT_TRACK_BUFFER=200` (`config.py:111`, `Trackers/botsort.yaml:5`, ~6.7s a 30fps) supera el umbral que el propio research señala como zona de "deriva fantasma" (>150 frames) | Extender T2 a asociación global — mayor impacto de las 5 áreas, esfuerzo incremental bajo porque reutiliza infraestructura ya construida |
| 3. Team assignment (SigLIP + UMAP + K-Means) | Sustituir histogramas/color HSV por embeddings contrastivos congelados | Ver corrección de T4 arriba: el código real es más frágil de lo que el plan describía (sin máscara de césped, sin re-cluster, portero contaminando el clustering) | Corregirlo es más urgente de lo que el research por sí solo sugeriría, porque no partimos del baseline robusto que creíamos tener |
| 4. Balón: interpolación + suavizado cinemático | Interpolación de huecos cortos + filtro Savitzky-Golay | La interpolación de huecos **ya existe** (`Trackers/tracker.py::interpolate_ball_positions`, lineal, `BALL_INTERP_LIMIT=10` frames). El suavizado Savitzky-Golay **no existe**: `speed_over_time` del balón se calcula como derivada frame-a-frame cruda sobre la posición ya interpolada y filtrada (`data_exporter/data_exporter.py::_export_ball`), sin ningún filtro de ruido posterior | Añadir solo el suavizado — quick win real y acotado, no duplica lo ya construido |
| 5. Balón parado (heurísticas + T-DEED opcional) | Filtro heurístico espaciotemporal como primario; T-DEED solo si sobra tiempo | Heurísticas ya implementadas (`analytics/set_piece_detector.py`), pendiente de validar contra `video_OG.mp4` real. T-DEED nunca se planteó ni se necesitó | Confirma que el enfoque ya tomado fue el correcto. No implementar T-DEED |

### Nuevos tiers propuestos

#### 🔲 T6 — Suavizado cinemático del balón (Savitzky-Golay)

**Objetivo:** reducir el ruido de `speed_over_time` del balón. Hoy se calcula como derivada frame-a-frame cruda sobre `position_transformed` (`data_exporter/data_exporter.py::_export_ball`, líneas 59-64), sin ningún filtro tras la interpolación de huecos. Esto afecta: (a) la fiabilidad de `SET_PIECE_STATIONARY_KMH`/`SET_PIECE_STATIONARY_SEC` en `analytics/set_piece_detector.py` — un pico de ruido puntual puede romper una ventana de "balón inmóvil" y perder una falta/tiro libre real; (b) la curva de velocidad del balón que ve el usuario en el dashboard; (c) indirectamente, la suavidad de la trayectoria dibujada en heatmap/minimapa.

**Qué añadir:** `scipy.signal.savgol_filter` sobre `position_history` del balón, aplicado después de `filter_ball_positions_by_speed` y antes de calcular `speed_over_time` en el exportador. Ventana 13-23 frames, orden polinomial 2 (valores de referencia del research; validar el óptimo contra `video_OG.mp4`, igual que se hizo con los demás umbrales del proyecto). No tocar `BALL_INTERP_LIMIT` — el research advierte contra alargar huecos interpolados linealmente más allá de ~3-5 frames, no contra suavizar la trayectoria ya interpolada.

**Esfuerzo:** ~1 día. Cero dependencias nuevas — `scipy>=1.10.0` ya está en `requirements.txt` desde T2.
**Riesgo:** muy bajo — post-procesado matemático puro sobre datos ya filtrados; no toca detección ni tracking.

---

#### 🔲 T7 — Team assignment: cerrar la brecha real (dos opciones, decidir antes de implementar)

Ver la tabla de corrección en la sección T4 arriba para el diagnóstico completo. Resumen: sin máscara de césped, sin re-cluster tras frame 0, voto global de todo el partido en vez de ventana deslizante, y portero contaminando el clustering de equipo porque se remapea a `player` (`Trackers/tracker.py:166-167`) antes de llegar al team assigner.

**Opción A — Arreglar dentro del paradigma actual (HSV + K-Means), esfuerzo bajo:**
- Enmascarar césped por rango HSV antes de recortar el jersey, en vez de confiar solo en el recorte geométrico centrado.
- Re-clusterizar periódicamente (cada N segundos) en vez de una sola vez en frame 0.
- Convertir el voto global de `Main.py:281-303` en una ventana deslizante real (últimos K segundos), para que el sistema se adapte si la iluminación cambia a mitad de partido.
- Excluir al portero del clustering de equipo (usar la clase original antes del remapeo) y asignarle equipo por proximidad/posición en vez de por color.
- Esfuerzo: 1-2 días. Cero dependencias nuevas.

**Opción B — Saltar directo a embeddings SigLIP/CLIP + K-Means (recomendación del research):**
- Sustituir la extracción de color (`get_player_color`) por un encoder congelado (SigLIP o CLIP), reportado como inmune a sombras duras, patrocinadores y reflectancia de lluvia — justo donde el color puro falla.
- Reutilizar la lógica de equipo/voto ya existente; solo cambia el paso de extracción de features.
- Coste según el research: <1.5GB VRAM en batch, milisegundos en CPU para K-Means sobre embeddings reducidos. Requiere una dependencia nueva y pesada (`transformers` u `open_clip` — evaluar cuál es más liviano), que hay que justificar explícitamente en el PR (regla de la sección 7 de este CLAUDE.md).
- Esfuerzo: 2-3 días, incluyendo validar que el modelo elegido corre bien offline en RTX 3070/4070 sin romper el procesamiento por lotes actual.

**Recomendación tech lead:** empezar por la Opción A. Resuelve la mayor parte del problema real (césped, portero, adaptación temporal) sin dependencias nuevas y es barata de revertir. Si tras validarla contra `video_OG.mp4` persisten fallos sistemáticos de color (uniformes muy parecidos entre sí, sombras extremas), la Opción B se justifica con evidencia concreta en mano — mismo criterio que se usó en T3 para decidir si hacía falta una segunda iteración con etiquetado propio.

**Riesgo:** bajo (Opción A) / medio (Opción B, por la dependencia nueva).

---

#### 🔲 T8 — Asociación global de tracklets (extender T2 más allá de fronteras de chunk)

**Objetivo:** generalizar `Trackers/cross_chunk_reid.py::ChunkBridge` — hoy solo reconcilia IDs en la frontera entre chunks consecutivos — a una asociación global estilo GTA-Link: tratar todos los tracklets del partido (dentro de un chunk y entre chunks) como nodos de un grafo:

1. **Splitter (pieza nueva):** detectar tracklets que mezclan dos identidades por oclusión física estrecha dentro de un mismo chunk. Hoy no existe ningún mecanismo para esto — si BoT-SORT confunde o fragmenta un track en mitad de un chunk, nada lo corrige después. El research usa DBSCAN sobre embeddings de apariencia para detectar el punto de colisión y cortar el tracklet ahí.
2. **Connector (extensión de lo ya construido):** el Hungarian matching que ya existe en `ChunkBridge._match_pair` (apariencia coseno + posición + equipo como restricción dura) generalizado para operar sobre **todos** los tracklets del partido dentro de una ventana temporal+espacial, no solo el par (fin de chunk N, inicio de chunk N+1).

**Por qué es la prioridad más alta con esfuerzo relativamente bajo:** toda la infraestructura pesada ya existe de T2 — extracción de embeddings (`Tracker.get_track_embeddings()`), matriz de costes con restricción dura de equipo, Hungarian assignment vía `scipy.optimize.linear_sum_assignment`. No es un área nueva, es la extensión natural de un Tier ya cerrado. El propio research la señala como la mejora de mayor impacto de las cinco áreas, precisamente porque reutiliza señales que un pipeline con ReID ya calcula, sin tocar los detectores online.

**Dependencia con la validación pendiente de T2:** T2 todavía tiene pendiente correr sobre un partido completo real en Kaggle vía `run_chunked.py` (ver tabla de estado al inicio del documento). Conviene bundlear T8 con esa misma validación — evita dos ciclos de Kaggle separados y permite medir `id_churn_ratio` con y sin el splitter/connector generalizado en la misma corrida.

**Relacionado — revisar `BOTSORT_TRACK_BUFFER`:** el research advierte explícitamente contra subir el buffer de persistencia del tracker online como forma de "aguantar" oclusiones largas (mantener una predicción activa >150 frames sin detección física genera "derivas fantasma": el tracker asigna identidades de jugadores activos a trayectorias lineales erráticas). Nuestro `BOTSORT_TRACK_BUFFER=200` ya está en esa zona de riesgo. Con T8 implementado, la recuperación de oclusiones largas pasa a resolverse offline (apariencia + grafo) en vez de online (Kalman + buffer largo) — tiene sentido probar a bajar el buffer (100-150) y dejar que T8 recupere lo que el tracker online ya no intenta sostener por sí solo. Validar ambos cambios juntos contra el mismo partido de Kaggle, no por separado.

**Esfuerzo:** 2-3 días (el Connector es extender código existente; el Splitter con DBSCAN es la pieza nueva).
**Riesgo:** medio — mismo riesgo ya documentado en T2 (dependencia de API interna de `ultralytics` para embeddings, ya mitigado con fallback automático). El Splitter añade una heurística nueva (umbral de DBSCAN) que necesita validación empírica, como todos los umbrales de este proyecto.

---

### Qué NO hacer (confirmado por el research + auditoría propia)

| Técnica | Por qué no | Fuente |
|---|---|---|
| PnLCalib completo (recalibración 3D punto+línea) | Nuestro error de homografía ya es 5.5m, suficiente para el caso de uso actual. Reescribir el pipeline de homografía consume semanas que no compensan una mejora marginal sobre un sistema ya considerado estable. Reevaluar solo si el cliente reporta problemas de precisión concretos trazables a la homografía | Research (defer explícito) + estado propio (T3.c y modelo_cancha ya validados) |
| T-DEED para refinar balón parado | El MVP heurístico ya implementado (`set_piece_detector.py`) reporta ~85% de precisión según el research para córners/interrupciones largas. Ya teníamos como condición propia no ir más allá "salvo que el MVP rule-based no satisfaga al cliente" (ver "NO incluye" más abajo) | Research + decisión propia preexistente |
| TrackNetV3/V4 en paralelo a YOLO para el balón | Duplicaría VRAM y tiempo de inferencia sin ROI medible sobre el suavizado cinemático post-hoc (T6) | Research |
| ReID con Swin-Transformer (SOLIDER-REID) | Sobrecarga computacional no justificada frente a OSNet (ya en uso vía BoT-SORT), mal optimizado para GPU de gama media | Research |
| SAM / YOLO-seg para team assignment | Coste computacional prohibitivo (research: <5 FPS con 22 jugadores) sin mejora medible sobre alternativas más ligeras (HSV mejorado o embeddings congelados) | Research |
| NeRF-guided calibration | Tiempos de entrenamiento por escena inviables para un equipo de 2 personas en un flujo offline rápido | Research |
| Regresión de homografía end-to-end (STN) | No generaliza fuera de distribución — el riesgo exacto de pasar de estadios profesionales a campos de academia con fondos ruidosos | Research |
| VideoMAE/SlowFast crudo para action spotting | Requiere GPUs de clase A100/H100, fuera de alcance total para RTX 3070/4070 | Research |
| Subir aún más el buffer de persistencia del tracker (Kalman) para tapar oclusiones largas | Genera derivas fantasma; la vía correcta es reconciliación offline (T8), no más buffer online | Research + auditoría propia (`BOTSORT_TRACK_BUFFER=200` ya en zona de riesgo) |

### Orden de ejecución recomendado (post-research)

| Orden | Tier | Esfuerzo | Motivo |
|---|---|---|---|
| 1 | T6 — Suavizado Savitzky-Golay del balón | ~1 día | Quick win puro, cero dependencias nuevas, mejora balón parado y dashboard a la vez |
| 2 | T7 (Opción A) — Team assignment: césped por HSV real + re-cluster periódico + voto deslizante + portero aparte | 1-2 días | Corrige una brecha real entre lo documentado y lo implementado; visible en cualquier demo con cambios de luz |
| 3 | T8 — Asociación global de tracklets + retune de `BOTSORT_TRACK_BUFFER` | 2-3 días, bundleado con la validación pendiente de T2 en Kaggle | Mayor impacto de calidad de las 5 áreas del research; reutiliza infraestructura ya construida en T2 |
| — | T7 (Opción B, SigLIP) | Solo si T7-A no resuelve casos reales de color ambiguo, con evidencia concreta | Evitar over-engineering sin evidencia, mismo criterio que T3 |
| Defer | PnLCalib completo, T-DEED, y el resto de la tabla "Qué NO hacer" | — | ROI insuficiente para 2 personas en este ciclo |

---

## Ideas futuras (sin comprometer, evaluar antes de construir)

### 🔲 Reconocimiento de dorsal (número de camiseta) como señal adicional de identidad

Surgió al cerrar T2: la debilidad real del matching por apariencia (embeddings ReID de BoT-SORT) es que, **dentro del mismo equipo**, dos jugadores con camiseta idéntica y contextura similar generan embeddings parecidos — el color de camiseta domina la señal, justo donde el equipo ya es una restricción dura y no aporta más discriminación. El dorsal, cuando es legible, es casi determinístico y resolvería exactamente ese caso.

Encajaría en dos sitios sin rediseñar lo ya construido:
- `Trackers/cross_chunk_reid.py::ChunkBridge._match_pair` — como un término más de coste (o incluso restricción dura si el número es legible con alta confianza en ambos extremos), análogo al de equipo.
- Dentro de un mismo chunk, como voto mayoritario por track (mismo patrón que `TeamAssigner` ya usa para el equipo en `Main.py`), para reducir *ID switches* por oclusión.

También tiene valor de producto más allá del tracking: si es fiable, permitiría mostrar "Jugador #10" con su dorsal real en vez de un ID de track arbitrario.

**Por qué no se aborda ahora:**
- Es un sub-problema de visión nuevo (detección/OCR de dígitos sobre camiseta), no un ajuste — necesitaría su propio modelo/dataset, con un esfuerzo comparable al reentrenamiento de T3.
- El dorsal solo es legible cuando el jugador da la espalda a cámara; en **cámara baja/lateral** (el caso prioritario de este proyecto) es razonable esperar que se vea legible con menos frecuencia que desde tribuna alta — sin validar esto contra footage real de Dinamó, no hay forma de saber si el esfuerzo se justifica.
- Añade otro modelo corriendo por frame, compitiendo por VRAM/tiempo de inferencia con los que ya hay (jugadores, balón, campo).
- No sustituye la apariencia+posición ya construidas, solo las complementaría en los frames donde el número es legible.

**Antes de comprometer esfuerzo:** revisar a ojo unos minutos de `video_OG.mp4` real y contar en cuántos frames el dorsal es efectivamente legible desde la cámara de Dinamó. Si es raro, no se justifica. Si aparece con frecuencia razonable, documentarlo como un Tier nuevo con diseño propio.

*(Nota 2026-07-03: el deep research externo auditado en "Roadmap 2026" menciona de forma tangencial un tracker, SportsSUSHI, que usa el número de dorsal como señal adicional y reporta mejor HOTA en secuencias donde es legible — confirma independientemente la misma idea documentada aquí desde antes. No cambia la conclusión: sigue sin validarse contra footage real de Dinamó, así que sigue sin comprometerse.)*

### 🔲 Homografía: explotar la invarianza de cámara fija (calibración base + refinamiento de rotación)

Surgió al auditar el research externo sobre calibración de campo (ver "Roadmap 2026" arriba, Área 1). Es una idea distinta de adoptar PnLCalib completo — eso se posterga explícitamente por ROI insuficiente. Esta es una optimización mucho más acotada sobre lo que ya existe.

Dado que la cámara de Dinamó es fija (no hay PTZ — sin paneo, tilt ni zoom durante el partido), los parámetros intrínsecos y la posición espacial de la cámara no deberían cambiar durante todo el encuentro; lo único que varía son microvibraciones o deriva de rotación. Hoy `view_transformer.py::calcular_matrices_para_video` recalcula la homografía completa por frame (con RANSAC y fallback a la última matriz válida cuando falla la detección de keypoints), sin distinguir "calibración base" de "deriva de rotación frame a frame".

Una versión mínima: acumular los primeros minutos del vídeo para calcular una calibración base robusta (ya tenemos RANSAC + filtrado por confianza para esto) y, en los frames siguientes, estimar solo el ajuste de rotación respecto a esa base en vez de recomputar la homografía entera desde cero. En teoría reduciría el error medio de homografía sin adoptar el pipeline completo de PnLCalib.

**Por qué no se compromete todavía:**
- Nuestro error actual (5.5m) ya se considera "aceptable" para este caso de uso, incluso según el propio research que evalúa el resto de áreas con un listón más exigente.
- No hay ninguna queja concreta del cliente sobre precisión de homografía que lo justifique hoy.
- Tocar `view_transformer.py` es tocar el corazón de heatmaps + balón parado + toda métrica en metros — cualquier cambio aquí exige validación exhaustiva antes de arriesgar producción, con un beneficio hoy especulativo.

**Antes de comprometer esfuerzo:** solo evaluar si aparece evidencia concreta (queja de precisión del cliente, o si T7/T8 exponen que la homografía es el cuello de botella real detrás de algún fallo de heatmap/balón parado).

---

## Lo que este plan NO incluye (a propósito)

- Análisis táctico (formaciones, pressing, líneas defensivas) — fuera de scope.
- Tracking en tiempo real durante el partido — fuera de scope corto plazo.
- Segunda iteración de T3 con etiquetado manual propio (200-300 frames) — la primera iteración ya se validó contra footage real de Dinamó y mostró mejora; solo se reconsideraría si en uso real aparecen fallos sistemáticos que este fine-tuning no resolvió.
- Clasificación avanzada de tipo de balón parado con modelo de eventos (T-DEED) — solo si el MVP rule-based no satisface al cliente. Confirmado como decisión correcta por el research externo auditado en "Roadmap 2026".
- Técnicas de visión evaluadas y descartadas explícitamente en la auditoría del deep research 2026-07-03 (PnLCalib completo, TrackNetV3/V4 en paralelo, ReID con Swin-Transformer, SAM/YOLO-seg para equipos, NeRF-guided calibration, regresión de homografía end-to-end, VideoMAE/SlowFast crudo, subir aún más el buffer de persistencia del tracker) — ver tabla "Qué NO hacer" en "Roadmap 2026" para la justificación de cada una.
