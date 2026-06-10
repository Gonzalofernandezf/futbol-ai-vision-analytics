# Plan — Mejora de detección y persistencia de IDs de jugadores

**Contexto:** después de fix del bug de balón + integración del nuevo modelo de balón, el detector de jugadores y la persistencia de tracking quedan como cuello de botella principal. Hoy vemos miles de IDs por jugador real.

**Pre-requisito:** este plan se ejecuta **después** de:
- Modelo de balón entrenado e integrado (`modelo_balon.pt` cargado en pipeline aparte).
- Confirmación de que `YOLO_BALL_CONF`/`YOLO_BALL_IOU` ya se usan SOLO para el modelo de balón, no para el detector principal.

**Regla general:** un PR por Tier. Antes/después medible con `utils/perf_monitor.py`.

---

## Tier 0 — Quick wins de configuración (PR-T0)

**Objetivo:** corregir errores de configuración del detector principal sin tocar arquitectura.
**Esfuerzo:** ~1h código + 1 corrida full match para validar.
**Riesgo:** bajo.

### Cambios

**1. `Trackers/tracker.py` — usar thresholds del modelo principal, no del balón**

En `detect_frames()` reemplazar:
```python
conf=_cfg.YOLO_BALL_CONF,
iou=_cfg.YOLO_BALL_IOU,
por:

conf=_cfg.YOLO_CONF,
iou=_cfg.YOLO_IOU,
imgsz=_cfg.YOLO_IMGSZ,
agnostic_nms=True,
Justificación:

YOLO_BALL_CONF=0.35 filtra jugadores con confianza media (ocluidos, laterales). Genera muertes de track → IDs nuevos al reaparecer.
imgsz no estaba pasándose → Ultralytics cae a 640 por defecto cuando el modelo se entrenó a 1280. Pérdida de recall en jugadores pequeños.
agnostic_nms=True para que jugador y portero (mismo objeto) no se dupliquen en la NMS antes del remapeo de clase.
2. Trackers/tracker.py — relajar ByteTrack

self.tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=200,                  # ~7s a 30fps (antes 100 = ~3.3s)
    minimum_matching_threshold=0.7          # antes 0.8
)
Justificación: oclusiones reales en área duran 5-10s. Matching de 0.8 es muy estricto para movimientos laterales con cámara baja.

3. config.py — exponer los nuevos parámetros

Añadir (con valores por defecto razonables):

BYTETRACK_LOST_BUFFER     = int  (os.getenv("BYTETRACK_LOST_BUFFER",     "200"))
BYTETRACK_MATCH_THRESHOLD = float(os.getenv("BYTETRACK_MATCH_THRESHOLD", "0.7"))
BYTETRACK_ACTIVATION      = float(os.getenv("BYTETRACK_ACTIVATION",      "0.25"))
YOLO_AGNOSTIC_NMS         = os.getenv("YOLO_AGNOSTIC_NMS", "true").lower() == "true"
Y en tracker.py leer desde _cfg en vez de hardcoded.

Métricas a registrar en utils/perf_monitor.py
Añadir al run_sanity_checks (o a una función nueva tracking_metrics):

num_unique_player_ids total
num_long_tracks (tracks con duración ≥ 5s)
avg_track_duration_seconds
id_churn_ratio = unique_ids / long_tracks → debería tender a 1.0
Sin estas métricas no se puede validar nada.

Criterio de éxito Tier 0
En un chunk de 4 min con video_OG.mp4:

num_long_tracks ∈ [14, 30]
id_churn_ratio ≤ 3.0 (antes probablemente >10)
Sin nuevos warnings en sanity checks.
Tier 1 — Cambio de tracker con ReID (PR-T1)
Objetivo: introducir embedding de apariencia para que oclusiones largas no rompan IDs.
Esfuerzo: 1-2 días.
Riesgo: medio — cambia la API de tracking.

Decisión técnica
Migrar de supervision.ByteTrack (solo IoU + Kalman) a BoT-SORT con ReID vía ultralytics.YOLO.track(). Ultralytics lo soporta nativo con archivo botsort.yaml.

Ventajas:

ReID con OSNet (apariencia visual) — resuelve cruces entre jugadores del mismo equipo.
Integrado en Ultralytics, no hay que mantener wrapper propio.
Persistencia de IDs entre llamadas con persist=True.
Cambios
1. Crear Trackers/botsort.yaml (config del tracker):

tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 200
match_thresh: 0.75
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: True
model: auto                  # usa OSNet por defecto
2. Reescribir Tracker.detect_frames → Tracker.track_frames

def track_frames(self, frames, persist=True):
    results = []
    for i in range(0, len(frames), _cfg.YOLO_BATCH_SIZE_TRACKER):
        batch = frames[i:i+_cfg.YOLO_BATCH_SIZE_TRACKER]
        batch_results = self.model.track(
            batch,
            persist=persist,                       # mantiene IDs entre batches
            tracker="Trackers/botsort.yaml",
            conf=_cfg.YOLO_CONF,
            iou=_cfg.YOLO_IOU,
            imgsz=_cfg.YOLO_IMGSZ,
            device=self.device,
            half=_cfg.YOLO_HALF,
            verbose=False,
            agnostic_nms=_cfg.YOLO_AGNOSTIC_NMS,
        )
        results += batch_results
    return results
3. Adaptar get_object_tracks

Ya no se llama a self.tracker.update_with_detections(...). Los IDs vienen del Boxes.id que devuelve model.track().

for frame_num, detection in enumerate(detections):
    boxes   = detection.boxes
    if boxes.id is None:
        # frame sin tracks
        tracks["players"].append({}); tracks["referees"].append({}); tracks["ball"].append({})
        continue
    xyxy    = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs   = boxes.conf.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)
    # ... resto del loop usa track_ids directamente
4. Eliminar import supervision as sv del tracker.

Validación
num_long_tracks ∈ [14, 26] (más cercano al real).
id_churn_ratio ≤ 1.5.
Visualmente: dos jugadores del mismo equipo cruzándose mantienen sus IDs.
Riesgos a vigilar
BoT-SORT con ReID es ~20-30% más lento que ByteTrack puro. Medir el impacto en perf_monitor — si > 30%, evaluar bajar imgsz del ReID o desactivar gmc_method.
Si model.track() no acepta listas (batch) — caer a loop frame a frame con persist=True.
Tier 2 — Re-identificación cross-chunk (PR-T2)
Objetivo: que un jugador mantenga el mismo track_id entre chunks del mismo partido.
Esfuerzo: 2-3 días.
Riesgo: medio.
Pre-requisito: Tier 1 completo (necesitamos embeddings de apariencia).

Diseño
Al final de cada chunk, persistir por cada track_id:

chunk_state = {
    "track_id_local": int,
    "team_id": int,
    "last_embedding": np.ndarray,      # 512-d OSNet feature
    "last_position_transformed": (x, y),
    "last_frame_global": int,
    "track_duration_frames": int,
}
Guardado en stubs/track_state_chunk{N}.pkl.

Al inicio del siguiente chunk:

Correr Tier 1 normal — obtenemos IDs locales del chunk.
Para cada nuevo ID local, calcular su embedding al primer aparecer.
Matching contra el chunk_state del chunk anterior usando:
Coste de apariencia: distancia coseno entre embeddings.
Coste espacial: distancia euclídea en metros (transformed).
Coste temporal: segundos transcurridos (penaliza match si > 30s).
Coste de equipo: ∞ si team_id distinto.
Hungarian assignment (scipy.optimize.linear_sum_assignment).
Renombrar IDs locales → IDs globales del partido.
Nuevo módulo
Trackers/
  cross_chunk_reid.py        # nueva: ChunkBridge class
Interface:

bridge = ChunkBridge()
for chunk_idx, frames in enumerate(chunks):
    tracks = tracker.get_object_tracks(frames)
    tracks = bridge.relabel_ids(tracks, chunk_idx)  # ← magia
    bridge.persist_chunk_end(tracks)
Validación
En run_chunked.py agregar match_unique_player_ids al RUN SUMMARY.
Para un partido completo: idealmente ≤ 30 IDs únicos globales.
Tier 3 — Detector mejorado (PR-T3)
Objetivo: subir la calidad base del detector de jugadores (especialmente cámara baja).
Esfuerzo: 1-2 semanas con GPU.
Riesgo: alto (requiere data + entrenamiento + validación), pero alto impacto.

Sub-tareas
T3.a — Dataset. Combinar:

Roboflow Universe → datasets de fútbol con cámara lateral / nivel del campo.
SoccerNet (subset etiquetado).
1-2 partidos de Dinamó etiquetados a mano (priorizar diversidad de ángulos).
T3.b — Entrenamiento. Mismo flujo que el modelo de balón en Kaggle. Recomendado: YOLOv11m, imgsz=1280, 80-100 epochs, augmentation agresivo de geometría (perspectiva, rotación).

T3.c — Validación. mAP50 ≥ 0.85 en holdout con cámara baja. Comparar contra best_100e.pt actual en sanity checks con vídeo real.

T3.d — Rollout. Nuevo path MODEL_PATH configurable. Mantener best_100e.pt como fallback durante 1-2 semanas.

Out of scope de este PR
No tocar el modelo de balón (ya entrenado aparte).
No tocar el modelo de campo.

Tier 4 — Team assignment robusto (PR-T4)
Objetivo: que los colores de equipo no fluctúen y no se cacheen para siempre.
Esfuerzo: 3-5 días.
Riesgo: bajo-medio.

Cambios en team_assigner/team_assigner.py
Segmentación de jugador antes del color extraction.
Opción A (ligera): usar la mask del bbox + umbral en HSV para descartar verde (césped) y blanco/grises (líneas, fondo).
Opción B (mejor): un modelo de segmentación ligero (YOLOv11n-seg) para obtener la mask del jugador.
Re-cluster periódico. Hoy el cluster se entrena en frame 0. Re-entrenar cada N segundos para resistir cambios de iluminación / sombras.
Eliminar cacheo permanente. Hoy player_team_dict[player_id] se escribe una vez y nunca se actualiza. Cambiar a votación móvil: mantener histograma de votos por ID, devolver la moda.
Goalkeeper detection. Hoy se remappea a player antes del color clustering, así que el portero contamina los clusters. Tratarlo como clase separada en team_assigner (un cluster propio).
Validación
Métrica en perf_monitor:

team_flip_rate = % de IDs que cambian de equipo durante su track. Debería ser <2%.
Tier 5 — Observabilidad (transversal, hacer en paralelo a T0)
Objetivo: que cualquier futuro cambio sea medible sin ejecutar a ojo.

Añadir a utils/perf_monitor.py
def tracking_metrics(tracks, fps):
    """Calcula métricas de tracking sobre tracks['players']."""
    track_durations = {}                    # {track_id: count_frames}
    track_teams     = {}                    # {track_id: [team_ids ...]}
    for frame in tracks["players"]:
        for tid, info in frame.items():
            track_durations[tid] = track_durations.get(tid, 0) + 1
            if "team" in info:
                track_teams.setdefault(tid, []).append(info["team"])

    long_tracks = [tid for tid, d in track_durations.items() if d >= fps * 5]
    flips = sum(1 for teams in track_teams.values() if len(set(teams)) > 1)

    return {
        "num_unique_player_ids":  len(track_durations),
        "num_long_tracks":        len(long_tracks),
        "avg_track_duration_s":   sum(track_durations.values()) / max(len(track_durations), 1) / fps,
        "id_churn_ratio":         len(track_durations) / max(len(long_tracks), 1),
        "team_flip_rate":         flips / max(len(track_teams), 1),
    }
Imprimir las métricas al final del run, comparables entre PRs.

Orden de ejecución sugerido
PR-T5 (observability) + PR-T0 (quick wins) — en el mismo PR. Sin T5 no podés validar T0.
PR-T1 (BoT-SORT + ReID) — el cambio de mayor impacto/esfuerzo.
PR-T4 (team assigner) — independiente de T2/T3, fácil de meter en paralelo.
PR-T2 (cross-chunk ReID) — requiere T1.
PR-T3 (modelo nuevo) — el más caro, después de Madrid si no llega el tiempo.
Cómo arrancar desde Claude Code
Al volver al repo, decir literalmente:

"Ejecutá el Tier 0 + Tier 5 del plan en docs/PLAN_player_tracking.md. Hacé un PR con los cambios, validá end-to-end con video_OG.mp4, y reportá métricas antes/después."

Y después, tier por tier, lo mismo.

Lo que este plan NO incluye (a propósito)
Análisis táctico (formaciones, pressing, etc.) — fuera de scope.
Heatmaps — ya están en el roadmap separado.
Mejoras del modelo de campo / homografía — separate concern.
Tracking en tiempo real (live) — fuera de scope corto plazo.
---
¿Querés que también te prepare un README corto para el PR de balón (cuando termine de entrenar) explicando dónde colocar `modelo_balon.pt` y qué celda del notebook descargar?