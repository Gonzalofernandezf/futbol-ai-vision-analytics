# Arquitectura y viabilidad de negocio — visión de alto nivel

Documento de referencia rápida para explicar el proyecto a alguien nuevo (socio,
inversor, cliente técnico) sin tener que leer el código. No sustituye a
[CLAUDE.md](../CLAUDE.md) (fuente de verdad operativa) ni a
[PLAN_player_tracking.md](PLAN_player_tracking.md) (roadmap técnico detallado).

---

## 1. Qué hace el producto

Plataforma de análisis de vídeo con IA para fútbol, **a posteriori** (no en
tiempo real). Un usuario graba un partido con una cámara 1080p+, sube el
vídeo, y la plataforma devuelve un dashboard con:

- Mapas de calor por jugador.
- Velocidad, distancia recorrida y aceleración por jugador (útil para
  preparadores físicos).
- Detección y clasificación de balón parado (córner, banda, saque de meta,
  falta).
- Métricas derivadas: sprints, zonas de intensidad, comparativa por mitades,
  percentiles, posición táctica estimada.

No hay competición en tiempo real ni tagging en vivo todavía — es
deliberadamente prioridad baja (ver sección 5).

---

## 2. Arquitectura técnica (alto nivel)

Dos sistemas independientes que se comunican **solo por archivos**, sin API:

```
┌─────────────────────────┐        archivos estáticos        ┌──────────────────────────┐
│   Pipeline Python        │  ───────────────────────────▶   │   Dashboard React/Vite    │
│   (visión por computador)│   match_data.json                │   (futbol-ai-dashboard/)  │
│                          │   demo_video.mp4                 │                            │
│  YOLOv8 + ByteTrack      │   processing_meta.json           │  React 19 + TanStack       │
│  + homografía + K-Means  │                                   │  Router + Recharts        │
└─────────────────────────┘                                   └──────────────────────────┘
```

**Pipeline (backend Python, `Main.py` / `run_chunked.py`):**
1. Detección de jugadores/balón/árbitro con YOLOv8 (`best_100e.pt`).
2. Tracking multi-objeto con ByteTrack (`supervision`).
3. Asignación de equipo por clustering de color de camiseta (K-Means).
4. Transformación de perspectiva píxeles→metros vía homografía, apoyada en un
   segundo modelo YOLO de keypoints de campo (`modelo_cancha.pt`), con RANSAC
   y fallback a la última matriz válida cuando la detección de puntos falla.
5. Compensación de movimiento de cámara (optical flow Lucas-Kanade).
6. Cálculo de velocidad/distancia/aceleración y detección de balón parado.
7. Exportación a JSON (schema versionado, ver `match.ts` para el contrato
   TypeScript) y depósito de artefactos en `futbol-ai-dashboard/public/`.

**Dashboard (frontend):** consume esos JSON/MP4 estáticos, no hay backend
HTTP activo (Flask está en dependencias pero deliberadamente apagado). Esto
mantiene el despliegue trivial (todo es estático) a costa de que hoy el flujo
es de un único partido/carpeta a la vez — no hay multi-tenant ni base de
datos.

**Por qué esta separación importa:** desacopla completamente el ritmo de
iteración de visión por computador (Python, pesado, GPU) del de producto/UX
(React, rápido de iterar). El contrato es el JSON — cualquier cambio de
schema requiere PR conjunto pipeline+dashboard, lo cual es una fricción
consciente para evitar romper el consumidor silenciosamente.

---

## 3. Punto fuerte / diferenciador técnico

El activo más defendible no es la detección de jugadores (YOLOv8 + ByteTrack
es relativamente estándar en el sector), sino la **cadena de calidad de
datos posicionales**: homografía robusta (RANSAC + fallback) → asignación de
equipo robusta (voto deslizante, re-cluster anclado, portero aparte) →
métricas derivadas. Los mapas de calor y el balón parado, que el cliente
piloto validó como más valiosos, dependen directamente de esa cadena, no de
la detección en sí.

---

## 4. Cliente piloto y validación de mercado

**Dinamó Guadalajara** (academia de fútbol, categorías inferiores y semipro)
es el primer cliente real, con reunión inicial ya realizada (09/04/2026).
Validó como valiosos: mapas de calor, balón parado, acceso a
minuto/instante concreto del vídeo, y datos de aceleración para preparación
física. El roadmap corto está derivado directamente de ese feedback, no de
hipótesis internas — es la señal más fuerte de viabilidad actual: hay un
cliente pagador tipo con necesidades concretas ya mapeadas al producto.

Segmento de expansión planeado: otras academias formativas, clubes de 2ª/3ª
categoría (masculino y femenino), y en fases posteriores reclutadores
internacionales (scouting).

---

## 5. Riesgos y huecos conocidos (honestos)

- **Cámara baja sin resolver.** La única prioridad ALTA 100% aspiracional:
  no hay ninguna lógica de adaptación a cámara en posición baja (vs.
  tribuna) todavía, y afecta directamente a la detección de bandas/laterales.
  Es un riesgo de producto real porque no todos los clientes (sobre todo
  categorías inferiores) graban desde tribuna.
- **Sin multi-tenant / sin backend HTTP.** El diseño actual (archivos
  estáticos, un directorio `DEMO_DIR`) es intencionadamente simple para
  validar producto rápido, pero no escala a "muchos clientes con muchos
  partidos concurrentes" sin trabajo de infraestructura no empezado.
- **Sin fusión de balón entre chunks** en `run_chunked.py` (vídeos largos
  procesados por partes) — limita el análisis de balón en partidos completos
  procesados de esa forma.
- **Sin suite de tests automatizada end-to-end** en el backend Python; la
  validación es manual (correr `Main.py` sobre el vídeo de muestra y
  verificar el JSON). Riesgo de regresión silenciosa al iterar rápido.
- **Procesamiento a posteriori únicamente.** Tiempo real y tagging en vivo
  están fuera de alcance a corto plazo (prioridad baja explícita) — esto es
  una limitación de producto conocida y aceptada, no un descuido.

---

## 6. Lectura de viabilidad de negocio

**A favor:**
- Producto ya funcional end-to-end (no es solo una demo técnica): pipeline
  completo + dashboard consumible hoy.
- Cliente piloto real con necesidades ya validadas y priorizadas, no
  especuladas.
- Coste de infraestructura bajo mientras el modelo sea "archivos estáticos +
  procesado local/batch" — no hay servidores que mantener por cliente activo
  todavía.
- Diferenciador (calidad de datos posicionales → mapas de calor + balón
  parado) alineado con lo que el mercado objetivo (formación, no élite) ya
  dijo que valora, y no requiere competir en tiempo real contra soluciones
  de clubes grandes.

**En contra / a vigilar:**
- El hueco de cámara baja es justo el escenario más probable en el segmento
  objetivo (academias, categorías inferiores, presupuestos de grabación
  bajos) — cerrarlo debería tratarse como bloqueante para escalar más allá
  del piloto, no como mejora incremental.
- El modelo "archivos estáticos, sin backend" es idóneo para validar pero es
  un techo bajo para monetizar a escala (multi-cliente, multi-partido,
  histórico) — en algún momento requerirá inversión en backend/almacenamiento
  que hoy está explícitamente fuera de alcance (`No activar Flask ni añadir
  un servidor backend Python sin discutirlo primero`).
- Un solo cliente piloto validado; la generalización a otros segmentos
  (clubes 2ª/3ª, scouting internacional) todavía no tiene evidencia directa,
  solo hipótesis de expansión.

---

*Generado como resumen de alto nivel — para detalle técnico exhaustivo ver
[CLAUDE.md](../CLAUDE.md) y [PLAN_player_tracking.md](PLAN_player_tracking.md).*
