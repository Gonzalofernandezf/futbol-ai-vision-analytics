# CLAUDE.md

Contexto y guía de trabajo para Claude (Code) en este repositorio.
Léelo entero antes de proponer cambios.

---

## 1. Qué es este proyecto

Plataforma de análisis de vídeo con IA para fútbol. El flujo actual es:

1. El usuario graba un partido con cámara 1080p+ (idealmente desde tribuna, pero también en posición baja).
2. Sube el vídeo a la plataforma.
3. Se procesa a posteriori: detección de jugadores, equipos, balón, velocidad, distancia y aceleración.
4. Los resultados se muestran en un dashboard HTML local servido en `localhost:5500`,
   alimentado por `demo_dashboard/match_data.json` + `demo_dashboard/demo_video.mp4`.

Entorno principal de desarrollo: Visual Studio Code + Claude Code.
Servidor local: Live Server (extensión) sobre `demo_dashboard/index.html`, o
`python -m http.server 5500` desde `demo_dashboard/`.

---

## 2. Cliente piloto y prioridades de producto

Primer cliente real: **Dinamó Guadalajara** (academia de fútbol, categorías inferiores
y semipro). Reunión inicial: 09/04/2026. Lo que validó como valioso y lo que pidió
marca la hoja de ruta corta. Siguientes clientes serán otras academias formativas, clubes de 2da o tercera categoría masculinos y/o femeninos y potencialmente reclutadores internacionales en fases posteriores.

**Prioridad ALTA (construir / pulir ya):**
- 🔥 Mapas de calor por jugador — diferenciador clave, lo más valorado.
- 🔥 Análisis de balón parado — demanda no cubierta en categorías inferiores.
- 🔥 Mejora de detección de puntos de la cancha para realizar transformación del plano (afecta directamente a mapas de calor)
- 🔥 Robustez del modelo de detección con cámara en posición BAJA (no solo tribuna).
  Afecta especialmente a la detección de bandas/laterales.

**Prioridad MEDIA:**
- Acceso por minuto/instante específico del vídeo desde el dashboard.
- Análisis combinado de posesión + posicionamiento.
- Cortes de vídeo automáticos por evento.
- Datos de aceleración expuestos de forma clara para preparadores físicos.

**Prioridad BAJA / futuro:**
- Análisis en tiempo real durante el partido (no a posteriori).
- Tags preprogramados en tiempo real durante la grabación, con clips automáticos
  de ~10 s alrededor del tag. 

**Principios de diseño que pidió el cliente:**
- Interfaz simple antes que potente. Si una feature añade complejidad visual,
  esconderla detrás de un modo avanzado.
- Todo lo que sea "un clic para ver el clip" es oro.

---

## 3. Stack y estructura

**Lenguaje:** Python 3.9+

**Frameworks de visión / ML:**
- `ultralytics==8.0.239` — YOLOv8 para detección de jugadores, balón, portero y árbitro.
  Modelo de jugadores: `best_100e.pt`. Modelo de campo (keypoints): `modelo_cancha.pt`.
- `supervision==0.19.0` — Wrappers de detección y `ByteTrack` para tracking multi-objeto.
- `opencv-python==4.8.1.78` — Lectura/escritura de vídeo, optical flow (Lucas-Kanade),
  homografía y transformación de perspectiva.
- `scikit-learn==1.3.0` — K-Means para clasificación de equipos por color de camiseta.
- `numpy==1.24.3` / `pandas==2.0.3` — Procesado numérico y exportación de datos.
- `python-dotenv==1.0.0` — Variables de entorno desde `.env`.
- `Flask==3.0.0` — Incluido como dependencia pero **no activo**; el dashboard es HTML estático.

**Frontend:** `demo_dashboard/index.html` — Tailwind CSS (CDN) + Chart.js.

**Gestor de dependencias:** `pip` / `requirements.txt`. Instalar con:
```bash
pip install -r requirements.txt
```
Opcionalmente como paquete editable:
```bash
pip install -e .
```

**Estructura del repo:**

```
/
├── Main.py                         # Orquestador principal: vídeo → JSON + vídeo anotado
├── config.py                       # ÚNICO lugar para parámetros (umbrales, FPS, rutas, etc.)
├── setup.py                        # Paquete + CLI entry point: `futbol-ai`
├── requirements.txt
├── .env.example                    # Plantilla de variables de entorno
├── best_100e.pt                    # Pesos YOLOv8 jugadores/balón
├── modelo_cancha.pt                # Pesos YOLO detección de campo (keypoints)
├── video_OG.mp4                    # Vídeo de muestra para pruebas
│
├── calibrate_pitch/
│   └── calibrate_pitch.py          # Herramienta interactiva de calibración (4 puntos)
│
├── player_detection/
│   └── Player_Detection.py         # Wrapper de inferencia YOLO
│
├── Trackers/
│   └── tracker.py                  # ByteTrack + YOLOv8 + dibujado de anotaciones
│
├── team_assigner/
│   └── team_assigner.py            # Clustering K-Means por color de camiseta
│
├── view_transformer/
│   └── view_transformer.py         # Píxeles → metros (homografía + YOLO campo)
│
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py  # Velocidad, distancia, aceleración
│
├── player_ball_assigner/
│   └── player_ball_assigner.py     # Asignación balón → jugador más cercano
│
├── camera_movement_estimator/
│   └── camera_movement_estimator.py    # Compensación movimiento de cámara (optical flow)
│
├── data_exporter/
│   └── data_exporter.py            # Exporta tracking → JSON
│
├── utils/
│   └── video_utils.py              # Helpers de lectura/escritura de vídeo
│
├── demo_dashboard/
│   ├── index.html                  # Dashboard interactivo (Tailwind + Chart.js)
│   ├── demo_video.mp4              # Último vídeo procesado (runtime, ignorado en git)
│   └── match_data.json             # Último JSON de estadísticas (runtime, ignorado en git)
│
├── stubs/                          # Caché de tracking (ignorado en git)
└── output_videos/                  # Salidas con fecha-versión (ignorado en git)
```

**Formato del JSON de salida** (`match_data.json`):
```json
{
  "match_meta": {
    "duration_seconds": 35.00,
    "fps": 29.97,
    "home_possession": 41.9,
    "away_possession": 58.1
  },
  "players": {
    "1": {
      "team": 1,
      "max_speed_kmh": 15.86,
      "total_distance_m": 4.74,
      "max_acceleration_ms2": 9.43,
      "speed_over_time": [null, null, 15.86, 2.77, ...]
    }
  }
}
```
**Nunca cambiar este schema sin actualizar el dashboard en el mismo PR.**

El schema crecerá: cuando se implementen los heatmaps (prioridad ALTA), habrá que añadir
una serie de posiciones por frame por jugador (p. ej. `positions_over_time: [[x,y], ...]`).
Cualquier extensión del schema requiere PR conjunto con el dashboard; no añadir campos
al JSON sin que el frontend los consuma en ese mismo PR.

---

## 4. Cómo correr y testear

**Instalar dependencias:**
```bash
pip install -r requirements.txt
# o como paquete:
pip install -e .
```

**Calibrar el campo (primera vez o nueva cámara):**
```bash
python calibrate_pitch/calibrate_pitch.py
```

**Procesar un vídeo:**
```bash
# Con el vídeo de muestra incluido en el repo:
python Main.py

# O configurando variables de entorno:
VIDEO_PATH=mi_partido.mp4 python Main.py

# Con el CLI instalado:
futbol-ai
```
Los parámetros se controlan desde `config.py` o vía `.env` (ver `.env.example` para la lista
completa de variables soportadas: rutas de modelo, ruta de vídeo, umbrales, device, etc.).
Usar siempre los nombres de variable definidos en `.env.example`; no inventar nombres nuevos.
El vídeo de muestra disponible para pruebas locales es `video_OG.mp4`.

**Lanzar el dashboard:**
```bash
# Opción A: Live Server en VS Code (abre demo_dashboard/index.html)
# Opción B:
cd demo_dashboard && python -m http.server 5500
# Luego abre http://localhost:5500
```

**Tests:** No hay suite de tests automatizados todavía. Antes de un PR,
la validación manual mínima es:
1. `python Main.py` con `video_OG.mp4` corre sin errores end-to-end.
2. El dashboard carga en `localhost:5500` sin errores en la consola del navegador.
3. El JSON generado tiene la estructura esperada (ver sección 3).

**Linter / formato:** No hay configuración formal todavía. Usar criterio propio
con estilo PEP 8. Si se añade linting, hacerlo en un PR dedicado y actualizar aquí.

---

## 5. Convenciones de código

- Nombres en inglés en código (variables, funciones, clases, archivos).
  Comentarios pueden ser en español.
- Funciones puras siempre que sea posible en `/speed_and_distance_estimator` (y en una futura
  carpeta `/analytics` si se separa la lógica analítica de los estimadores).
- No mezclar lógica de modelo con lógica de dashboard.
- **Todo parámetro "mágico"** (umbrales de confianza, FPS, dimensiones de campo,
  velocidades máximas, etc.) va exclusivamente a `config.py`. Nunca hardcodeado
  en módulos individuales.
- El caso "cámara baja" es de primera clase: no asumir que la perspectiva es siempre
  cenital o desde tribuna.

---

## 6. Cómo quiero trabajar con Claude

- Yo (Gonzalo) actúo como dirección de producto. Claude actúa como Tech Lead.
- Para cada tarea no trivial, antes de tocar código:
  1. Resumir el problema en 2-3 frases.
  2. Proponer 1-2 enfoques con pros/contras.
  3. Esperar a que confirme el enfoque, salvo que la tarea sea claramente trivial.
- Cambios siempre vía Pull Request, nunca directo a `main`.
- Un PR = un cambio con una intención. Nada de PRs "limpieza + feature + refactor".
- En la descripción del PR incluir: qué prioridad del cliente ataca (heatmaps,
  balón parado, tags en vivo, robustez cámara baja, etc.), qué se probó, y
  qué quedó fuera a propósito.
- Si algo del repo contradice este archivo, gana este archivo y se abre un PR
  para alinear lo demás.

---

## 7. Cosas que NO hacer

- No reescribir módulos enteros sin pedirlo.
- No añadir dependencias pesadas sin justificarlo en el PR.
- No commitear vídeos, modelos `.pt` ni datasets a git
  (`output_videos/`, `*.mp4`, `*.pt`, `stubs/` están en `.gitignore`).
- No tocar el formato de los JSON que consume el dashboard sin actualizar el
  dashboard en el mismo PR.
- No asumir que la cámara está en tribuna: el caso "cámara baja" es de primera clase.
- No hardcodear parámetros fuera de `config.py`.
- No activar Flask ni añadir un servidor backend sin discutirlo primero
  (el dashboard es estático a propósito).
