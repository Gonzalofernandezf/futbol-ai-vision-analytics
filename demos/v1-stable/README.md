# Demo v1 - Stable

Snapshot estable del 18 Abril 2026 para presentaciones.

## Archivos
- `index.html` — Dashboard completo (standalone, no requiere servidor)
- `demo_video.json` — Datos de tracking del vídeo de 30s analizado
- `match_data.json` — Estadísticas y métricas del partido

## Limitaciones conocidas
- Heatmap con ~60% null rate por fallo en detección de keypoints de cancha (`modelo_cancha.pt`)
- El modelo pierde la cancha a partir del segundo 18 cuando la cámara panea
- **Fix en progreso:** reentrenamiento con datasets de Roboflow (semana 21-25 Abril)

## Cómo usar
Abrir `index.html` directamente en el navegador. No requiere conexión ni servidor.

## Vídeo analizado
El vídeo original de 30s no está en el repo por tamaño. Guardarlo localmente o en Drive
junto a este snapshot como `video_demo_v1.mp4`.
