import json
import numpy as np

# Esta clase ayuda a convertir cosas raras de NumPy a cosas normales de Python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class GameStatsExporter:
    def __init__(self, fps=24):
        self.fps = fps

    # AHORA (Añadimos los argumentos con valor 0 por defecto por si acaso):
    def export_json(self, tracks, output_path="match_data.json", home_possession=0, away_possession=0, view_transformer=None):
        # Estructura final
        export_data = {
            "match_meta": {
                "duration_seconds": 0,
                "fps": self.fps,
                "home_possession": round(home_possession, 1), 
                "away_possession": round(away_possession, 1)
            },
            "players": {}
        }

        player_stats_temp = {}
        total_frames = len(tracks['players'])
        export_data["match_meta"]["duration_seconds"] = total_frames / self.fps

        print(" 💾  Procesando datos para exportación web...")

        for frame_num, frame_players in enumerate(tracks['players']):
            for player_id, info in frame_players.items():
                
                # CORRECCIÓN CRÍTICA: Convertimos el ID a String (Texto)
                # Esto evita el error "int64" porque JSON prefiere claves en texto
                pid_str = str(player_id)

                if pid_str not in player_stats_temp:
                    player_stats_temp[pid_str] = {
                        "team": info.get('team', 0),
                        "speeds": [],
                        "distances": [],
                        "speed_history": []
                    }

                speed = info.get('speed', 0)
                distance = info.get('distance', 0)
                bbox = info.get('bbox', None) # Recuperamos la caja del jugador

                if speed is None: speed = 0
                if distance is None: distance = 0

                player_stats_temp[pid_str]["speeds"].append(speed)
                player_stats_temp[pid_str]["distances"].append(distance)

                # --- NUEVO: CÁLCULO DE POSICIÓN (X, Y) EN METROS ---
                # --- NUEVO: CÁLCULO DE POSICIÓN (X, Y) SEGURO ---
                position = None
                if bbox is not None and view_transformer is not None:
                    try:
                        import numpy as np # Importamos la herramienta matemática
                        
                        # 1. CORRECCIÓN: Convertimos la lista a Numpy Array
                        foot_position = np.array([(bbox[0] + bbox[2])/2, bbox[3]])
                        
                        # 2. Transformamos
                        position = view_transformer.transform_point(foot_position)
                        
                        # 3. Convertimos y LIMPIAMOS (Quitamos corchetes extra)
                        if position is not None:
                            # A. Convertir Numpy a Lista
                            if hasattr(position, 'tolist'):
                                position = position.tolist()
                            elif hasattr(position, '__iter__'):
                                position = list(position)
                            
                            # B. APLANAR: Si es [[x,y]], nos quedamos con [x,y]
                            # Verificamos si es una lista dentro de una lista
                            if isinstance(position, list) and len(position) > 0 and isinstance(position[0], list):
                                position = position[0]
                                
                    except Exception as e:
                        # Si falla, imprimimos aviso pero NO rompemos el programa
                        if frame_num == 0: print(f"⚠️ Warning en transformación: {e}")
                        position = None
                    
                    # Si la transformación falla (devuelve None), guardamos None
                    if position is not None:
                        # Convertimos numpy array a lista normal [x, y]
                        position = position.tolist() if hasattr(position, 'tolist') else list(position)
                
                # Guardamos la posición (o null si no se pudo calcular)
                if "positions" not in player_stats_temp[pid_str]:
                    player_stats_temp[pid_str]["positions"] = []
                player_stats_temp[pid_str]["positions"].append(position)

                if frame_num % self.fps == 0:
                    player_stats_temp[pid_str]["speed_history"].append(round(speed, 2))

        # Calcular métricas finales
        # 2. Calcular métricas finales y FILTRAR FANTASMAS
        # REEMPLAZA LO QUE TIENES DENTRO DEL BUCLE POR ESTO:
        for player_id, stats in player_stats_temp.items():
            
            # FILTRO A: ¿Aparece muy poco tiempo? (Ruido / Fantasma)
            # FIX CTO: Bajamos de 1.5s a 0.5s para no perder jugadores reales con tracks cortados
            speeds = stats["speeds"]
            if len(speeds) < (self.fps * 0.5): continue
            
            # FILTRO B: ¿Velocidad imposible? (Teletransportación)
            # Filtramos picos absurdos mayores a 45km/h para el cálculo
            # 1. MANTENEMOS LOS FILTROS (Esto no cambia)
            valid_speeds = [s for s in speeds if s < 45]
            max_speed = max(valid_speeds) if valid_speeds else 0
            
            # FIX CTO: Subimos tolerancia a 40km/h (algunos glitches generan picos altos)
            if max_speed > 40: continue

            total_distance = max(stats["distances"]) if stats["distances"] else 0
            # FIX CTO: Permitir porteros que se mueven poco (bajamos de 5m a 1m)
            if total_distance < 1: continue

            # 2. NUEVO: CÁLCULO DE ACELERACIÓN (Física)
            max_accel = 0
            # Necesitamos un mínimo de datos para calcular cambios
            if len(speeds) > 10: 
                # Convertimos de km/h a m/s (dividiendo por 3.6)
                speeds_ms = [s / 3.6 for s in speeds]
                
                # Usamos una "ventana" de medio segundo para suavizar el cálculo
                window = int(self.fps / 2)
                if window < 1: window = 1
                
                accels = []
                # Recorremos la lista comparando velocidad actual vs velocidad hace 0.5s
                for i in range(window, len(speeds_ms)):
                    v_final = speeds_ms[i]
                    v_inicial = speeds_ms[i - window]
                    time_delta = window / self.fps
                    
                    # Fórmula física: a = (vf - vi) / t
                    a = (v_final - v_inicial) / time_delta
                    accels.append(a)
                
                if accels:
                    max_accel = max(accels)

            # 3. GUARDAR (Añadimos el campo nuevo al diccionario)
            export_data["players"][player_id] = {
                "team": int(stats["team"]),
                "max_speed_kmh": round(float(max_speed), 2),
                "total_distance_m": round(float(total_distance), 2),
                "max_acceleration_ms2": round(float(max_accel), 2), # <--- ¡DATO NUEVO!
                "speed_over_time": stats["speed_history"],
                "position_history":stats["positions"]
            }

        # Guardar archivo usando el codificador especial que definimos arriba
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4, cls=NumpyEncoder)
        
        print(f" ✅  Datos exportados exitosamente a: {output_path}")