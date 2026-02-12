from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # Diccionario para guardar ID_jugador -> ID_equipo
    
    def get_clustering_model(self, image):
        # Cambiamos la imagen a 2D array
        image_2d = image.reshape(-1, 3)
        
        # Usamos K-means con 2 clusters (Fondo vs Camiseta)
        kmeans = KMeans(n_clusters=1, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # 1. Tomamos la mitad superior
        top_half = image[0:int(image.shape[0]/2), :]
        
        # 2. Recortamos los laterales para quedarnos con el centro (pecho)
        # Esto evita que el césped de fondo contamine el color
        height, width, _ = top_half.shape
        start_x = int(width * 0.20) # Ignoramos el 20% izquierdo
        end_x = int(width * 0.80)   # Ignoramos el 20% derecho
        
        player_crop = top_half[:, start_x:end_x]
        
        # Obtenemos el modelo usando la imagen recortada
        kmeans = self.get_clustering_model(player_crop)

        # Obtenemos las etiquetas
        labels = kmeans.labels_

        # Al usar K=1, el único centroide que existe (índice 0) es el color de la camiseta.
        player_color = kmeans.cluster_centers_[0]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        
        # 1. Recolectamos el color de TODOS los jugadores detectados en el primer frame
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # 2. Hacemos un K-Means global para dividir esos colores en 2 Equipos
        # Aseguramos que haya al menos 2 jugadores para evitar error
        if len(player_colors) < 2:
            print("⚠️ No hay suficientes jugadores en el frame 0 para entrenar colores.")
            # Definimos colores default por seguridad (Blanco y Negro)
            self.team_colors[1] = (255, 255, 255)
            self.team_colors[2] = (0, 0, 0)
            # Creamos un kmeans dummy para evitar crash
            self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            if len(player_colors) > 0:
                 self.kmeans.fit(player_colors + player_colors) # Duplicamos para que no falle
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        
        # --- CORRECCIÓN CRÍTICA: CAST A ENTEROS ---
        # OpenCV necesita enteros (int), no decimales (float)
        self.team_colors[1] = kmeans.cluster_centers_[0].astype(int).tolist()
        self.team_colors[2] = kmeans.cluster_centers_[1].astype(int).tolist()

    def get_player_team(self, frame, player_bbox, player_id):
        # Si ya sabemos de qué equipo es este ID, no recalculamos (ahorro de recursos)
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id] 

        player_color = self.get_player_color(frame, player_bbox)

        # Predecimos a qué equipo pertenece este color específico
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1 # Ajuste para que sea equipo 1 o 2

        # Damos "persistencia": si es el ID 10, siempre será del equipo X
        self.player_team_dict[player_id] = team_id

        return team_id