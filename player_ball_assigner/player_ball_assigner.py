import sys 
sys.path.append('../')

class PlayerBallAssigner():
    def __init__(self):
        # NOTA: Si ves que sigue parpadeando, SUBE este número a 80 o 90.
        # Depende mucho de la calidad/zoom de tu video.
        self.max_player_ball_distance = 70 
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = self.get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            # La lógica EXACTA del repositorio:
            # player_bbox es [x1, y1, x2, y2]
            # [0] es x1 (izq), [2] es x2 (der), [-1] es y2 (abajo/pies)
            
            # Distancia al pie izquierdo (x1, y2)
            distance_left = self.measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            
            # Distancia al pie derecho (x2, y2)
            distance_right = self.measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            
            # Nos quedamos con la mejor de las dos
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

    def get_center_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1+x2)/2), int((y1+y2)/2)

    def measure_distance(self, p1, p2):
        # Distancia Euclidiana (Pitágoras)
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5