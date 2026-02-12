import sys 
sys.path.append('../')

"""
Ball Possession Assignment Module

Assigns the ball to the closest player in each frame by measuring
distance from player feet to ball centroid.
"""

class PlayerBallAssigner():
    """
    Determines which player possesses the ball in each frame.
    
    Uses Euclidean distance from player feet (bottom center of bounding box)
    to ball position. The closest player within a threshold distance is
    assigned possession.
    
    Attributes:
        max_player_ball_distance (int): Maximum distance (pixels) to assign possession
                                       tune based on video quality/zoom
    """
    
    def __init__(self):
        # NOTE: If it keeps flickering, RAISE this number to 80 or 90.
        # Depends a lot on the quality/zoom of your video.
        self.max_player_ball_distance = 70 
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = self.get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            # The EXACT logic of the repository:
            # player_bbox is [x1, y1, x2, y2]
            # [0] is x1 (left), [2] is x2 (right), [-1] is y2 (bottom/feet)
            
            # Distance to left foot (x1, y2)
            distance_left = self.measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            
            # Distance to right foot (x2, y2)
            distance_right = self.measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            
            # Keep the best of the two
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
        # Euclidean distance (Pythagoras)
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5