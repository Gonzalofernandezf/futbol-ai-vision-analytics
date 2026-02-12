from sklearn.cluster import KMeans
import numpy as np

"""
Team Assignment Module

Classifies players into teams based on jersey colors using K-Means clustering.
Provides per-frame team predictions and color learning from the first frame.
"""

class TeamAssigner:
    """
    Automatically assigns players to teams based on jersey color clustering.
    
    Uses K-Means to identify dominant colors and classify new players into
    one of two teams. Provides persistent team assignments across frames.
    
    Attributes:
        team_colors (dict): Stores the dominant color for each team (ID -> RGB)
        player_team_dict (dict): Caches player ID -> team ID mappings
        kmeans (KMeans): Global clustering model trained on first frame
    """
    
    def __init__(self):
        """Initialize empty team assigner, trained on first video frame."""
        self.team_colors = {}
        self.player_team_dict = {} # Dictionary to store player_id -> team_id
    
    def get_clustering_model(self, image):
        # Change the image to a 2D array
        image_2d = image.reshape(-1, 3)
        
        # We use K-means with 2 clusters (Background vs Jersey)
        kmeans = KMeans(n_clusters=1, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # 1. Take the top half
        top_half = image[0:int(image.shape[0]/2), :]
        
        # 2. Crop the sides to keep the center (chest)
        # This prevents the background grass from contaminating the color
        height, width, _ = top_half.shape
        start_x = int(width * 0.20) # Ignore left 20% of the image
        end_x = int(width * 0.80)   # Ignore right 20% of the image
        
        player_crop = top_half[:, start_x:end_x]
        
        # Get the model using the cropped image
        kmeans = self.get_clustering_model(player_crop)

        # Get the labels
        labels = kmeans.labels_

        # By using K=1, the only centroid (index 0) is the jersey color.
        player_color = kmeans.cluster_centers_[0]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        
        # 1. Collect the color of ALL players detected in the first frame
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # 2. Run a global K-Means to split those colors into 2 teams
        # Ensure there are at least 2 players to avoid an error
        if len(player_colors) < 2:
            print("⚠️ Not enough players in frame 0 to train colors.")
            # Define default colors for safety (White and Black)
            self.team_colors[1] = (255, 255, 255)
            self.team_colors[2] = (0, 0, 0)
            # Create a dummy kmeans to avoid crash
            self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            if len(player_colors) > 0:
                 self.kmeans.fit(player_colors + player_colors) # We duplicate so it doesn't fail
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        
        # Fix: cast to integers
        # OpenCV needs integers (int), not decimals (float)
        self.team_colors[1] = kmeans.cluster_centers_[0].astype(int).tolist()
        self.team_colors[2] = kmeans.cluster_centers_[1].astype(int).tolist()

    def get_player_team(self, frame, player_bbox, player_id):
        # If we already know which team this ID belongs to, we don't recalculate (saves resources)
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id] 

        player_color = self.get_player_color(frame, player_bbox)

        # Predict which team this specific color belongs to
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1 # Ajuste para que sea equipo 1 o 2

        # Provide 'persistence': if it's ID 10, it will always belong to team X
        self.player_team_dict[player_id] = team_id

        return team_id