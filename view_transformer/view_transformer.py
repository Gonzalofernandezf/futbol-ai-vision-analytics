import numpy as np 
import cv2 

"""
Perspective Transformation Module

Converts 2D pixel coordinates from video to real-world meter coordinates
on the football field using homography transformation.
Calibration is done via interactive tool (calibrate_pitch.py).
"""

class ViewTransformer():
    """
    Transforms video coordinates to real-world field coordinates.
    
    Uses a 4-point perspective transformation (homography) to map video pixels
    to real meters on the pitch. The transformation is calibrated using the
    penalty box corners which have known FIFA dimensions.
    
    Attributes:
        pixel_vertices (np.array): 4 corner points in video (source)
        target_vertices (np.array): 4 corner points in real meters (target)
        H (np.array): Homography matrix for perspective transformation
    """
    
    def __init__(self):
        # REAL FIELD WIDTH (in meters)
        # A standard field is 68 meters wide.
        # From the sideline (x=0) to the other sideline (x=68).
        court_width = 68
        court_length = 23.32 # Length of penalty box segment (FIFA standard)

        # 1. POINTS IN THE VIDEO (SOURCE)
        # You need to find 4 fixed points in your video that form a rectangle in real life (large right area)
        self.pixel_vertices = np.array([[922, 346], [1212, 328], [1754, 510], [1367, 552]]).astype(np.float32)

        # 2. REAL POINTS (LARGE RIGHT AREA - FIFA)
        self.target_vertices = np.array([
            [88.5, 13.84],  # Pico Arriba
            [105,  13.84],  # Fondo Arriba
            [105,  54.16],  # Fondo Abajo
            [88.5, 54.16]   # Pico Abajo
        ]).astype(np.float32)

        # 3. Compute the Homography Matrix (the magic formula)
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        # Convert a point (x, y) from pixels to meters
        p = (int(point[0]), int(point[1]))
        
        # --- REMOVE POLYGON RESTRICTION ---
        # If you limit this, you remove any player who leaves the calibrated area.
        #is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        #if not is_inside: 
        #    return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        # Iterate over all objects (ball, players) and add their position in meters
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted'] # Usamos la posición de los pies (ajustada)
                    
                    # Convert to numpy for the function
                    position = np.array(position)
                    
                    # Calculate position in meters
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        # squeeze() removes extra useless dimensions
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed