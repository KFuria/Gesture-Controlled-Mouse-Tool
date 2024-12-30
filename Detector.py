from enum import IntEnum
import numpy as np

class HandGest(IntEnum):
    '''
    Enum to map finger states to binary numbers
    '''
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    INDEX = 8
    THUMB = 16    
    PALM = 31
    
    first2 = 12
    last4 = 15
    thumb_first2 = 28
    
    # Mappings for action gestures
    space_but = 35
    left_arrow = 36
    right_arrow = 37
    left_click = 38
    right_click = 39
    

class Detector:
    """
    Convert Mediapipe Landmarks to hand gestures
    """
    def __init__(self):
        self.hand_landmarks = None
        self.current_gest = None
        self.prev_gest = None
        self.res_gest = None
        self.frame_count = 0

    def update_hand_landmarks(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks
    
    def palm_facing_camera(self):
        '''
        To detect palm direction.
        True if palm is facing the camera, else false.
        Using landmarks 0,5,17 to form two vectors on the face of the palm, 
        perform cross product to get a third vector perpendicular to two vectors,
        Check x,y,z of perpendicular vector to see if palm is facing camera
        '''
        landmark = self.hand_landmarks.landmark
        points = np.asarray([[landmark[idx].x, landmark[idx].y, landmark[idx].z] for idx in [0, 5, 17]])
        normal_vector = np.cross(points[2] - points[0], points[1] - points[2])
        normal_vector /= np.linalg.norm(normal_vector)
        return (normal_vector[2] < -0.7 and normal_vector[1] > 0 and normal_vector[1] < 0.4)
        
    def get_finger_dir(self):
        if self.hand_landmarks == None:
            return
        
        fingers = 0
        # detect thumb state
        if self.hand_landmarks.landmark[4].x < self.hand_landmarks.landmark[2].x:
            fingers = fingers | 1
        
        # detect finger states
        finger_points = [[8,6], [12,10], [16,14], [20,18]]
        for point in finger_points:
            fingers = fingers << 1
            if self.hand_landmarks.landmark[point[0]].y < self.hand_landmarks.landmark[point[1]].y:
                fingers = fingers | 1
        return fingers
        
    def dist(self, points):
        landmark = self.hand_landmarks.landmark
        coords = np.asarray([[landmark[idx].x, landmark[idx].y] for idx in points])
        return np.sqrt(np.sum(np.square(coords[0]-coords[1])))            
            
    def detect(self):
        if self.hand_landmarks == None:
            return None
        
        self.current_gest = None
        if self.palm_facing_camera():
            fingers = self.get_finger_dir()
            self.current_gest = fingers
                
        if self.current_gest == self.prev_gest:
            self.frame_count += 1
        else:
            self.frame_count = 0
        
        self.prev_gest = self.current_gest  
        if self.frame_count > 4:
            self.res_gest = self.current_gest

        return self.res_gest
        