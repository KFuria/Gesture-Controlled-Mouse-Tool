import cv2
import mediapipe as mp
from Detector import Detector, HandGest
from Controller import Controller
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class App:
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    
    def __init__(self, DEBUG):
        App.cap = cv2.VideoCapture(0)
        App.CAM_HEIGHT = App.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        App.CAM_WIDTH = App.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        
        self.DEBUG = DEBUG
        self.hands = mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands = 2)
        
        self.detector = Detector()
        self.controller = Controller()
        
    def run(self):
        while App.cap.isOpened():
            success, image = App.cap.read()
            if not success:
                continue
            
            image = cv2.flip(image, 1)
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imageRGB.flags.writeable = False
            results = self.hands.process(imageRGB)

            # If right hand is present, detect gesture, and handle controls
            if results.multi_hand_landmarks and results.multi_handedness[0].classification[0].label == 'Right':
                hand_landmarks = results.multi_hand_landmarks[0]
                self.detector.update_hand_landmarks(hand_landmarks)
                gesture = self.detector.detect()
                if gesture is not None:
                    self.controller.handle_controls(gesture, hand_landmarks)
                
                if(self.DEBUG):
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
            if(self.DEBUG):
                cv2.imshow('Frame', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                    
        App.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = App(DEBUG=True)
    app.run()