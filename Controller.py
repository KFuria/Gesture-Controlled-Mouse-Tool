import pyautogui
from Detector import HandGest
import numpy as np

class Controller:
    def __init__(self):
        self.prev_action = None
        self.curr_action = None
        self.prev_finger_pos = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.mouse_finger_tip = 12
        
    def get_mouse_coords(self, hand_landmarks):
        landmark = hand_landmarks.landmark[self.mouse_finger_tip]
        finger_pos = np.asarray([int(landmark.x * self.screen_width), int(landmark.y * self.screen_height)])
        if self.prev_finger_pos is None:
            self.prev_finger_pos = finger_pos 
        delta = finger_pos - self.prev_finger_pos
        dist_sq = np.sum(np.square(delta))
        self.prev_finger_pos = finger_pos
        
        ratio = 1
        if dist_sq <= 20:
            ratio = 0
        elif dist_sq <= 900:
            ratio = 0.08 * (np.sqrt(dist_sq))
            
        mouse_old_x, mouse_old_y = pyautogui.position()
        mouse_x, mouse_y = mouse_old_x + delta[0]*ratio, mouse_old_y + delta[1]*ratio
        return int(mouse_x), int(mouse_y)
    
    def handle_controls(self, gesture, hand_landmarks):
        if gesture == HandGest.thumb_first2:
            mouse_x, mouse_y = self.get_mouse_coords(hand_landmarks)
            pyautogui.moveTo(mouse_x, mouse_y, duration=0.1, _pause = False)
            self.curr_action = HandGest.thumb_first2
        
        elif gesture == HandGest.THUMB and self.prev_action == HandGest.FIST:
            pyautogui.press('left')
            self.curr_action = HandGest.left_arrow
        
        elif gesture == HandGest.PINKY and self.prev_action == HandGest.FIST:
            pyautogui.press('right')
            self.curr_action = HandGest.right_arrow
            
        elif gesture == HandGest.FIST:
            self.curr_action = HandGest.FIST
            self.prev_finger_pos = None
        
        elif gesture == HandGest.first2:
            self.curr_action = HandGest.first2
            self.prev_finger_pos = None
        
        elif gesture == HandGest.last4 and self.prev_action == HandGest.FIST:
            pyautogui.press('space')
            self.curr_action = HandGest.space_but
            
        elif gesture == HandGest.MID and self.prev_action == HandGest.first2:
            pyautogui.click(button='left')
            self.curr_action = HandGest.left_click
        
        elif gesture == HandGest.INDEX and self.prev_action == HandGest.first2:
            pyautogui.click(button='right')
            self.curr_action = HandGest.right_click
                  
        self.prev_action = self.curr_action