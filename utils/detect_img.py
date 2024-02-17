import mediapipe as mp
import cv2
from detect_frame import *
from common import *

IMG_PATH = "./data/w1.jpg"                                        

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.2, static_image_mode=True)

image = cv2.imread(IMG_PATH)
cv2.imshow('Original', image)

detected_hands = detect_hand_frame(image, hands)

if detected_hands.multi_hand_landmarks:
    for hand_lms in detected_hands.multi_hand_landmarks:
        draw_single_hand(image, hand_lms)
        
hpoints = get_hand_points(detected_hands)
#print(hpoints[0])
#print(len(hpoints[0]))

cv2.imshow('Dataset Maker', image)

cv2.waitKey()
cv2.destroyAllWindows