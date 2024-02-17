import mediapipe as mp
import cv2
from utils.detect_frame import *
from utils.common import *

POINT_COLOR = (255,0,255)
CONN_COLOR = (20, 180, 90)
HEIGH = 480
WIDTH = 640
SAVE_TO_MP4 = False

landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=POINT_COLOR, thickness=2, circle_radius=2)
connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=CONN_COLOR, thickness=2, circle_radius=2)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)
capture.set(3,WIDTH)
capture.set(4,HEIGH)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('./data/output.mp4', fourcc, 20.0, (WIDTH,HEIGH))

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
  while capture.isOpened():
      success, frame = capture.read()
      if not success:
          break  
      detected_image = detect_hand_frame(frame, hands)  
      if detected_image.multi_hand_landmarks:
          for hand_lms in detected_image.multi_hand_landmarks:
              draw_single_hand(frame, hand_lms)
      cv2.imshow('Webcam', frame)
      if SAVE_TO_MP4: 
        out.write(frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

capture.release()
out.release()
cv2.destroyAllWindows()

