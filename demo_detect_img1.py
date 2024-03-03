import cv2
import mediapipe as mp
from utils.common import *


MAX_HANDS = 2
MODEL_PATH = './task/hand_landmarker.task'
IMG_PATH = "./data/img1.jpg"


img = cv2.imread(IMG_PATH)
cv2.imshow('original', img)
#(height, width, dim) = img.shape[:3]


# STEP 2: Create an HandLandmarker object.
base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=MAX_HANDS)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMG_PATH)

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)
# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow('Hand',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


#points = get_hand_points(detection_result)
#print(points)
#print(len(points))


points = get_world_points(detection_result)
#print(points)
#print(len(points))

#plot_hand_points(points)
ext_points = interpolate_fingers(points)
plot_points(ext_points)

cv2.waitKey()
cv2.destroyAllWindows