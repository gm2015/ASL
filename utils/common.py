from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
POINT_COLOR = (255,0,255)
CONN_COLOR = (20, 180, 90)
POINTS_PER_JOINT = 10
NUM_POINTS = 21


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)
  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]
    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())
    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN
    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
  return annotated_image


def draw_single_hand(image, hand_lms):
  landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=POINT_COLOR, thickness=2, circle_radius=2)
  connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=CONN_COLOR, thickness=2, circle_radius=2)
  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands   
  mp_drawing.draw_landmarks(image, hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec,
                            connection_drawing_spec
                            )


def get_hand_points(detection_result):
  return get_points(detection_result.hand_landmarks)


def get_world_points(detection_result):
  return get_points(detection_result.hand_world_landmarks)


def get_points(hand_landmarks_list):
  points = list()
  if hand_landmarks_list:
    for hand_landmarks in hand_landmarks_list:
        one_hand = list()
        for i in range(NUM_POINTS):
            x = hand_landmarks[i].x
            y = hand_landmarks[i].y
            z = hand_landmarks[i].z
            one_hand.append([x, y, z])
        points.append(one_hand)
  return points


def draw_hand_points(mhpoints, fig_title = 'Points'):
  height = 600
  width = 600
  mp_hands = mp.solutions.hands 
  for points in mhpoints:
    h = 1
    # Plot the Hand
    img = np.zeros([height, width, 3], dtype=np.uint8)
    img.fill(255)
    for hc in mp_hands.HAND_CONNECTIONS:
      x0 = int((points[hc[0]][0]) * width)
      y0 = int((points[hc[0]][1]) * height)
      x1 = int((points[hc[1]][0]) * width)
      y1 = int((points[hc[1]][1]) * height)          
      cv2.line(img, (x0, y0), (x1,y1), color=(255, 0, 255), thickness=4)
    for p in points:
      cv2.circle(img, (int(p[0] * width), int(p[1] * height)), radius=5, color=(255, 20, 80), thickness=-1)
    fig_title = fig_title + str(h) 
    h = h + 1
    cv2.imshow(fig_title, img)


def plot_hand_points(mhpoints):
  mp_hands = mp.solutions.hands 
  fig = plt.figure()
  k=1
  for points in mhpoints:
    #ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(1,len(mhpoints),k)
    print(k)
    for hc in mp_hands.HAND_CONNECTIONS:
      x0, y0, z0 = points[hc[0]]
      x1, y1, z1 = points[hc[1]]
      #ax.plot([x0, x1], [y0,y1], [z0,z1])
      ax.plot([x0, x1], [y0,y1],'bo-')
    ax.axis('equal')
    plt.gca().invert_yaxis()
    k=k+1
  plt.show()


def plot_points(mh_ext_points):
  fig = plt.figure()
  #ax = fig.add_subplot(projection='3d')
  k = 1
  for ext_points in mh_ext_points:
    ax = fig.add_subplot(1, len(mh_ext_points),k)
    for i in range(len(ext_points)-1):
      x0, y0, z0 = ext_points[i]
      x1, y1, z1 = ext_points[i+1]
      #ax.plot([x0, x1], [y0,y1], [z0,z1],'bo')
      ax.plot([x0, x1], [y0,y1], 'bo')
    ax.axis('equal')
    plt.gca().invert_yaxis()
    k = k + 1
  plt.show()


def interpolate_fingers(mhpoints):
  mp_hands = mp.solutions.hands 
  mh_inter_points = list()
  for points in mhpoints:
    inter_points = list()
    for hc in mp_hands.HAND_CONNECTIONS:
        x1, y1, z1 = points[hc[0]]
        x2, y2, z2 = points[hc[1]]
        step_size = 1.0 / POINTS_PER_JOINT
        for i in range(POINTS_PER_JOINT):
            x_interpolated = x1 + (x2 - x1) * step_size * i
            y_interpolated = y1 + (y2 - y1) * step_size * i
            z_interpolated = z1 + (z2 - z1) * step_size * i
            inter_points.append([x_interpolated, y_interpolated, z_interpolated])
    mh_inter_points.append(inter_points)
  return mh_inter_points


def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr