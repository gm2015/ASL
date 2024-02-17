import cv2

def detect_hand_frame(image, hands):   
    detected_image = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))    
    return detected_image