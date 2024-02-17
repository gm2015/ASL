import cv2


capture = cv2.VideoCapture(0)


while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()