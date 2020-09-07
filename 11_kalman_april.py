import numpy as np
import cv2
import apriltag

cap = cv2.VideoCapture(0)
aprildet = apriltag.Detector()

kalman = None
def kalman_start():
    global kalman
    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

found = False
found_counter = 0

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if found:
        tp = kalman.predict()
        center = ( int(tp[0]), int(tp[1]) )
        print(center)
        cv2.circle(frame, center, 10, (0,0,255), 3)

    result = aprildet.detect(gray)
    if len(result)>0:
        if not found:
            print('A')
            kalman_start() # Restart kalman
            found_counter = 0
        found = True
        center = ( result[0].center[0], result[0].center[1] )
        center_int = ( int(center[0]), int(center[1]) )
        cv2.circle(frame, center_int, 10, (0,255,0), 3)
        mp1 = np.array([[np.float32(center[0])],[np.float32(center[1])]])
        print(center)
        kalman.correct(mp1)
    else:
        found_counter+=1
        if found_counter>200:
            found = False

    cv2.imshow('frame',np.flip(frame, axis=1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()