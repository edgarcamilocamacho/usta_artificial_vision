import cv2
import numpy as np

def hsv_to_cv(hsv):
    return np.array([int(hsv[0]/2.0), int(255.0*hsv[1]/100.0), int(255.0*hsv[2]/100.0)])

#BLUE
# hsv_limits = [ [210,30,30], [270,100,100] ]
#GREEN
hsv_limits = [ [90,30,30], [150,100,100] ]

cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_to_cv(hsv_limits[0]), hsv_to_cv(hsv_limits[1]))
    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=5)

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        center = (x, y)
        _, _, h, w = cv2.boundingRect(c)
        rat = h/w
        if radius > 10 and rat>0.9 and rat<1.1:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imshow('frame', np.flip(frame, axis=1) )
    cv2.imshow('mask', np.flip(mask, axis=1) )
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()