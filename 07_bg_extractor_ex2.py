import numpy as np
import cv2
import collections
from functools import *

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
	# Capture frame and convert it to gray scale
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    mask = np.zeros(gray.shape).astype('uint8')
    mask[fgmask>0] = 255
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Compute foreground (input masked)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('foreground',foreground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()