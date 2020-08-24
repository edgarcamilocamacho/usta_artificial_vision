import numpy as np
import cv2
import collections
from functools import *

cap = cv2.VideoCapture(0)

# Create circular buffer
buffer = collections.deque(maxlen=500)
th = 50

while(True):
	# Capture frame and convert it to gray scale
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Append new frame to buffer
    buffer.append(gray.astype(float))
    # Calculate background model
    mean = (reduce(lambda x, y: x + y, buffer) / len(buffer)).astype('uint8')
    # Compute difference
    diff = np.abs(gray.astype('int')-mean.astype('int')).astype('uint8')
    # Calculate mask
    mask = np.zeros(gray.shape).astype('uint8')
    mask[diff>50] = 255
    # Compute foreground (input masked)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame',gray)
    cv2.imshow('background',mean)
    cv2.imshow('difference',diff)
    cv2.imshow('mask',mask)
    cv2.imshow('foreground',foreground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()