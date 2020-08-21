import cv2
import numpy as np

def hsv_to_cv(hsv):
    return np.array([int(hsv[0]/2.0), int(255.0*hsv[1]/100.0), int(255.0*hsv[2]/100.0)])

#BLUE, límites en H, S y V
# hsv_limits = [ [210,30,30], [270,100,100] ]
#GREEN, límites en H, S y V
hsv_limits = [ [90,30,30], [150,100,100] ]

# Crea un objeto de captura desde la cámara 0
# Puede ser un archivo
cap = cv2.VideoCapture(0)

while(1):
    # Lee un grame de la cámara
    _, frame = cap.read()
    # Lo convierte a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Realiza la binarización
    mask = cv2.inRange(hsv, hsv_to_cv(hsv_limits[0]), hsv_to_cv(hsv_limits[1]))
    # Muestra
    cv2.imshow('frame', np.flip(frame, axis=1) )
    cv2.imshow('mask', np.flip(mask, axis=1) )
    # Espera antes de leer el siguiente frame
    k = cv2.waitKey(5) & 0xFF 
    if k == 27: # Sale si se presiona Esc
        break

cv2.destroyAllWindows()