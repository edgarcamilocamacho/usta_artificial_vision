import cv2
import numpy as np

# Crea un objeto de captura desde la cámara 0
# Puede ser un archivo
cap = cv2.VideoCapture(0)

while(1):
    # Lee un grame de la cámara
    _, frame_rgb = cap.read()
    # Lo convierte a escala de grises
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    # Muestra
    cv2.imshow('frame_RGB', np.flip(frame_rgb, axis=1) )
    cv2.imshow('frame_gray', np.flip(frame_gray, axis=1) )
    # Espera antes de leer el siguiente frame
    k = cv2.waitKey(5) & 0xFF 
    if k == 27: # Sale si se presiona Esc
        break

cv2.destroyAllWindows()