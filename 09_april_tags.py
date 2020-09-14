import numpy as np
import cv2
import apriltag

cap = cv2.VideoCapture(0)  

aprildet = apriltag.Detector()

found = False
found_counter = 0

box_color = (0,255,0)
box_tickness = 3
box_corner_radio = 10

id_font = cv2.FONT_HERSHEY_SIMPLEX 
id_scale = 1.0
id_color = (0,0,255)
id_tickness = 3

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = aprildet.detect(gray)
    if len(results)>0:
        print(results)
        for result in results:    
            center = ( int(result.center[0]), int(result.center[1]) )
            corners = result.corners.astype(int)
            cv2.circle(frame, tuple(corners[0]), box_corner_radio, box_color, -1)
            cv2.line(frame, tuple(corners[0]), tuple(corners[1]), box_color, box_tickness) 
            cv2.line(frame, tuple(corners[1]), tuple(corners[2]), box_color, box_tickness) 
            cv2.line(frame, tuple(corners[2]), tuple(corners[3]), box_color, box_tickness) 
            cv2.line(frame, tuple(corners[3]), tuple(corners[0]), box_color, box_tickness) 
            cv2.putText(frame, str(result.tag_id), (center[0]-10, center[1]+10), id_font, id_scale, id_color, id_tickness, cv2.LINE_AA) 

    # cv2.imshow('frame',np.flip(frame, axis=1))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
