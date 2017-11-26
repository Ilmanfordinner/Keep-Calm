import numpy as np
import cv2

capcam = cv2.VideoCapture(0)
cap = cv2.VideoCapture('vid.mp4')

#cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    ret, fram2 = capcam.read()
    if ret == True:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Window', frame)
        cv2.imshow('Window2',fram2)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()