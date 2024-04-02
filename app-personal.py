import cv2
import numpy as np
from Network import Network
cap = cv2.VideoCapture(0)

hand_signals = ['PALM', 'I', 'FIST', 'THUMB', 'OK']
CNN = Network()
CNN = CNN.load_model('HandRegconition-Personal-epoch10')

while cap.isOpened():
    ret, frame = cap.read()
    frame=cv2.flip(frame, 1)
    if not ret:
        continue
    key = cv2.waitKey(10)
    if (key == 27):
            break
    roi = frame[100:400, 120:420]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
    copy = frame.copy()
    cv2.rectangle(copy, (120, 100), (420, 400), (255,0,0), 5)
    
    roi = roi.reshape(1, 1, 28, 28) 
    roi = roi/255
    result = np.argmax(CNN.predict(roi))
    cv2.putText(copy, str(hand_signals[result]), (300 , 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)    
        
cap.release()
cv2.destroyAllWindows() 