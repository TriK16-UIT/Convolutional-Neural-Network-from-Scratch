import cv2
import os

def select_signals(key, number, mode, signals_length):
    if 48 <= key <= signals_length + 47:
        number = key - 48
    if key == 73 or key == 105:
        mode = 1
    if key == 79 or key == 111:
        mode = 0
    return number, mode

cap = cv2.VideoCapture(0)

hand_signals = ['PALM', 'I', 'FIST', 'THUMB', 'OK']
mode = 0
number = -1    
while cap.isOpened():

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
            continue
    key = cv2.waitKey(10)
    if (key == 27): #Press ESC to exit
        break
    number, mode = select_signals(key, number, mode, len(hand_signals))
    #define region of interest
    roi = frame[100:400, 120:420]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)

    cv2.imshow('roi sacled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (120, 100), (420, 400), (255,0,0), 5)

    cv2.putText(copy, 'Current hand signal for recording: {} - {}'.format(str(number), str(hand_signals[number])), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
    if mode == 0:
        image_count = 0
        cv2.putText(copy, 'Press I to start capturing hand gesture.', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
    elif mode == 1:
        image_count+=1
        cv2.putText(copy, 'Currently recording! Press O to stop capturing hand gesture.', (0, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(copy, str(image_count), (400 , 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.imwrite('./Personal/' + str(number) + '/' + str(image_count) + ".jpg", roi)
    cv2.imshow('frame', copy)    
cap.release()
cv2.destroyAllWindows() 
