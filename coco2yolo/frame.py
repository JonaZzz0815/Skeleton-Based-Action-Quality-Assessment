import numpy as np 
import cv2 
cap = cv2.VideoCapture('v2.mp4') 
ret, frame = cap.read() 
cv2.imshow('frame', frame) 
c = cv2.waitKey(50000) 


cv2.destroyAllWindows() 

