
import cv2
import mediapipe as mp
import time
import argparse
import matplotlib.pyplot as plt

class PoseDetector:


    def __init__(self, mode = True, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.pose = mp.solutions.pose.Pose(self.mode,model_complexity=1,min_detection_confidence=0.0,min_tracking_confidence=0.0)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

    def findPose(self, img, draw=True,display=False):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if not self.results.pose_landmarks:
            # no pose detected????
            print("No pos detected")
            
            
        return img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS

    def getPosition(self, img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
