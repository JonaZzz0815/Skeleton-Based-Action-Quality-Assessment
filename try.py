
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

        self.pose = mp.solutions.pose.Pose(self.mode,min_detection_confidence=0.5)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

    def findPose(self, img, draw=True,display=False):
        copied_image = img.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(copied_image, 
                                            self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS,
                                            landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                            connection_drawing_spec=self.mpDraw.DrawingSpec(color=(49,125,237),
                                                                               thickness=2, circle_radius=2)
                                            )
        if display:
        
            plt.figure(figsize=[22,22])
            plt.subplot(121);plt.imshow(img[:,:,::-1]);plt.title("Input Image");plt.axis('off');
            plt.subplot(122);plt.imshow(copied_image[:,:,::-1]);plt.title("Pose detected Image");plt.axis('off');
            plt.savefig('test.jpg')
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

if __name__ == "__main__":



    ### construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input", required=True,
    #  help="path to our input video")
    # ap.add_argument("-o", "--output", required=True,
    #  help="path to our output video")
    ap.add_argument("-s", "--fps", type=int, default=30,
     help="set fps of output video")
    ap.add_argument("-b", "--black", type=str, default=False,
     help="set black background")
    args = vars(ap.parse_args())

    detector = PoseDetector()
    image_path = '/public/home/CS272/zhangqj-cs272/final/FineDiving_Dataset/Trimmed_Video_Frames/FINADiving_MTL_256s/03/1/00008181.jpg'
    output = cv2.imread(image_path)
    img,landmarks,connect=detector.findPose(output, draw=True, display=True)
    print(landmarks)
    
    print(len(landmarks))

    
    cv2.destroyAllWindows()