import math
import time

import cv2
import mediapipe as mp


class poseDetector:

    # mode: webcam feed vs. recorded video
    # upBody: full body pose vs. upper body pose
    # smooth: whether to apply smoothing to the output (to reduce jitters)
    # detectionCon: minimum detection confidence threshold
    # trackCon: minimum tracking confidence threshold

    def __init__(
        self, mode=False, upBody=False, smooth=True, detectionCon=0.7, trackCon=0.7
    ):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        # Instantiate the pose model
        self.pose = self.mpPose.Pose(
            self.mode,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )

    # img: image to process
    # draw: whether to draw the output on the image
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            for id, lm in enumerate(
                self.results.pose_landmarks.landmark
            ):  # iterate through all the landmarks
                if id == 0:  # Skip the nose landmark
                    continue
                if id not in [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    17,
                    18,
                    19,
                    20,
                ]:  # Skip facial landmarks and fingers
                    # height, width, channels
                    h, w, c = (
                        img.shape
                    )  # calculate pixel coordinates of landmarks relative to image dimensions
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(
                        img, (cx, cy), 5, (255, 0, 0), cv2.FILLED
                    )  # draw landmark on original image

            # Manually draw connections w/o facial landmarks
            for (
                conn
            ) in self.mpPose.POSE_CONNECTIONS:  # iterate through predefined connections
                if conn[0] > 10 and conn[1] > 10:  # Exclude facial landmarks
                    # retrieve positions of connected landmarks and draw lines between them using calculated pixel coordinates
                    pt1 = self.results.pose_landmarks.landmark[conn[0]]
                    pt2 = self.results.pose_landmarks.landmark[conn[1]]
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        return img
    
    # img: image to process
    # draw: whether to draw the output on the image
    def findPosition(self, img, draw=True):
        self.lmList = []  # store detected landmarks
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if id == 0:  # Skip the nose landmark
                    continue
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)  # calculate pixel coordinates
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
   
    # p1, p2, p3: landmark indices for the 3 points
    # p2 = vertex
    def findAngle(self, img, p1, p2, p3, draw=True):

        # Extract coordinates of the points
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate angle in degrees
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        # make angles positive and within 360 degrees
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(
                img,
                str(int(angle)),
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
        return angle
