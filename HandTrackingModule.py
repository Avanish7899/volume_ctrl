import cv2
import mediapipe as mp
import math


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, min_detection_confidence=float(self.detectionCon),
                                        min_tracking_confidence=float(self.trackCon))
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def fingersUp(self):
        """Returns a list of 1s and 0s representing each finger's status (1 for up, 0 for down)"""
        fingers = []
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]

            # Thumb
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                fingers.append(1)  # Thumb up
            else:
                fingers.append(0)  # Thumb down

            # Four Fingers (Index, Middle, Ring, Pinky)
            for id in range(8, 21, 4):  # Landmark IDs for finger tips
                if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
                    fingers.append(1)  # Finger up
                else:
                    fingers.append(0)  # Finger down

        return fingers
