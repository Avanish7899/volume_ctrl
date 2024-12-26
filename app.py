from flask import Flask, render_template, Response
import cv2
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

# Initialize camera and volume control
cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.7)

# Set up audio controls
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

def generate_frames():
    vol = 0
    volBar = 400
    volPer = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw circles and line between thumb and index finger
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                # Calculate the distance between thumb and index finger
                length = math.hypot(x2 - x1, y2 - y1)

                # Map the length to volume range
                vol = np.interp(length, [50, 300], [minVol, maxVol])
                volBar = np.interp(length, [50, 300], [400, 150])
                volPer = np.interp(length, [50, 300], [0, 100])

                # Set volume
                volume.SetMasterVolumeLevel(vol, None)

                # Volume bar and percentage display
                cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            # Yield frame in byte format for streaming in HTML
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the main page

@app.route('/')
def index():
    return render_template('volume_control.html')

if __name__ == "__main__":
    app.run(debug=True)
