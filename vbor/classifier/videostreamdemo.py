from VideoStream import VideoStream
import datetime
import argparse
import imutils
import time
import cv2

vs = VideoStream(usePiCamera=True, resolution=(160, 120)).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, 400)
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
vs.stop()
