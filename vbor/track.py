import cv2
import numpy as np
import time
from VideoStream import VideoStream

resolution = (640, 480)
vs = VideoStream(src=1, usePiCamera=False, resolution=resolution).start()
time.sleep(2.0)

w = int(resolution[0] / 3)
h = int(resolution[1] / 3)
lx = 0 + w
ly = 0 + h
rx = 639 - w
ry = 479 - h

while True:
    frame = vs.read()

    cv2.imshow("Frame", frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([80, 100, 100])
    upper = np.array([120, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    # mask = cv2.medianBlur(mask, 5)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    res = cv2.medianBlur(res, 5)

    """Contour detection using mask"""
    _, contours, hierarchy = cv2.findContours(mask, 1, 2)
    if len(contours) > 0:
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        # cv2.drawContours(res, cnt, -1, (0, 255, 0), 3)
        (x, y), r = cv2.minEnclosingCircle(cnt)

        center = (int(x), int(y))
        radius = int(r)

        res = cv2.circle(res, center, radius, (0, 255, 0), 2)
        res = cv2.circle(res, center, 2, (0, 0, 255), 3)
        res = cv2.putText(res, ("%d, %d" % (x, y)), center, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75,
                          (0, 255, 255),
                          1,
                          cv2.LINE_AA)

        if x > lx and x < rx and y > ly and y < ry:
            res = cv2.putText(res, "Inside", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75,
                              (0, 255, 255),
                              1,
                              cv2.LINE_AA)
        else:
            if x < lx:
                dx = lx - x
                print "move left"
                res = cv2.putText(res, "Move left", (20, h-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75,
                                  (255, 255, 0),
                                  1,
                                  cv2.LINE_AA)
            if x > rx:
                dx = x - rx
                print "move right"
                res = cv2.putText(res, "Move right", (20, h-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75,
                                  (255, 255, 0),
                                  1,
                                  cv2.LINE_AA)
            if y < ly:
                dy = ly - x
                print "move up"
                res = cv2.putText(res, "Move up", (20, h-40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75,
                                  (255, 255, 0),
                                  1,
                                  cv2.LINE_AA)
            if y > ry:
                dy = y - ry
                print "move down"
                res = cv2.putText(res, "Move down", (20, h-40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75,
                                  (255, 255, 0),
                                  1,
                                  cv2.LINE_AA)

    res = cv2.rectangle(res, (0, 0), (639, 479), (255, 0, 0), 3)
    res = cv2.rectangle(res, (lx, ly), (rx, ry), (255, 0, 0), 3)
    res = cv2.circle(res, (320, 240), 2, (255, 0, 0), 3)

    cv2.imshow("Display", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
vs.stop()
