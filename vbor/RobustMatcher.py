import cv2
import numpy as np
from math import sqrt

class RobustMatcher(object):
    def __init__(self):

        self.MIN_MATCH_COUNT = 10

        self._hessianThreshold = 500

        self._surf = cv2.xfeatures2d.SURF_create(self._hessianThreshold);

        self.ratio = 0.75

    def bfWithRatioMatcher(self, img1, img2):
        matched = False

        result = img2
        h, w = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).shape
        x = w/2
        y = h/2
        area = h*w
        normArea = 1

        ## SURF features keypoints and descriptors extraction
        kp1, des1 = self._surf.detectAndCompute(img1, None)
        kp2, des2 = self._surf.detectAndCompute(img2, None)

        ## Match the two image descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        if des1 == None or des2 == None:
            print "None type detected"
            return result,normArea,x,y
        
        if len(des1) <= 1 or len(des2) <= 1:
            return result,normArea,x,y
        matches = matcher.knnMatch(des1, des2, k=2)

        print "Matches: ", len(matches)

        # ratio test
        good_matches = self.ratioTest(matches)

        print "Good: ", len(good_matches)

        #matchedImage = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
        #cv2.imshow("Brute Force Matching", matchedImage)

        #print "Matched: %d / %d" % (len(good_matches), self.MIN_MATCH_COUNT)

        if len(good_matches) > self.MIN_MATCH_COUNT:

            matched = True

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).shape
            # h, w = img1.shape

            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            if M == None:
                return result,normArea,x,y
            dst = cv2.perspectiveTransform(pts, M)

            # finding midpoint
            x = y = 0
            for i in dst:
                x += i[0, 0]
                y += i[0, 1]
            x, y = np.int32([x / 4, y / 4])

            #finding area
            x1,y1 = [dst[0,0,0], dst[0,0,1]]
            x2,y2 = [dst[1,0,0], dst[1,0,1]]
            x3,y3 = [dst[2,0,0], dst[2,0,1]]
            x4,y4 = [dst[3,0,0], dst[3,0,1]]
            tempArea = 0.5 * (sqrt( (x3 - x1)**2 + (y3 - y1)**2 ) * sqrt( (x4 - x2)**2 + (y4 - y2)**2 ))
            normArea = tempArea/area
            
            # finding text display position
            xt, yt = np.int32([dst[0, 0, 0], dst[0, 0, 1] - 10])

            result = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
            result = cv2.circle(result, (x, y), 10, (0, 0, 255), 2, cv2.LINE_AA)
            result = cv2.putText(result, ("%d, %d" % (x, y)), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 0),
                                 1, cv2.LINE_AA)
        else:
            matched = False

        if matched:
            print "Matched"
            result = cv2.putText(result, "Matched", (xt, yt), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                     0.75,
                                     (0, 255, 0),
                                     1, cv2.LINE_AA)
        return result,normArea,x,y

    def ratioTest(self, matches):
        good = []
        if len(matches) > 1 and len(matches) < 500:
            for m, n in matches:
                # check distance ratio test
                if m.distance / n.distance < self.ratio:
                    good.append(m)
        return good
