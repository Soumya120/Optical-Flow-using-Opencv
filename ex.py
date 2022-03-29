import numpy as np
import cv2 as cv
print(cv.__version__)
cap = cv.VideoCapture('people.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 300,
                       qualityLevel = 0.2,
                       minDistance = 2,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = (0, 255, 0)

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        print(i, mask.shape, (a,b), (c,d), color)
        mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color, 2)
        frame = cv.circle(frame,(int(a),int(b)),5,color,-1)
    img = cv.add(frame,mask)

    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv.destroyAllWindows()
cap.release()
