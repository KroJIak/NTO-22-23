import cv2
import numpy as np
from scipy import ndimage
from time import time
cam1 = cv2.VideoCapture(0)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam1.set(cv2.CAP_PROP_FOCUS, 0)
t = time() + 1
while t > time(): succes, frame1 = cam1.read()
cam2 = cv2.VideoCapture(8)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam2.set(cv2.CAP_PROP_FOCUS, 0)
t = time() + 1
while t > time(): succes, frame2 = cam2.read()
cam3 = cv2.VideoCapture(4)
cam3.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam3.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam3.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam3.set(cv2.CAP_PROP_FOCUS, 0)
t = time() + 1
while t > time(): succes, frame3 = cam3.read()
h, w = frame1.shape[:2]
center = (int(w / 2), int(h / 2))
rotationMatrix = cv2.getRotationMatrix2D(center, -2, 10)

frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
rFrame1 = ndimage.rotate(frame1, -1, reshape=False)
frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
rFrame2 = ndimage.rotate(frame2, 1.7, reshape=False)
frame3 = cv2.rotate(frame3, cv2.ROTATE_90_CLOCKWISE)
rFrame3 = ndimage.rotate(frame3, -0.1, reshape=False)

rFrame1 = rFrame1[:, :-17]
frame12 = np.append(rFrame1, rFrame2, axis=1)

frame12 = frame12[:, :-280]
lastFrame = np.append(frame12, rFrame3, axis=1)
for i, line in enumerate(lastFrame):
    for j, pix in enumerate(line):
        if list(pix) != [0, 0, 0]: lastFrame[i][j] = np.array(list(pix))
        else: lastFrame[i][j] = np.array([255, 255, 255])
cv2.imwrite('R.png', lastFrame)
#cv2.imshow('Frame', cv2.hconcat([
#    cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE),
#    cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE),
#    cv2.rotate(frame3, cv2.ROTATE_90_CLOCKWISE)]))
#while cv2.waitKey(1) != 27: pass
#cv2.destroyAllWindows()