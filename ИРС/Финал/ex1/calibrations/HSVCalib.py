from cv2 import createTrackbar, getTrackbarPos
import cv2
import numpy as np

def nothing(*arg): pass
#####camera = cv2.VideoCapture(2)
#####succes, img = camera.read()
img = cv2.imread('../output-ssh/img-3.png')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
cl = clahe.apply(l_channel)
limg = cv2.merge((cl, a, b))
eimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow('buttons')
createTrackbar('hmin', 'buttons', 0, 255, nothing)
createTrackbar('smin', 'buttons', 0, 255, nothing)
createTrackbar('vmin', 'buttons', 0, 255, nothing)
createTrackbar('hmax', 'buttons', 0, 255, nothing)
createTrackbar('smax', 'buttons', 0, 255, nothing)
createTrackbar('vmax', 'buttons', 0, 255, nothing)

while True:
    hmin, smin, vmin, hmax, smax, vmax = getTrackbarPos('hmin', 'buttons'), getTrackbarPos('smin', 'buttons'), getTrackbarPos('vmin', 'buttons'), getTrackbarPos('hmax', 'buttons'), getTrackbarPos('smax', 'buttons'), getTrackbarPos('vmax', 'buttons')
    hsv_min = np.array([hmin, smin, vmin])
    hsv_max = np.array([hmax, smax, vmax])
    img2 = cv2.inRange(img1, hsv_min, hsv_max)
    cv2.imshow('img', img2)
    if cv2.waitKey(1) == 27:
        print(f'HSVMin = ({hmin}, {smin}, {vmin})')
        print(f'HSVMax = ({hmax}, {smax}, {vmax})')
        cv2.destroyAllWindows()
        break

HSVMin = (12, 40, 188)
HSVMax = (17, 89, 255)