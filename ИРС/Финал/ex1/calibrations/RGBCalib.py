from cv2 import createTrackbar, getTrackbarPos
import cv2
import numpy as np

def nothing(*arg): pass
#camera = cv2.VideoCapture(2)
#succes, img = camera.read()
img = cv2.imread('../cam4-t2.png')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
cl = clahe.apply(l_channel)
limg = cv2.merge((cl, a, b))
eimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
img1 = cv2.cvtColor(eimg, cv2.COLOR_BGR2RGB)
cv2.namedWindow('buttons')
createTrackbar('rmin', 'buttons', 0, 255, nothing)
createTrackbar('gmin', 'buttons', 0, 255, nothing)
createTrackbar('bmin', 'buttons', 0, 255, nothing)
createTrackbar('rmax', 'buttons', 0, 255, nothing)
createTrackbar('gmax', 'buttons', 0, 255, nothing)
createTrackbar('bmax', 'buttons', 0, 255, nothing)

while True:
    rmin, gmin, bmin, rmax, gmax, bmax = getTrackbarPos('rmin', 'buttons'), getTrackbarPos('gmin', 'buttons'), getTrackbarPos('bmin', 'buttons'), getTrackbarPos('rmax', 'buttons'), getTrackbarPos('gmax', 'buttons'), getTrackbarPos('bmax', 'buttons')
    rgb_min = np.array([rmin, gmin, bmin])
    rgb_max = np.array([rmax, gmax, bmax])
    img2 = cv2.inRange(img1, rgb_min, rgb_max)
    cv2.imshow('img', img2)
    if cv2.waitKey(1) == 27:
        print(f'RGBMin = ({rmin}, {gmin}, {bmin})')
        print(f'RGBMax = ({rmax}, {gmax}, {bmax})')
        cv2.destroyAllWindows()
        break