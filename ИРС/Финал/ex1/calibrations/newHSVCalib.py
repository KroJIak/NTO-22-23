import cv2
import numpy as np

def imageResize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def empty(x): pass

cv2.namedWindow('HSV')
cv2.resizeWindow('HSV', 640, 240)
cv2.createTrackbar('HUE Min', 'HSV', 0, 179, empty)
cv2.createTrackbar('HUE Max', 'HSV', 179, 179, empty)
cv2.createTrackbar('SAT Min', 'HSV', 0, 255, empty)
cv2.createTrackbar('SAT Max', 'HSV', 255, 255, empty)
cv2.createTrackbar('VALUE Min', 'HSV', 0, 255, empty)
cv2.createTrackbar('VALUE Max', 'HSV', 255, 255, empty)

while True:
    img = cv2.imread('../../R.png')
    resizedImg = imageResize(img, height=700)
    imgHSV = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2HSV)

    hMin = cv2.getTrackbarPos('HUE Min', 'HSV')
    hMax = cv2.getTrackbarPos('HUE Max', 'HSV')
    sMin = cv2.getTrackbarPos('SAT Min', 'HSV')
    sMax = cv2.getTrackbarPos('SAT Max', 'HSV')
    vMin = cv2.getTrackbarPos('VALUE Min', 'HSV')
    vMax = cv2.getTrackbarPos('VALUE Max', 'HSV')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    mask = cv2.inRange(imgHSV, lower, upper)
    #result = cv2.bitwise_and(resizedImg, resizedImg, mask=mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #hStack = np.hstack([resizedImg, mask, result])
    hStack = np.hstack([resizedImg, mask])

    #cv2.imshow('Original', resizedImg)
    #cv2.imshow('HSV Color Space', imgHSV)
    #cv2.imshow('Mask', mask)
    #cv2.imshow('Result', result)
    cv2.imshow('Horizontal Stacking', hStack)
    if cv2.waitKey(1) == 27: break

print(f'HSVMin = ({hMin}, {sMin}, {vMin})')
print(f'HSVMax = ({hMax}, {sMax}, {vMax})')