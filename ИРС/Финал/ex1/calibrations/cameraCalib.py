from cv2 import createTrackbar, getTrackbarPos
import cv2

def nothing(*arg): pass
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cv2.namedWindow('buttons')
createTrackbar('bright', 'buttons', 0, 200, nothing)
createTrackbar('contrast', 'buttons', 0, 150, nothing)
createTrackbar('saturation', 'buttons', 0, 150, nothing)
createTrackbar('hue', 'buttons', 0, 150, nothing)
createTrackbar('gain', 'buttons', 0, 150, nothing)
createTrackbar('exposure', 'buttons', 0, 150, nothing)

while True:
    bright = getTrackbarPos('bright', 'buttons')
    contrast = getTrackbarPos('contrast', 'buttons')
    saturation = getTrackbarPos('contrast', 'buttons')
    hue = getTrackbarPos('contrast', 'buttons')
    gain = getTrackbarPos('contrast', 'buttons')
    exposure = getTrackbarPos('contrast', 'buttons')
    camera.set(cv2.CAP_PROP_BRIGHTNESS, bright)
    camera.set(cv2.CAP_PROP_CONTRAST, contrast)
    camera.set(cv2.CAP_PROP_SATURATION, saturation)
    camera.set(cv2.CAP_PROP_HUE, hue)
    camera.set(cv2.CAP_PROP_GAIN, gain)
    camera.set(cv2.CAP_PROP_EXPOSURE, exposure)

    succes, img = camera.read()
    cv2.imshow('img', img)
    if cv2.waitKey(1) == 27:
        print(bright, contrast, saturation, hue, gain, exposure)
        camera.release()
        cv2.destroyAllWindows()
        break