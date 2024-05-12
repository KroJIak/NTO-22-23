import cv2
import cv2.aruco as aruco
from time import time
from scipy import ndimage
import numpy as np

def correctImage(frame1, frame2, frame3):
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print(frame1.shape)
    rFrame1 = ndimage.rotate(frame1, -1, reshape=False)
    frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
    rFrame2 = ndimage.rotate(frame2, 1.7, reshape=False)
    frame3 = cv2.rotate(frame3, cv2.ROTATE_90_CLOCKWISE)
    rFrame3 = ndimage.rotate(frame3, -0.1, reshape=False)

    rFrame1 = rFrame1[:, :-252]
    frame12 = np.append(rFrame1, rFrame2, axis=1)

    frame12 = frame12[:, :-402]
    lastFrame = np.append(frame12, rFrame3, axis=1)
    for i, line in enumerate(lastFrame):
        for j, pix in enumerate(line):
            lastFrame[i][j] = np.array(list(pix))
            # if list(pix) != [0, 0, 0]: lastFrame[i][j] = np.array(list(pix))
            # else: lastFrame[i][j] = np.array([255, 255, 255])
    return lastFrame

def detectAruco(img):
    global tempIds
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_1000),
                                   aruco.DetectorParameters())
    bbox, ids, _ = detector.detectMarkers(imgGray)
    if ids is not None:
        for i, id in enumerate(ids):
            x1, x2 = bbox[i][0][0][0], bbox[i][0][2][0]
            y1, y2 = bbox[i][0][0][1], bbox[i][0][2][1]
            try: tempIds[id[0]] = [tempIds[id[0]][0] + 1, [x1+(x2-x1)/2, y2+(y1-y2)/2]]
            except: tempIds[id[0]] = [1, [x1+(x2-x1)/2, y2+(y1-y2)/2]]

def contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    cImg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cImg

def main():
    global arucoIds, tempIds
    cam1 = cv2.VideoCapture(8)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam1.set(cv2.CAP_PROP_FOCUS, 0)
    t = time() + ttt
    while t > time(): succes, frame1 = cam1.read()
    cam2 = cv2.VideoCapture(8)
    cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam2.set(cv2.CAP_PROP_FOCUS, 0)
    t = time() + ttt+20
    while t > time(): succes, frame2 = cam2.read()
    cam3 = cv2.VideoCapture(4)
    cam3.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam3.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam3.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam3.set(cv2.CAP_PROP_FOCUS, 0)
    t = time() + ttt
    while t > time(): succes, frame3 = cam3.read()
    img = correctImage(frame1, frame2, frame3)
    n = 0
    while len(arucoIds) == 0 and n < 5:
        tempIds = {}
        lTime = time() + T
        while lTime > time(): detectAruco(contrast(img))
        n += 1
        arucoIds = {}
        for id in tempIds:
            if tempIds[id][0] >= p and id in nums: arucoIds[id] = list(map(int, tempIds[id][1]))
    print('Arucos:', end=' ')
    for id in tempIds: print(f'{id},', end=' ')

    #####img = cv2.imread(url)
    #succes, img = cameras[index].read()
    HSVImg = cv2.cvtColor(contrast(img), cv2.COLOR_BGR2HSV)
    grayImg = cv2.inRange(HSVImg, HSVMin, HSVMax)
    #####cv2.imshow('g', grayImg)
    #####while cv2.waitKey(1) != 27: pass
    size = 35
    for id in arucoIds:
        x = arucoIds[id][0] - size if arucoIds[id][0] - size > 0 else 0
        y = arucoIds[id][1] - size if arucoIds[id][1] - size > 0 else 0
        for i in range(x, x+size*2):
            for j in range(y, y+size*2):
                grayImg[j][i] = 255
    #####cv2.imshow('g', grayImg)
    #####while cv2.waitKey(1) != 27: pass

    ret, thresh = cv2.threshold(grayImg, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= squareCell:
            x, y, w, h = cv2.boundingRect(c)
            for i, aruco in enumerate(sorted(arucoIds)):
                if x < arucoIds[aruco][0] < x+w and y < arucoIds[aruco][1] < y+h:
                    if i == 0: cv2.rectangle(img, (x, y), (x + w, y + h), colors[0], 2)
                    elif i == len(arucoIds)-1: cv2.rectangle(img, (x, y), (x + w, y + h), colors[2], 2)
                    else: cv2.rectangle(img, (x, y), (x + w, y + h), colors[1], 2)
                    break
    cv2.imwrite(f'Img.png', img)
    #while cv2.waitKey(1) != 27: pass
    cam1.release()
    cam2.release()
    cam3.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    nums = list(map(int, input().split()))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    T = 1
    p = 5
    ttt = 18
    squareCell = 1250
    #####HSVMin = (0, 0, 173)
    #####HSVMax = (0, 0, 180)
    #реальные значения
    HSVMin = (12, 40, 188)
    HSVMax = (17, 89, 255)
    arucoIds = {}
    main()
    cv2.destroyAllWindows()
