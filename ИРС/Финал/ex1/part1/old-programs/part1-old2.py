import cv2
import cv2.aruco as aruco
from time import time
import numpy as np

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

def main(url):
    global arucoIds, tempIds
    n = 0
    while len(arucoIds) == 0 and n < 10:
        tempIds = {}
        lTime = time() + T
        while lTime > time():
            #succes, img = camera.read()
            img = cv2.imread(url)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            eimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            detectAruco(eimg)
        n += 1
        arucoIds = {}
        for id in tempIds:
            if tempIds[id][0] >= p and id in nums: arucoIds[id] = list(map(int, tempIds[id][1]))

    #succes, img = camera.read()
    img = cv2.imread(url)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    eimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    HSVImg = cv2.cvtColor(eimg, cv2.COLOR_BGR2HSV)
    grayImg = cv2.inRange(HSVImg, HSVMin, HSVMax)
    for id in arucoIds:
        x = arucoIds[id][0] - 17 if arucoIds[id][0] - 17 > 0 else 0
        y = arucoIds[id][1] - 17 if arucoIds[id][1] - 17 > 0 else 0
        for i in range(x, x+34):
            for j in range(y, y+34):
                grayImg[j][i] = 255
    #cv2.imshow('g', grayImg)
    #while cv2.waitKey(1) != 27: pass
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
    cv2.imshow(f'Img-{url}', img)
    cv2.imwrite('output.png', img)
    while cv2.waitKey(1) != 27: pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    nums = list(map(int, input().split()))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    T = 1
    p = 40
    #2, 4, 10, 9
    squareCell = 1250
    #HSVMin = (0, 0, 173)
    #HSVMax = (0, 0, 180)
    #реальные значения
    HSVMin = (14, 94, 123)
    HSVMax = (101, 250, 255)
    arucoIds = {}
    camera = cv2.VideoCapture(0)
    urls = ['../cam1-t1.png', '../cam4-t1.png', '../cam8-t1.png']
    for url in urls: main(url)
    cv2.destroyAllWindows()