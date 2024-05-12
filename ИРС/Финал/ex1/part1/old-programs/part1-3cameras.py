import cv2
import cv2.aruco as aruco
from time import time
from scipy import ndimage
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

def contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    cImg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cImg

def main(index, url):
    global arucoIds, tempIds

    cameras.append(cv2.VideoCapture(url))
    cameras[index].set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cameras[index].set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cameras[index].set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cameras[index].set(cv2.CAP_PROP_FOCUS, 0)
    t = time() + 10
    while t > time(): succes, img = cameras[index].read()

    n = 0
    while len(arucoIds) == 0 and n < 5:
        tempIds = {}
        lTime = time() + T
        while lTime > time():
            #####img = cv2.imread(url)
            succes, img = cameras[index].read()
            detectAruco(contrast(img))
        n += 1
        arucoIds = {}
        for id in tempIds:
            if tempIds[id][0] >= p and id in nums: arucoIds[id] = list(map(int, tempIds[id][1]))
    print('Arucos:', end=' ')
    for id in tempIds: print(f'{id},', end=' ')

    #####img = cv2.imread(url)
    succes, img = cameras[index].read()
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
    cv2.imwrite(f'Img-{url}.png', img)
    while cv2.waitKey(1) != 27: pass
    cameras[index].release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    nums = list(map(int, input().split()))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    T = 1
    p = 5
    squareCell = 1250
    #####HSVMin = (0, 0, 173)
    #####HSVMax = (0, 0, 180)
    #реальные значения
    HSVMin = (10, 13, 105)
    HSVMax = (18, 255, 255)
    arucoIds = {}
    cameras = []
    urls = [0, 0, 0]
    for i, url in enumerate(urls): main(i, url)
    cv2.destroyAllWindows()