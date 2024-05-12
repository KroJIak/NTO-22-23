import cv2
import cv2.aruco as aruco
from time import time


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
            try:
                tempIds[id[0]] = [tempIds[id[0]][0] + 1, [x1 + (x2 - x1) / 2, y2 + (y1 - y2) / 2]]
            except:
                tempIds[id[0]] = [1, [x1 + (x2 - x1) / 2, y2 + (y1 - y2) / 2]]


def main(url):
    global arucoIds, tempIds
    n = 0
    camera = cv2.VideoCapture(url)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    camera.set(cv2.CAP_PROP_FOCUS, 0)
    t = time() + 5
    while t > time(): succes, img = camera.read()
    while len(arucoIds) == 0 and n < 5:
        tempIds = {}
        lTime = time() + T
        while lTime > time():
            succes, img = camera.read()
            #img = cv2.imread(url)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            eimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            detectAruco(eimg)
        n += 1
        arucoIds = {}
        for id in tempIds:
            if tempIds[id][0] >= p: arucoIds[id] = list(map(int, tempIds[id][1]))

    succes, img = camera.read()
    #img = cv2.imread(url)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    eimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    HSVImg = cv2.cvtColor(eimg, cv2.COLOR_BGR2HSV)
    grayImg = cv2.inRange(HSVImg, HSVMin, HSVMax)
    cv2.imshow('g', grayImg)
    while cv2.waitKey(1) != 27: pass
    raz = 30
    for id in arucoIds:
        x = arucoIds[id][0] - raz if arucoIds[id][0] - raz > 0 else 0
        y = arucoIds[id][1] - raz if arucoIds[id][1] - raz > 0 else 0
        for i in range(x, x + raz * 2):
            for j in range(y, y + raz * 2):
                grayImg[j][i] = 255
    cv2.imshow('g', grayImg)
    while cv2.waitKey(1) != 27: pass
    ret, thresh = cv2.threshold(grayImg, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if 2600 >= cv2.contourArea(c) >= squareCell:
            x, y, w, h = cv2.boundingRect(c)
            for aruco in arucoIds:
                if x < arucoIds[aruco][0] < x + w and y < arucoIds[aruco][1] < y + h:
                    cv2.rectangle(img, (x, y), (x + w, y + h), colors[0], 2)
                    break
            else: cv2.rectangle(img, (x, y), (x + w, y + h), colors[1], 2)
    cv2.imshow(f'Img-{url}', img)
    cv2.imwrite('../output.png', img)
    while cv2.waitKey(1) != 27: pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    colors = [(0, 0, 255), (0, 255, 0)]
    T = 2
    p = 20
    squareCell = 1000
    #HSVMin = (0, 0, 173)
    #HSVMax = (0, 0, 180)
    # реальные значения
    HSVMin = (12, 40, 188)
    HSVMax = (17, 89, 255)
    arucoIds = {}

    urls = [0]
    for url in urls: main(url)
    cv2.destroyAllWindows()
