import cv2
import cv2.aruco as aruco
from time import time

def settingCamera(cam):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(1920))
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(1080))
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cam.set(cv2.CAP_PROP_EXPOSURE, -5)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    cam.set(cv2.CAP_PROP_AUTO_WB, 0)
    cam.set(cv2.CAP_PROP_TEMPERATURE, 7000)
    t = time() + 5
    r = False
    while t > time() or not r: r, _ = cam.read()
    print(f'Set settings camera <{cam}>')

def detectAruco(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_1000),
                                   aruco.DetectorParameters())
    bbox, ids, _ = detector.detectMarkers(imgGray)
    corIds = {}
    if ids is not None:
        for i, id in enumerate(ids):
            if id[0] in nums:
                x1, x2 = bbox[i][0][0][0], bbox[i][0][2][0]
                y1, y2 = bbox[i][0][0][1], bbox[i][0][2][1]
                corIds[id[0]] = [x1+(x2-x1)/2, y2+(y1-y2)/2]
    return corIds

def main():
    for i in range(3):
        '''if i == 0:
            cam1 = cv2.VideoCapture(camIds[0])
            settingCamera(cam1)
        elif i == 1:
            cam2 = cv2.VideoCapture(camIds[1])
            settingCamera(cam2)
        else:
            cam3 = cv2.VideoCapture(camIds[2])
            settingCamera(cam3)'''

        #if i == 0: _, img = cam1.read()
        #elif i == 1: _, img = cam2.read()
        #else: _, img = cam3.read()
        if i == 0: img = cv2.imread('../../new-img-8.png')
        elif i == 1: img = cv2.imread('../../new-img-4.png')
        else: img = cv2.imread('../../new-img-0.png')

        arucoIds = detectAruco(img)
        if len(arucoIds) != 0:
            print(arucoIds)
            HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if i == 0:
                HSVMin = (12, 48, 217)
                HSVMax = (22, 98, 255)
            elif i == 1:
                HSVMin = (9, 66, 190)
                HSVMax = (20, 114, 252)
            else:
                HSVMin = (7, 67, 185)
                HSVMax = (15, 102, 252)
            grayImg = cv2.inRange(HSVImg, HSVMin, HSVMax)
            cv2.imwrite(f'Img-{i+1}/img-{i+1}-gray-before.png', grayImg)
            for id in arucoIds:
                x = int(arucoIds[id][0] - size) if arucoIds[id][0] - size > 0 else 0
                y = int(arucoIds[id][1] - size) if arucoIds[id][1] - size > 0 else 0
                for j in range(y, y+size*2):
                    for g in range(x, x+size*2):
                        try: grayImg[j][g] = 255
                        except: pass
            cv2.imwrite(f'Img-{i+1}/img-{i+1}-gray-after.png', grayImg)
            _, thresh = cv2.threshold(grayImg, 127, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) >= squareCell:
                    x, y, w, h = cv2.boundingRect(c)
                    for id in arucoIds:
                        if x < arucoIds[id][0] < x+w and y < arucoIds[id][1] < y+h:
                            if id == min(nums): cv2.rectangle(img, (x, y), (x + w, y + h), colors[0], 2)
                            elif id == max(nums): cv2.rectangle(img, (x, y), (x + w, y + h), colors[2], 2)
                            else: cv2.rectangle(img, (x, y), (x + w, y + h), colors[1], 2)
                            break
        else:
            print(f'Camera of index {i} dont find arucos')
            break
        cv2.imwrite(f'Img-{i + 1}/img-{i + 1}.png', img)

        '''if i == 0: cam1.release()
        elif i == 1: cam2.release()
        else: cam3.release()'''

if __name__ == '__main__':
    nums = list(map(int, input().split()))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    size = 27
    squareCell = 1250
    #реальные значения
    HSVMin = (4, 67, 183)
    HSVMax = (16, 101, 253)
    camIds = [8, 4, 0]
    main()
    cv2.destroyAllWindows()