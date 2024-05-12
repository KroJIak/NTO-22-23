import cv2
from time import time
from math import hypot

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

        HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        grayImg = cv2.inRange(HSVImg, HSVMin, HSVMax)
        cv2.imwrite(f'Img-{i+1}/img-{i+1}-gray.png', grayImg)
        #cv2.imshow('gray', grayImg)
        #while cv2.waitKey(1) != 27: pass
        robots = []
        _, thresh = cv2.threshold(grayImg, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= squareCell:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                robots.append((cX, cY))

        for num, pos in enumerate(robots):
            grayImg = cv2.inRange(HSVImg, PHSVMin, PHSVMax)
            cv2.imshow('gray', grayImg)
            while cv2.waitKey(1) != 27: pass
            _, thresh = cv2.threshold(grayImg, 127, 255, 0)
            purpleCircles = []
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                print('CCC', cv2.contourArea(c))
                if cv2.contourArea(c) >= 5:
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    #print(hypot(abs(pos[0]-cX), abs(pos[1]-cY)), lens)
                    if hypot(abs(pos[0]-cX), abs(pos[1]-cY)) == lens:
                        print('YEz')
                        purpleCircles.append((cX, cY))

            grayImg = cv2.inRange(HSVImg, GHSVMin, GHSVMax)
            _, thresh = cv2.threshold(grayImg, 127, 255, 0)
            greenCircles = []
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) >= 5:
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if hypot(abs(pos[0]-cX), abs(pos[1]-cY)) < lens: greenCircles.append((cX, cY))
            print(len(purpleCircles), len(greenCircles))
            cv2.circle(img, pos, radius, (0, 0, 255), 2)
            cv2.putText(img, f'id: {num}', (pos[0] + 50, pos[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imwrite(f'Img-{i+1}/img-{i+1}.png', img)
        '''if i == 0: cam1.release()
        elif i == 1: cam2.release()
        else: cam3.release()'''

if __name__ == '__main__':
    colors = (0, 0, 255)
    squareCell = 10
    radius = 30
    lens = 40
    #реальные значения
    HSVMin = (92, 108, 204)
    HSVMax = (130, 166, 247)
    GHSVMin = (63, 31, 182)
    GHSVMax = (71, 255, 252)
    PHSVMin = (130, 76, 199)
    PHSVMax = (171, 155, 255)
    camIds = [8, 4, 0]
    main()
    cv2.destroyAllWindows()