import sys
sys.path.insert(0, '../lib/')

from py_nto_task1 import Task
import cv2

import signal

run = False

def SIGINT_handler(signum, frame):
    global run
    print('SIGINT')
    run = False

def solve():
    from math import acos, hypot, cos, sin, ceil, floor, pi
    from time import sleep

    def update():
        global sceneImg, key, task
        sleep(0.02)
        #cv2.imshow('Image', sceneImg)
        #key = cv2.waitKey(20)
        #if key == 27: sys.exit()
        sceneImg = task.getTaskScene()

    def getCentersObjectsByColor(R, G, B):
        global sceneImg
        color = (R, G, B)
        imgRGB = cv2.cvtColor(sceneImg, cv2.COLOR_BGR2RGB)
        objects = []
        while len(objects) == 0 and run:
            contours, hierarchies = cv2.findContours(cv2.inRange(imgRGB, color, color), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for i in contours:
                M = cv2.moments(i)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    if list(imgRGB[cy][cx]) == list(color): objects.append([cx, cy])
        return objects

    def getArgs(cenPoint, getCor=False):
        global robot
        cenRobot = getCentersObjectsByColor(137, 137, 193)[0]
        cenDir = getCentersObjectsByColor(41, 41, 58)[0]
        dx, dy = cenDir[0] - cenRobot[0], cenRobot[1] - cenDir[1]
        px, py = cenPoint[0] - cenRobot[0], cenRobot[1] - cenPoint[1]
        axb = dx * px + dy * py
        hypDir, hypPoint = hypot(abs(dx), abs(dy)), hypot(abs(px), abs(py))
        try:
            rad = acos(axb / (hypDir * hypPoint))
        except:
            rad = 0
        if getCor:
            return rad, hypDir, hypPoint, dx, dy
        else:
            return rad, hypDir, hypPoint

    def getHyps(i):
        global hyp, mhyp
        hyps = [hyp > mhyp // 2, hyp > mhyp // 8, hyp <= mhyp]
        return hyps[i]

    def findRotate(cenPoint, i):
        global v1
        rad = 0
        n = 3.05
        v = 0.15
        if i == 2:
            n = 3.13
            v = 0.05
        while rad < n and run:
            #cv2.circle(sceneImg, cenPoint, 5, (0, 0, 0))
            rad, _, _ = getArgs(cenPoint)
            if 0 < rad < 0.1: n -= 0.005
            if rad > 3:
                robot.setMotorVoltage([v, -v])
            else:
                robot.setMotorVoltage([v1, -v1])
            update()

    def findHypot(cenPoint, i):
        global v2, hyp, mhyp
        hyp = float('inf')
        v = v2
        if i == 2:
            mhyp = float('inf')
            v = 2.5
        while getHyps(i) and run:
            if i == 2: mhyp = hyp
            _, _, hyp = getArgs(cenPoint)
            robot.setMotorVoltage([v, v])
            update()

    def getSensors():
        global sceneImg
        cenPoint = getCentersObjectsByColor(137, 137, 193)[0]
        cenSensors = getCentersObjectsByColor(137, 137, 193)[0]
        cenPoint[0] += 40
        rad, hyp, _, dx, dy = getArgs(cenPoint, getCor=True)
        rad = pi - rad
        hyp += 15
        hyp = hyp / cos(0.3)
        dist = [0.3, -0.3]
        s = [0, 0]
        for i in range(2):
            nrad = rad + dist[i]
            x, y = hyp * cos(nrad), hyp * sin(nrad)
            if dy < 0: y *= -1
            rx = 0.5 if x >= 0 else -0.5
            ry = 0.5 if y >= 0 else -0.5
            x = ceil(x) if x - int(x) >= rx else floor(x)
            y = ceil(y) if y - int(y) >= ry else floor(y)
            color = list(sceneImg[cenSensors[1] + y][cenSensors[0] + x])
            if color != [255, 255, 255]: s[i] = 1
        if dy >= 0: s[0], s[1] = s[1], s[0]
        return s

    def line():
        v = 10
        k = 0.5
        s = getSensors()
        if s[0] and s[1]:
            robot.setMotorVoltage([v, v])
        elif s[0]:
            robot.setMotorVoltage([v * k, v])
        elif s[1]:
            robot.setMotorVoltage([v, v * k])
        update()

    global run, sceneImg, robot, cenPoint, hyp, mhyp, task, v1, v2
    task = Task()
    task.start()
    countRound = int(task.getTask())
    #countRound = 2
    run = True
    robot = task.getRobots()[0]
    sceneImg = task.getTaskScene()
    cenPoint = getCentersObjectsByColor(254, 0, 0)[0]
    v1, v2 = 4, 15
    _, _, mhyp = getArgs(cenPoint)
    for i in range(3):
        findRotate(cenPoint, i)
        findHypot(cenPoint, i)
    cenPoint = getCentersObjectsByColor(137, 137, 193)[0]
    cenPoint[1] += 30
    findRotate(cenPoint, 2)
    cenPoint = getCentersObjectsByColor(137, 137, 193)[0]
    for i in range(countRound):
        _, _, hyp = getArgs(cenPoint)
        while hyp < 10 and run:
            _, _, hyp = getArgs(cenPoint)
            line()
        while hyp > 10 and run:
            _, _, hyp = getArgs(cenPoint)
            line()
    while hyp < 10 and run:
        _, _, hyp = getArgs(cenPoint)
        line()
    cenPoint = [870, 150]
    _, _, mhyp = getArgs(cenPoint)
    for i in range(3):
        findRotate(cenPoint, i)
        findHypot(cenPoint, i)
    task.stop()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, SIGINT_handler)
    solve()