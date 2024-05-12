import sys
sys.path.insert(0, '../lib/')

from py_nto_task2 import Task
import cv2

import signal

run = False

def SIGINT_handler(signum, frame):
    global run
    print('SIGINT')
    run = False

def solve():
    from math import acos, hypot
    from time import sleep

    def update():
        global sceneImg, key, task
        sleep(0.02)
        #cv2.imshow('Image', sceneImg)
        #key = cv2.waitKey(20)
        #if key == 27: sys.exit()
        sceneImg = task.getTaskScene()

    def getCentersObjectsByColor(R, G, B, points=False):
        global sceneImg
        color = (R, G, B)
        imgRGB = cv2.cvtColor(sceneImg, cv2.COLOR_BGR2RGB)
        if points: objects = {}
        else: objects = []
        while len(objects) == 0 and run:
            n = 1
            contours, hierarchies = cv2.findContours(cv2.inRange(imgRGB, color, color), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for i in contours:
                M = cv2.moments(i)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    if list(imgRGB[cy][cx]) == list(color) and points:
                        objects[f'P{n}'] = [cx, cy]
                        n += 1
                    elif list(imgRGB[cy][cx]) == list(color):
                        objects.append([cx, cy])
        return objects

    def getArgs(cenPoint):
        global robot
        cenRobot = getCentersObjectsByColor(137, 137, 193)[0]
        cenDir = getCentersObjectsByColor(41, 41, 58)[0]
        dx, dy = cenDir[0] - cenRobot[0], cenRobot[1] - cenDir[1]
        px, py = cenPoint[0] - cenRobot[0], cenRobot[1] - cenPoint[1]
        axb = dx * px + dy * py
        hypDir, hypPoint = hypot(abs(dx), abs(dy)), hypot(abs(px), abs(py))
        try: rad = acos(axb / (hypDir * hypPoint))
        except: rad = 0
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
            if rad > 3: robot.setMotorVoltage([v, -v])
            else: robot.setMotorVoltage([v1, -v1])
            update()

    def findHypot(cenPoint, i):
        global v2, hyp, mhyp
        hyp = float('inf')
        v = v2
        if i == 2:
            mhyp = float('inf')
            v = 3
        while getHyps(i) and run:
            if i == 2: mhyp = hyp
            _, _, hyp = getArgs(cenPoint)
            robot.setMotorVoltage([v, v])
            update()


    global run, sceneImg, robot, cenPoint, hyp, mhyp, task, v1, v2
    task = Task()
    task.start()
    run = True
    offset = 40
    directions = {'S': [0, offset],
                  'SW': [-offset, offset],
                  'W': [-offset, 0],
                  'NW': [-offset, -offset],
                  'N': [0, -offset],
                  'NE': [offset, -offset],
                  'E': [offset, 0],
                  'SE': [offset, offset]}
    robot = task.getRobots()[0]
    sceneImg = task.getTaskScene()
    cenPoints = getCentersObjectsByColor(204, 204, 204, points=True)
    msg_out = str(len(cenPoints))
    for num in cenPoints: msg_out += r' \n ' + num + ' ' + str(cenPoints[num])

    #print(msg_out)
    task.sendMessage(msg_out)
    msg_in = task.getTask()
    #msg_in = r'9 \n P3_SW P2_S P4_W P5_NE P16_W P8 P13_N P15 P16'

    count, points = msg_in.split('\n')
    count = int(count)
    points = points.split()
    for i in range(count):
        _ = points[i].find('_')
        if _ != -1:
            point = points[i][:_]
            dir = points[i][_+1:]
        else:
            point = points[i]
            dir = 'Any'
        cenPoint = cenPoints[point]
        v1, v2 = 4, 15
        _, _, mhyp = getArgs(cenPoint)
        for j in range(3):
            findRotate(cenPoint, j)
            findHypot(cenPoint, j)
        if dir != 'Any':
            cenPoint = getCentersObjectsByColor(137, 137, 193)[0]
            cenPoint = list(map(sum, zip(cenPoint, directions[dir])))
            findRotate(cenPoint, 2)
        robot.setMotorVoltage([0, 0])
        update()
        sleep(0.5)
        msg_out = f'{points[i]} OK'
        #print(msg_out)
        task.sendMessage(msg_out)
        sleep(0.5)
    cenPoint = [900, 150]
    _, _, mhyp = getArgs(cenPoint)
    for i in range(3):
        findRotate(cenPoint, i)
        findHypot(cenPoint, i)
    task.stop()

if __name__ == '__main__':

    signal.signal(signal.SIGINT, SIGINT_handler)
    solve()
