import sys
sys.path.insert(0, '../lib/')

from py_nto_task4 import Task
import cv2

import signal

run = False

def SIGINT_handler(signum, frame):
    global run
    print('SIGINT')
    run = False

## Здесь должно работать ваше решение
def solve():
    global run

    ## Запуск задания и таймера (внутри задания)
    task = Task()
    task.start()
    run = True

    ## Загружаем изображение из задачи
    # сцена отправляется с определенной частотй
    # для этого смотри в документацию к задаче
    mapWithZones = task.getTaskMapWithZones()
    sceneImg     = task.getTaskScene()
    # cv2.imshow('some', mapWithZones)
    # cv2.waitKey(0)

    robots = task.getRobots()
    print('Number of robots:', len(robots))
    print("Robot 0")
    print("Left Motor angle:", robots[0].state.leftMotorAngle)
    print("Right Motor angle:", robots[0].state.rightMotorAngle)
    print("Left Motor speed:", robots[0].state.leftMotorSpeed)
    print("Right Motor speed:", robots[0].state.rightMotorSpeed)


    while (run):

        v = [10, 0]
        robots[0].setMotorVoltage(v)

        sceneImg = task.getTaskScene()
        cv2.imshow('Scene', sceneImg)
        cv2.waitKey(20) # мс

    task.stop()

if __name__ == '__main__':

    signal.signal(signal.SIGINT, SIGINT_handler)
    solve()
