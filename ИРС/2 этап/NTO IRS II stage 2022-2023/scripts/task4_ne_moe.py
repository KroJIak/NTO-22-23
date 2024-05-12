import base64
import sys
import os
import time

from scipy.spatial import KDTree

"""

 █████ █████   █████████   ███████████     ███████    █████   ███   █████
░░███ ░░███   ███░░░░░███ ░░███░░░░░███  ███░░░░░███ ░░███   ░███  ░░███ 
 ░░███ ███   ░███    ░███  ░███    ░███ ███     ░░███ ░███   ░███   ░███ 
  ░░█████    ░███████████  ░██████████ ░███      ░███ ░███   ░███   ░███ 
   ███░███   ░███░░░░░███  ░███░░░░░░  ░███      ░███ ░███   ░███   ░███  
  ███ ░░███  ░███    ░███  ░███        ░░███     ███   ░███████████████ 
 █████ █████ █████   █████ █████        ░░░███████░      ░░██████████     
░░░░░ ░░░░░ ░░░░░   ░░░░░ ░░░░░           ░░░░░░░         ░░░░░░░░░░      


"""

sys.path.insert(0, '../lib/')
if sys.platform.startswith('win'):
    try:
        os.add_dll_directory("C:\\mingw64\\bin")
        os.add_dll_directory("C:\\NTO\\nto_robotics\\Windows\\opencv\\x64\\mingw\\lib")
        os.add_dll_directory("C:\\NTO\\nto_robotics\\Windows\\opencv\\x64\\mingw\\bin")
        os.add_dll_directory("C:\\NTO\\nto_robotics\\Windows\\lib")
    except Exception:
        os.add_dll_directory("E:\\mingw64\\bin")
        os.add_dll_directory(
            "C:\\Users\mmpan\\PycharmProjects\\NTI\\second\\IRS\\nto_robotics-main\\Windows\\opencv\\x64\mingw\\lib")
        os.add_dll_directory(
            "C:\\Users\\mmpan\\PycharmProjects\\NTI\\second\\IRS\\nto_robotics-main\\Windows\\opencv\\x64\\mingw\\bin")
        os.add_dll_directory("C:\\Users\\mmpan\\PycharmProjects\\NTI\\second\\IRS\\\\nto_robotics-main\\Windows\\lib")
from py_nto_task4 import Task
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

import signal

# FF6666
Flag = True
run = False
DEBUG = False
if sys.platform.startswith('win'):
    DEBUG = True

MIN_RGB_GRAY = (145, 145, 145)
MAX_RGB_GRAY = (155, 155, 155)

MIN_RGB_RED = (230, 100, 100)
MAX_RGB_RED = (255, 120, 120)

K_P, K_I, K_D = 0.0045, 0.0035, 0.0000001
DELTA_T = 1 / 62

V_KOEF = 1
if not sys.platform.startswith('win'):
    V_KOEF = 100

MAX_ERROR = 0.065
DELTA_PX = 30
ROTATE_SPEED = 0.003 * V_KOEF
FORWARD_SPEED = 5 * 1e-2 * V_KOEF
END_POINT = (900, 150)

SEA_OF_THIEVES = {'W': math.pi * 1.5, 'N': 0, 'S': math.pi, 'E': math.pi / 2, 'NE': math.pi / 4, 'SE': 3 * math.pi / 4,
                  'SW': 5 * math.pi / 4, 'NW': 7 * math.pi / 4, '': 0}

SEA_OF_THIEVES_DELTAS = {'W': (DELTA_PX, 0), 'N': (0, DELTA_PX), 'S': (0, -DELTA_PX), 'E': (-DELTA_PX, 0),
                         'NE': (-DELTA_PX, DELTA_PX), 'SE': (-DELTA_PX, -DELTA_PX),
                         'SW': (DELTA_PX, -DELTA_PX), 'NW': (DELTA_PX, DELTA_PX), '': (0, 0)}


class PID:
    def __init__(self, K_P: float, K_I: float, K_D: float):
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.I = 0
        self.D = 0
        self.obj = 0
        self.prev_err = 0

    def reset(self):
        self.I = 0
        self.D = 0
        self.prev_err = 0

    def set_obj(self, obj: float):
        self.obj = obj

    def calc_out(self, inp: float):
        err = self.obj - inp
        P = err
        self.I += err * DELTA_T
        self.D = (err - self.prev_err) / DELTA_T

        self.prev_err = err

        return self.K_P * P + self.K_I * self.I + self.K_D * self.D


class Robot:
    def __init__(self, cords: tuple, angle: float, sim_robot):
        self.cords = cords
        self.angle = angle
        self.sim_robot = sim_robot
        self.pid = PID(K_P, K_I, K_D)

    def update_state(self, new_cords: tuple, angle: float):
        self.cords = new_cords
        self.angle = angle

    def rotate(self, obj: float):
        obj = transform_angle(obj)
        self.angle = transform_angle(self.angle)
        self.pid.set_obj(obj)
        voltage = self.pid.calc_out(self.angle)

        """if voltage >= 0.11:
            voltage = 0
            self.pid.reset()"""

        err = abs(obj - self.angle)

        # v = [voltage, -voltage]
        if obj >= self.angle:
            k = 1
        else:
            k = -1

        # ПОЧЕМУ ОНО РАБОТАЕТ ТО А


        v = [ROTATE_SPEED * k, -ROTATE_SPEED * k]

        self.sim_robot.setMotorVoltage(v)
        if abs(obj - self.angle) <= MAX_ERROR or (abs(obj - self.angle + math.pi * 2) <= MAX_ERROR):
            return True
        return False


def calc_angle(robot_cords: tuple, obj_cords: tuple):
    delta_x, delta_y = obj_cords[0] - robot_cords[0], -obj_cords[1] + robot_cords[1]

    angle = math.atan2(delta_x, delta_y)
    # if 0 < angle <= 0.5:
    #     angle = 0.08
    # elif 0 > angle >= -0.5:
    #     angle = -0.08
    return angle


"""

 ██████████      ███████    ██████   █████  ██ ███████████
░░███░░░░███   ███░░░░░███ ░░██████ ░░███  ███░█░░░███░░░█
 ░███   ░░███ ███     ░░███ ░███░███ ░███ ░░░ ░   ░███  ░ 
 ░███    ░███░███      ░███ ░███░░███░███         ░███    
 ░███    ░███░███      ░███ ░███ ░░██████         ░███    
 ░███    ███ ░░███     ███  ░███  ░░█████         ░███    
 ██████████   ░░░███████░   █████  ░░█████        █████   
░░░░░░░░░░      ░░░░░░░    ░░░░░    ░░░░░        ░░░░░    


 ███████████    ███████    █████  █████   █████████  █████   █████
░█░░░███░░░█  ███░░░░░███ ░░███  ░░███   ███░░░░░███░░███   ░░███ 
░   ░███  ░  ███     ░░███ ░███   ░███  ███     ░░░  ░███    ░███ 
    ░███    ░███      ░███ ░███   ░███ ░███          ░███████████ 
    ░███    ░███      ░███ ░███   ░███ ░███          ░███░░░░░███ 
    ░███    ░░███     ███  ░███   ░███ ░░███     ███ ░███    ░███ 
    █████    ░░░███████░   ░░████████   ░░█████████  █████   █████
   ░░░░░       ░░░░░░░      ░░░░░░░░     ░░░░░░░░░  ░░░░░   ░░░░░ 


"""
N_SAMPLE = 1200  # number of sample_points
N_KNN = 6  # number of edge from one sampled point
MAX_EDGE_LEN = 200.0  # [m] Maximum edge length
MIN_EDGE_LEN = 50

show_animation = False


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + \
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(start_x, start_y, goal_x, goal_y,
                 obstacle_x_list, obstacle_y_list, robot_radius, *, rng=None):
    """
    Run probabilistic road map planning
    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    :param robot_radius: robot radius
    :param rng: (Optional) Random generator
    :return:
    """
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)

    sample_x, sample_y = sample_points(start_x, start_y, goal_x, goal_y,
                                       robot_radius,
                                       obstacle_x_list, obstacle_y_list,
                                       obstacle_kd_tree, rng)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y,
                                 robot_radius, obstacle_kd_tree)

    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy

    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN or d <= MIN_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    Road map generation
    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):

        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    #  plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]
    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True

    while True:
        if not open_set:
            print('not')
            plt.cla()
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []



    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        dist, index = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def main(image, s, g, rng=None):
    sx, sy = s
    gx, gy = g
    # start and goal position
    sy = sy  # [m]
    gy = gy  # [m]
    robot_size = 27.5  # [m]

    ox, oy = get_obstacles_for_find(image)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, rng=rng)

    return rx, ry


MIN_RGB_BLACK = (0, 0, 0)
MAX_RGB_BLACK = (85, 85, 85)


def get_mask_for_find(image, min_rgb: tuple, max_rgb: tuple):
    mask = cv.inRange(image, min_rgb, max_rgb)
    return mask


def get_obstacles_for_find(image):
    mask = get_mask_for_find(image, MIN_RGB_BLACK, MAX_RGB_BLACK)

    # mask = cv.bitwise_not(mask)

    ox, oy = [], []
    for y, row in enumerate(mask):
        for x, px in enumerate(row):
            if px == 255:
                ox.append(x)
                oy.append(800 - y)

    return ox, oy


"""
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
"""


def transform_angle(angle: float):
    return angle + 2 * math.pi if angle <= 0.0010 else angle


def get_cords_square(image):
    MIN_RGB = (200, 200, 200)
    MAX_RGB = (210, 210, 210)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    mask = cv.inRange(image, MIN_RGB, MAX_RGB)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
    nice_contours = []
    data = []

    for i in contours:
        nice_contours.append(i)
    for index in nice_contours:
        M = cv.moments(index)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            data.append((cx, cy))
    data = sorted(data)
    return data


def get_center(cnt) -> tuple:
    rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
    box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
    box = np.int0(box)  # округление координат

    center = int(rect[0][0]), int(rect[0][1])
    return rect, box, center


def get_mask(image, min_rgb: tuple, max_rgb: tuple):
    mask = cv.inRange(image, min_rgb, max_rgb)
    return mask


def get_image(imageString: str):
    buffer = base64.b64decode(imageString)
    array = np.frombuffer(buffer, dtype=np.uint8)
    img = cv.imdecode(array, flags=1)
    return img


def norm_angle(angle: float):
    while abs(angle) > 2 * math.pi:
        angle += 2 * math.pi
    return angle


def detect_robot(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.GaussianBlur(image, (5, 5), 0)

    mask = get_mask(image, MIN_RGB_GRAY, MAX_RGB_GRAY)
    mask = cv.erode(mask, None, iterations=1)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
    if hierarchy is None or contours is None:
        return []

    data = []

    big_cnt = contours[0]

    mask_red = get_mask(image, MIN_RGB_RED, MAX_RGB_RED)
    contours, hierarchy = cv.findContours(mask_red, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]

    _, _, center = get_center(big_cnt)

    big_x, big_y = center

    if hierarchy is None or contours is None:
        return []

    cords = []

    for cnt in contours:
        _, _, center = get_center(cnt)
        cords.append(center)

    eye = None
    for x, y in cords:
        distances = []
        for x1, y1 in cords:
            if x1 == x and y1 == y:
                continue
            dist = round((math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2) - 3) / 10) * 10
            distances.append(dist)
        if distances[0] == distances[1]:
            eye = x, y

    if not eye:
        return []

    center = big_x, big_y
    center_small = eye

    x, y = center_small[0] - center[0], center_small[1] - center[1]

    vector = (y, -x)  # повернули на 90 градусов, чтобы вектор был вaлиден для вычисления угла
    angle = -1 * math.atan2(-vector[1], vector[0])
    angle = norm_angle(angle)

    data.append((center, angle))

    return data


def SIGINT_handler(signum, frame):
    global run
    print('SIGINT')
    run = False


"""
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
"""


def generate_out(points: list):
    out_str = f'{len(points)}\n'
    for i in range(len(points)):
        out_str += f'P{i + 1} {points[i][0]} {points[i][1]}\n'
    return out_str


def get_objectives(task_str: str):
    task_str = task_str.split('\n')[1].split()
    out = []
    virgin = []
    for elem in task_str:
        elem = elem.replace('P', '')
        data = elem.split('_')
        out.append((int(data[0]) - 1, data[1] if len(data) == 2 else ''))
        virgin.append(elem)

    return out, virgin


def generate_nice_objectives(points: list, task: list) -> list:
    out = []
    for elem in task:
        point = points[elem[0]]
        new_point = (point[0] + SEA_OF_THIEVES_DELTAS[elem[1]][0], point[1] + SEA_OF_THIEVES_DELTAS[elem[1]][1])
        out.append(new_point)
        if elem[1] != '':
            out.append(point)

    out.append(END_POINT)
    return out


def solve():
    global run, Flag

    task = Task()
    task.start()
    run = True

    robots = task.getRobots()

    sceneImg = task.getTaskScene()

    state = detect_robot(sceneImg)[0]
    robot = Robot(state[0], state[1], robots[0])

    points = get_cords_square(sceneImg)

    if DEBUG:
        objectives, virgin = get_objectives('9\n P3_N P2 P4 P5 P16 P8 P13 P15 P16')
    else:
        task.sendMessage(generate_out(points))
        objectives, virgin = get_objectives(task.getTask())

    objectives = generate_nice_objectives(points, objectives)

    objectives = [(state[0][0], state[0][1])] + objectives
    real_time_objectives = []

    for ix in range(0, len(objectives) - 1):
        way = []
        new_obj_i = (objectives[ix][0], 800 - objectives[ix][1])
        new_obj_i1 = (objectives[ix + 1][0], 800 - objectives[ix + 1][1])

        rx, ry = main(sceneImg, new_obj_i, new_obj_i1)
        while not rx or not ry:
            plt.cla()
            rx, ry = main(sceneImg, new_obj_i, new_obj_i1)
        rx, ry = rx[::-1], ry[::-1]
        for ii in range(0, len(rx), 2):
            way.append((rx[ii], 800 - ry[ii]))
        way.append((rx[-1], 800 - ry[-1]))
        real_time_objectives += way

    i = 0

    while run:
        if i > len(real_time_objectives) - 1:
            break
        obj = real_time_objectives[i]
        state = detect_robot(sceneImg)[0]
        robot.update_state(state[0], state[1])

        robot.angle = transform_angle(robot.angle)

        angle = calc_angle(robot.cords, obj)

        angle = transform_angle(angle)

        flag = robot.rotate(angle)
        if flag:
            robots[0].setMotorVoltage([FORWARD_SPEED, FORWARD_SPEED])
            robot.pid.reset()

        sceneImg = task.getTaskScene()


        if DEBUG:
            cv.imshow('Scene', sceneImg)
            cv.waitKey(20)
        else:
            time.sleep(0.1)
        if abs(robot.cords[0] - obj[0]) <= 15 and abs(robot.cords[1] - obj[1]) <= 15:
            i += 1
            if i % 2 == 0:
                if DEBUG:
                    print(virgin[i // 2 - 1])
                    print(robot.cords[0], robot.cords[1])
                    print(obj[0], obj[1])
                    print(robot.angle)

                elif not (DEBUG):
                    robots[0].setMotorVoltage([0, 0])
                    time.sleep(0.5)
                    task.sendMessage('P' + virgin[i // 2 - 1] + ' OK')
                    time.sleep(0.5)
    task.stop()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, SIGINT_handler)
    solve()
