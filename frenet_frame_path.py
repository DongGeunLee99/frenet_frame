#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import numpy as np
import matplotlib.pyplot as plt
import copy
import math

from .quintic_polynomials_planner import QuinticPolynomial
from . import cubic_spline_planner

# Parameter
MAX_SPEED = 10.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 3.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 2.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 3.0  # maximum road width [m]
D_ROAD_W = 1.5  # road width sampling length [m]
DT = 0.5  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 4.0 / 3.6  # target speed [m/s]
D_T_S = 2.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1.2  # sampling number of target speed
ROBOT_RADIUS = 0.5  # robot radius [m]

# cost weights 
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

#------------------------------------------------------------------------
s_h_e = 0.01
k = 0.5 
dt = 0.01
L = 1.0
max_steer = np.radians(29.0)
#------------------------------------------------------------------------

show_animation = True

class State():

    def __init__(self, x, y, yaw, v=0.0):
        super().__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        delta = np.clip(delta, -max_steer, max_steer)
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = FrenetFrameNode.normalize_angle(self.yaw)
        self.v += acceleration * dt

class QuarticPolynomial:
    
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                    [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                    axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
            self.a3 * t ** 3 + self.a4 * t ** 4
        
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))
                Js = sum(np.power(tfp.s_ddd, 2))

                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv
                
                frenet_paths.append(tfp)
    
    return frenet_paths

def calc_global_paths(fplist, csp):
    for fp in fplist:

        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist

def check_collision(fp, ob):
    for i in range(len(ob)):
        d = [((ix - ob[i][0]) ** 2 + (iy - ob[i][1]) ** 2)
            for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= (ROBOT_RADIUS) ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                fplist[i].s_dd]):
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                fplist[i].c]):
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]

def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    if not fplist:
        print("Empty fplist")
        return None

    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp
    return best_path


def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


class FrenetFrameNode(Node):
    def __init__(self):
        super().__init__("node_name")
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.number_timer_ = self.create_timer(0.07, self.run_simulation)
        self.subscriber_ = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 5)
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.converted = []
        self.cspp = None
        self.s0 = 0.0

        self.my_x = 0.0
        self.my_y = 0.0
        self.is_spline_calculated = False

    print(" start!!")

#------------------------------------------------------------------------
    def stanley_control(self, state, cx, cy, cyaw, last_target_idx):
        global k, s_h_e

        current_target_idx, error_front_axle = self.calc_target_index(state, cx, cy)

        if last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx

        theta_e = self.normalize_angle_(cyaw[current_target_idx] - state.yaw)
        theta_e *= s_h_e
        theta_d = np.arctan2(k * error_front_axle, state.v)
        delta = theta_e + theta_d

        if delta < -max_steer:
            delta = -max_steer
        elif delta > max_steer:
            delta = max_steer

        return delta, current_target_idx


    def normalize_angle_(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def normalize_angle(angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle


    def calc_target_index(self, state, cx, cy):
        fx = state.x + L * np.cos(state.yaw)
        fy = state.y + L * np.sin(state.yaw)

        dx = [fx - icx for icx in cx]
        dy = [fy - icy for icy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                        -np.sin(state.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle
#------------------------------------------------------------------------
    
    def odom_callback(self, odom_msg):

        position = odom_msg.pose.pose.position
        self.my_x = round(position.x, 3)
        self.my_y = round(position.y, 3)

        self.wx_ = [self.my_x, 10.0]
        self.wy_ = [self.my_y, 0.0]
        self.cspp = cubic_spline_planner.CubicSpline2D(self.wx_, self.wy_)
        _, _, _, _, csp = generate_target_course(self.wx_,self.wy_)
        s0 = csp.s[0]
        self.s0 = s0
        if not self.is_spline_calculated:
            self.is_spline_calculated = True 
            self.cx, self.cy, self.cyaw, ck, s = cubic_spline_planner.calc_spline_course(
                self.wx_, self.wy_, ds=0.1)
            state = State(x = self.my_x, y = self.my_y, yaw = 0.0, v=0.0)
            self.target_idx, _ = self.calc_target_index(state, self.cx, self.cy)
            
    

    def lidar_callback(self, msg):
        self.converted.clear()
        close_indices = []

        for i, value in enumerate(msg.ranges):
            if not np.isinf(value):
                angle = msg.angle_increment * i
                x = round(np.cos(angle) * value + self.my_x, 3)
                y = round(np.sin(angle) * value + self.my_y, 3)

                if -5 <= y <= 5 and 0.5 < x < 10:
                    self.converted.append([x, y])
                    if value <= ROBOT_RADIUS:
                        close_indices.append(len(self.converted) - 1)
                        # print("Close ", close_indices)
                        print(len(close_indices))



    c_speed = 1.0 / 3.6  # current speed [m/s] 
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    result_x = []
    result_y = []
    

    def run_simulation(self):
        global path

        path = frenet_optimal_planning(
            self.cspp, self.s0, self.c_speed, self.c_accel, self.c_d, self.c_d_d, self.c_d_dd, self.converted)
        
        self.s0 = path.s[1]
        self.c_d = path.d[1]
        self.c_d_d = path.d_d[1]
        self.c_d_dd = path.d_dd[1]
        self.c_speed = path.s_d[1]
        self.c_accel = path.s_dd[1]
        
#----------------------------------------------------------------------

        state = State(x = self.my_x, y = self.my_y, yaw = path.yaw[0], v=0.0)

        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        ai = 0.3
        di, self.target_idx = self.stanley_control(state, self.cx, self.cy, self.cyaw, self.target_idx)

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)

        self.result_x.extend(x)
        self.result_y.extend(y)
#----------------------------------------------------------------------

        cmd_vel_msg = Twist()
        if abs(self.c_d_d) >= 0.05:
            print("Frenet")
            print(self.c_d_d)
            cmd_vel_msg.linear.x = self.c_speed
            cmd_vel_msg.angular.z = self.c_d_d
        else:
            print("Stanley")
            cmd_vel_msg.linear.x = ai
            cmd_vel_msg.angular.z = di
            print(np.degrees(di))
        self.cmd_vel_pub.publish(cmd_vel_msg)

        if show_animation:
            area = 5.0
            area_ = 3.0
            wx = [0.0, 10.0]
            wy = [0.0, 0.0]
            self.tx, self.ty, _, _, _ = generate_target_course(wx, wy)

            if not hasattr(self, 'fig'):
                self.fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            self.fig.clf()
            plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
            axs = [self.fig.add_subplot(121), self.fig.add_subplot(122)]

            axs[0].plot(self.tx, self.ty, label="Target Course")
            axs[0].plot([point[0] for point in self.converted], [point[1] for point in self.converted], "xk", label="Lidar Points")
            axs[0].plot(path.x[1:], path.y[1:], "-or", label="Frenet Optimal Path")
            axs[0].plot(path.x[1], path.y[1], "vc", label="Current Position")
            axs[0].set_xlim(path.x[1] - area, path.x[1] + area)
            axs[0].set_ylim(path.y[1] - area, path.y[1] + area)
            axs[0].set_title("Frenet")
            axs[0].grid(True)
            axs[0].legend()

            axs[1].plot(self.cx, self.cy, ".r", label="Course Points")
            axs[1].plot(self.result_x, self.result_y, "-b", label="Trajectory")
            axs[1].plot(self.cx[self.target_idx], self.cy[self.target_idx], "xg", label="Target")
            axs[1].axis("equal")
            axs[1].set_xlim(self.my_x - area_, self.my_x + area_)
            axs[1].set_ylim(self.my_y - area_, self.my_y + area_)
            axs[1].grid(True)
            axs[1].set_title("Global")
            axs[1].legend()

            plt.pause(0.0001)
            plt.show(block=False)

        if np.hypot(path.x[1] - self.tx[-1], path.y[1] - self.ty[-1]) <= 0.8:
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = 0.0
            cmd_vel_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_vel_msg)
            print("Finish")
            plt.grid(True)
            plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = FrenetFrameNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()