import numpy as np
import math
import matplotlib.pyplot as plt

k = 0.1  # 前视距离系数
Lfc = 1.5  # 前视距离
Kp = 1.0  # 速度P控制器系数
dt = 0.2  # 时间间隔，单位：s
L = 0.9  # 车辆轴距，单位：m

class VehicleState:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
def update(state, a, delta):
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt
    return state

def PControl(target, current):
    a = Kp * (target - current)
    return a

def pure_pursuit_control(state, cx, cy, pind, Lfc):
    ind = calc_target_index(state, cx, cy, Lfc)
    if pind >= ind:
        ind = pind
    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1
    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
    if state.v < 0:  # back
        alpha = math.pi - alpha
    Lf = k * state.v + Lfc
    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)
    return delta, ind

def calc_target_index(state, cx, cy, Lfc = 1.5):
    # 搜索最临近的路点
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))
    L = 0.0
    Lf = k * state.v + Lfc
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind + 1] - cx[ind]
        dy = cy[ind + 1] - cy[ind]
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1
    return ind

def main():
    #  设置目标路点
    cx = np.arange(0, 50, 1)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    target_speed = 10.0 / 3.6  # [m/s]
    T = 100.0  # 最大模拟时间

    # 设置车辆的初始状态
    state = VehicleState(x=-0.0, y=-1.0, yaw=0.0, v=0.0)
    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy)
    while T >= time and lastIndex > target_ind:
        ai = PControl(target_speed, state.v)
        di, target_ind = pure_pursuit_control(state, cx, cy, target_ind)
        state = update(state, ai, di)
        time = time + dt
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.plot(cx[target_ind], cy[target_ind], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        plt.pause(0.001)


def pure_pursuit(path, Lfc = 1.0,v=0.1, dt=0.2, L=1.0):
    # Initialize empty lists for x and y coordinates
    cx = []
    cy = []

    # Iterate through each point in the path
    for point in path:
        # Append the x and y coordinates to the appropriate lists
        cx.append(point[0])
        cy.append(point[1])

    # Convert lists to numpy arrays for further processing
    cx = np.array(cx)
    cy = np.array(cy)

    target_speed = 1.0  # [m/s]
    T = 10.0  # 最大模拟时间

    # 设置车辆的初始状态
    state = VehicleState(x=-0.0, y=-0.5, yaw=np.pi / 2, v=v)
    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy, Lfc)
    # track = [(state.x, state.y, state.yaw, state.v)]
    track = [(state.x, state.y)]

    while lastIndex > target_ind:
        ai = PControl(target_speed, state.v)
        di, target_ind = pure_pursuit_control(state, cx, cy, target_ind, Lfc)
        state = update(state, ai, di)
        time = time + dt
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        # track.append((state.x, state.y, state.yaw, state.v))
        track.append((state.x, state.y))
    return track

def test_main():
    #  设置目标路点
    # cx = np.arange(0, 50, 1)
    # cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    path = [(0.3, 0), (0.3, 1), (0.3, 2), (0.3, 3), (0.3, 4), (0.3, 5), (0.3, 6), (0.3, 7), (0.3, 8), (0.3, 9)]
    cx = np.ones(10, dtype=float) * 2.0
    cy = np.arange(0, 10, 1)

    target_speed = 1.0  # [m/s]
    T = 100.0  # 最大模拟时间

    # 设置车辆的初始状态
    state = VehicleState(x=-0.0, y=-1.0, yaw=np.pi/2, v=0.0)
    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy)

    while T >= time and lastIndex > target_ind:
        ai = PControl(target_speed, state.v)
        di, target_ind = pure_pursuit_control(state, cx, cy, target_ind)
        state = update(state, ai, di)
        time = time + dt
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, ".b", label="trajectory")
        plt.plot(cx[target_ind], cy[target_ind], "go", label="target")

        plt.axis("equal")
        plt.grid(True)
        plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        plt.pause(0.001)
    plt.show()

if __name__ == '__main__':
    # main()
    test_main()