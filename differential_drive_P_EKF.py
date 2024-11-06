import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller, regulation_polar_coordinates, regulation_polar_coordinate_quat, wrap_angle, velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel
from robot_localization_system import FilterConfiguration, Map, RobotEstimator

map = Map()
filter_config = FilterConfiguration()

# # global variables
# W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
# landmarks = np.array([
#     [5, 10],
#     [15, 5],
#     [10, 15]
# ])
W_range = filter_config.W_range
landmarks = map.landmarks

def generate_range_bearing_observations(base_position):
    observations = []
    for lm in landmarks:
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
        bearing_meas = np.arctan2(dy, dx) - base_position[2]
        bearing_meas = wrap_angle(bearing_meas)
        observations.append([range_meas, bearing_meas])
    return np.array(observations)

# Functions for EKF integration
def landmark_range_observations(base_position):
    y = []
    for lm in landmarks:
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
        y.append(range_meas)
    return np.array(y)

def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()
    rot_quat = quat.toRotationMatrix()
    base_euler = pin.rpy.matrixToRpy(rot_quat)
    return base_euler[2]

def init_simulator(conf_file_name):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    return sim, dyn_model, dyn_model.getNumberofActuatedJoints()

def main():
    conf_file_name = "robotnik_noise.json"
    sim, dyn_model, num_joints = init_simulator(conf_file_name)
    sim.SetFloorFriction(100)
    time_step = sim.GetTimeStep()
    time = 0

    # MPC setup
    num_states = 3
    num_controls = 2
    C = np.eye(num_states)
    N_mpc = 10
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    init_pos = np.array([2.0, 3.0])
    init_quat = np.array([0, 0, 0.3827, 0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    regulator.updateSystemMatrices(sim, [init_pos[0], init_pos[1], init_base_bearing_], np.zeros(num_controls))
    regulator.setCostMatrices([1, 1, 0], [0.1, 0.0005])

    # EKF setup
    estimator = RobotEstimator(filter_config, map)
    estimator.start()
    x_est, Sigma_est = estimator.estimate()
    
    # Control and simulation variables
    u_mpc = np.zeros(num_controls)
    wheel_radius = 0.11
    wheel_base_width = 0.46
    cmd = MotorCommands()
    cmd.SetControlCmd(np.zeros(4), ["velocity"] * 4)

    # 记录状态数据
    true_state_all = []  # 真实状态
    est_state_all = []  # 估计状态

    # Main control loop
    while True:
        sim.Step(cmd, "torque")
        
        # Get estimated state from EKF

        # EKF 控制输入和预测步骤
        u_est = np.array([u_mpc[0]*0.5,u_mpc[1]]) # problem : u_est or u_mpc?
        # print("check u_est",u_mpc[1])
        # estimator.set_control_input(u_est)
        estimator.set_control_input(u_mpc)
        estimator.predict_to(time)
        
        # 获取新的观测并更新 EKF
        # base_position = [x_est[0], x_est[1], x_est[2]]
        obs_pos = sim.GetBasePosition()
        obs_ori = sim.GetBaseOrientation()
        obs_bearing = quaternion2bearing(obs_ori[3], obs_ori[0], obs_ori[1], obs_ori[2])
        obs_base_position = [obs_pos[0], obs_pos[1], obs_bearing]
        print("check base_position",obs_base_position)
        y = generate_range_bearing_observations(obs_base_position)
        # print("check y",y[:3])
        estimator.update_from_range_bearing_observations(y)

        x_est, Sigma_est = estimator.estimate()
        # x_est = base_position

        # 计算最优控制序列
        x0_mpc = np.hstack((x_est[0], x_est[1], x_est[2]))

        # 当前状态用于线性化
        cur_state_x_for_linearization = [x_est[0], x_est[1], x_est[2]]
        cur_u_for_linearization = u_mpc

        # 更新系统矩阵A和B，以当前状态为线性化点
        regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # print("check u_mpc",len(u_mpc))
        u_mpc = u_mpc[:2]

        # 转换为轮子速度控制指令
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(
            u_mpc[0], u_mpc[1], wheel_base_width, wheel_radius
        )
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity,
                                                left_wheel_velocity, right_wheel_velocity])
        cmd.SetControlCmd(angular_wheels_velocity_cmd, ["velocity"] * 4)
        time += time_step
        # print("check time",time)

        # 在主循环内每步后记录真实和估计的状态
        true_pos = sim.bot[0].base_position.copy()
        true_ori = sim.bot[0].base_orientation.copy()
        true_bearing = quaternion2bearing(true_ori[3], true_ori[0], true_ori[1], true_ori[2])
        # true_state = [true_pos[0], true_pos[1], true_bearing]
        true_state = [true_pos[0], true_pos[1]]

        true_state_all.append(true_state)
        # est_state_all.append([x_est[0], x_est[1], x_est[2]])
        est_state_all.append([x_est[0], x_est[1]])
        # print("check x_est",x_est,"check umpc",u_mpc)
        # 退出逻辑
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

    # 主循环退出后进行绘图
    true_state_all = np.array(true_state_all)
    est_state_all = np.array(est_state_all)

    plt.figure(figsize=(10, 6))
    plt.plot(true_state_all[:, 0], true_state_all[:, 1], label='true_state', color='blue')
    plt.plot(est_state_all[:, 0], est_state_all[:, 1], label='est_state', color='orange', alpha=0.5)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color='red', label='地标')
    plt.xlabel('X位置 [m]')
    plt.ylabel('Y位置 [m]')
    plt.title('机器人真实轨迹和估计轨迹')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()