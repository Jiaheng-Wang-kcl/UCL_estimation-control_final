import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15]
        ])


def landmark_range_observations(base_position):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
       
        y.append(range_meas)

    y = np.array(y)
    return y


def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def main():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    # conf_file_name = "robotnik_noise.json"  # Configuration file for the robot
    sim,dyn_model,num_joints=init_simulator(conf_file_name)

    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

   
    # Initialize data storage
    base_pos_all, base_bearing_all = [], []#

    # initializing MPC
     # Define the matrices
    num_states = 3
    num_controls = 2
   
    
    # Measuring all the state
    
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    # update A,B,C matrices
    # TODO provide state_x_for_linearization,cur_u_for_linearization to linearize the system
    # you can linearize around the final state and control of the robot (everything zero)
    # or you can linearize around the current state and control of the robot
    # in the second case case you need to update the matrices A and B at each time step
    # and recall everytime the method updateSystemMatrices
    init_pos  = np.array([2.0, 3.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    # Define the cost matrices
    Qcoeff = np.array([1, 1, 0])
    Rcoeff = [0.1,0.0005]
    regulator.setCostMatrices(Qcoeff,Rcoeff)
   

    u_mpc = np.zeros(num_controls)

    ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    v_linear = 0.0
    v_angular = 0.0
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)
    
    while True:
        # True state propagation (with process noise)
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Measurements of the current state (real measurements with noise)
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        y = landmark_range_observations(base_pos)  # 获取观测值
        
        # 更新EKF预测和更新步骤 (此处应添加EKF更新代码，如果适用)
        
        # 当前状态用于线性化
        cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        cur_u_for_linearization = u_mpc

        # 更新系统矩阵A和B，以当前状态为线性化点
        regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)

        # 计算用于MPC优化的矩阵
        # S_bar, T_bar, Q_bar, R_bar, P = regulator.propagation_model_regulator_fixed_std()
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)

        # # 构建匹配 H 形状的终端权重矩阵
        # P_full = np.zeros(H.shape)
        # P_full[-P.shape[0]:, -P.shape[1]:] = P  # 将 P 放置在 H 的右下角

        # 将终端权重P添加到优化问题中
        # H += P
        # H += P_full


        # 初始化x0_mpc为当前状态的拼接形式
        x0_mpc = np.hstack((base_pos[:2], base_bearing_)).flatten()
        
        # 计算最优控制序列
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        u_mpc = u_mpc[:2]  # 只保留第一个控制输入

        # 设置控制指令并转换为轮子速度
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(
            u_mpc[0], u_mpc[1], wheel_base_width, wheel_radius
        )
        print("dif vel",left_wheel_velocity - right_wheel_velocity)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        # 退出逻辑
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # 储存数据用于后续可视化
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)
        current_time += time_step
        
# 可视化最终路径和角度的轨迹

    # while True:


    #     # True state propagation (with process noise)
    #     ##### advance simulation ##################################################################
    #     sim.Step(cmd, "torque")
    #     time_step = sim.GetTimeStep()

    #     # Kalman filter prediction
       
    
    #     # Get the measurements from the simulator ###########################################
    #      # measurements of the robot without noise (just for comparison purpose) #############
    #     base_pos_no_noise = sim.bot[0].base_position
    #     base_ori_no_noise = sim.bot[0].base_orientation
    #     base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
    #     base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
    #     base_ang_vel_no_noise  = sim.bot[0].base_ang_vel
    #     # Measurements of the current state (real measurements with noise) ##################################################################
    #     base_pos = sim.GetBasePosition()
    #     base_ori = sim.GetBaseOrientation()
    #     base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
    #     y = landmark_range_observations(base_pos)
    
    #     # Update the filter with the latest observations
        
    
    #     # Get the current state estimate
        

    #     # Figure out what the controller should do next
    #     # MPC section/ low level controller section ##################################################################
       
   
    #     # Compute the matrices needed for MPC optimization
    #     # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
    #     # add this 3 lines if you want to update the A and B matrices at each time step 
    #     #cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
    #     #cur_u_for_linearization = u_mpc
    #     #regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    #     S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    #     H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    #     x0_mpc = np.hstack((base_pos[:2], base_bearing_))
    #     x0_mpc = x0_mpc.flatten()
    #     # Compute the optimal control sequence
    #     H_inv = np.linalg.inv(H)
    #     u_mpc = -H_inv @ F @ x0_mpc
    #     # Return the optimal control sequence
    #     u_mpc = u_mpc[0:num_controls] 
    #     # Prepare control command to send to the low level controller
    #     left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
    #     angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
    #     interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    #     cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)


    #     # Exit logic with 'q' key (unchanged)
    #     keys = sim.GetPyBulletClient().getKeyboardEvents()
    #     qKey = ord('q')
    #     if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
    #         break
        

    #     # Store data for plotting if necessary
    #     base_pos_all.append(base_pos)
    #     base_bearing_all.append(base_bearing_)

    #     # Update current time
    #     current_time += time_step


    # Plotting 
    #add visualization of final x, y, trajectory and theta
        # 可视化最终路径和角度的轨迹
    # 将 `base_pos_all` 和 `base_bearing_all` 转换为 NumPy 数组，方便后续绘图
    base_pos_all = np.array(base_pos_all)
    base_bearing_all = np.array(base_bearing_all)

    # 创建一个图形窗口
    plt.figure(figsize=(10, 8))

    # 绘制机器人位置轨迹
    plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label="Robot Path", color="blue")
    
    # 绘制地标位置
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color='red', label="Landmarks")

    # 可视化方向角（每隔一定步长绘制一个箭头）
    step = max(1, len(base_pos_all) // 20)  # 每20个点绘制一次方向角
    for i in range(0, len(base_pos_all), step):
        x, y = base_pos_all[i, 0], base_pos_all[i, 1]
        theta = base_bearing_all[i]
        plt.arrow(x, y, 0.5 * np.cos(theta), 0.5 * np.sin(theta), head_width=0.3, head_length=0.3, fc='green', ec='green')

    # 设置图表信息
    plt.title("Robot Path with MPC Control and Landmarks")
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

       # 可视化小车的朝向和位置距离原点的变化
    base_pos_all = np.array(base_pos_all)
    base_bearing_all = np.array(base_bearing_all)
    time_steps = np.arange(len(base_pos_all))
    distances_from_origin = np.linalg.norm(base_pos_all, axis=1)

    # 绘制小车的朝向变化
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, base_bearing_all, label="Robot Orientation (radians)", color="blue")
    plt.title("Robot Orientation Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Orientation (radians)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制小车距离原点的变化
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, distances_from_origin, label="Distance from Origin (m)", color="purple")
    plt.title("Robot Distance from Origin Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 显示图像
    plt.show()

    
    
    

if __name__ == '__main__':
    main()