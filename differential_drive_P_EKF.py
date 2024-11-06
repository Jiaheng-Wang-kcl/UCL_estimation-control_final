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

    # Data recording
    true_state_all = []  # True state
    est_state_all = []  # Estimated state
    obs_pos_all = []  # Observed state

    # Main control loop
    while True:
        sim.Step(cmd, "torque")
        
        # EKF control input and prediction
        estimator.set_control_input(u_mpc)
        estimator.predict_to(time)
        
        # Get new observations and update EKF
        obs_pos = sim.GetBasePosition()
        obs_ori = sim.GetBaseOrientation()
        obs_bearing = quaternion2bearing(obs_ori[3], obs_ori[0], obs_ori[1], obs_ori[2])
        obs_base_position = [obs_pos[0], obs_pos[1], obs_bearing]
        y = generate_range_bearing_observations(obs_base_position)
        estimator.update_from_range_bearing_observations(y)

        x_est, Sigma_est = estimator.estimate()

        # Compute optimal control sequence
        x0_mpc = np.hstack((x_est[0], x_est[1], x_est[2]))
        cur_state_x_for_linearization = [x_est[0], x_est[1], x_est[2]]
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        u_mpc = u_mpc[:2]

        # Convert control input to wheel velocities
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(
            u_mpc[0], u_mpc[1], wheel_base_width, wheel_radius
        )
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity,
                                                left_wheel_velocity, right_wheel_velocity])
        cmd.SetControlCmd(angular_wheels_velocity_cmd, ["velocity"] * 4)
        time += time_step

        # Record positions for plotting
        true_pos = sim.bot[0].base_position.copy()
        true_ori = sim.bot[0].base_orientation.copy()
        true_bearing = quaternion2bearing(true_ori[3], true_ori[0], true_ori[1], true_ori[2])
        true_state = [true_pos[0], true_pos[1]]

        true_state_all.append(true_state)
        est_state_all.append([x_est[0], x_est[1]])
        obs_pos_all.append([obs_pos[0], obs_pos[1]])

        # Exit logic
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

    # Convert to arrays for plotting
    true_state_all = np.array(true_state_all)
    est_state_all = np.array(est_state_all)
    obs_pos_all = np.array(obs_pos_all)

    # Plot 1: Robot true path, observed path, and estimated path
    plt.figure(figsize=(10, 6))
    plt.plot(true_state_all[:, 0], true_state_all[:, 1], label='True Path', color='blue')
    plt.plot(est_state_all[:, 0], est_state_all[:, 1], label='Estimated Path', color='orange', alpha=0.5)
    plt.plot(obs_pos_all[:, 0], obs_pos_all[:, 1], label='Observed Path', color='green', alpha=0.5)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color='red', label='Landmarks')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title('Robot True Path, Observation Path, and Estimation Path')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # Plot 2: Ideal grid path from start to end with marked start and end points
    plt.figure(figsize=(10, 6))
    plt.plot(true_state_all[:, 0], true_state_all[:, 1], color='blue', label="Actual Path")
    plt.scatter([2], [3], color="purple", s=100, marker="o", label="Start (2, 3)")
    plt.scatter([0], [0], color="red", s=100, marker="X", label="End (0, 0)")
    grid_x, grid_y = np.linspace(2, 0, 100), np.linspace(3, 0, 100)
    plt.plot(grid_x, grid_y, color="gray", linestyle="--", label="Ideal Path (Grid)")
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.title("Ideal Path from Start to End")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    main()
