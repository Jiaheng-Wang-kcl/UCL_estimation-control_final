#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Helper function for wrapping angles between -pi and pi
def wrap_angle(angle): 
    return np.arctan2(np.sin(angle), np.cos(angle))

class FilterConfiguration(object):
    def __init__(self):
        # # Process and measurement noise covariance matrices
        # self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # # Measurement noise variance (range measurements)
        # self.W_range = 0.5 ** 2
        # self.W_bearing = (np.pi * 0.5 / 180.0) ** 2

        # # Process and measurement noise covariance matrices
        # self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # # Measurement noise variance (range measurements)
        # self.W_range = 0.5 ** 2
        # self.W_bearing = (np.pi * 0.5 / 180.0) ** 2

        # for task 3
        self.V = np.diag([0.15, 0.15, 0.15]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements)
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2
        
        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2


class Map(object):
    def __init__(self):
        # self.landmarks = np.array([
        #     [5, 10],
        #     [15, 5],
        #     [10, 15]
        # ])

        # 3 x 3 landmarks
        self.landmarks = np.array([
            [-20, -10], [0, -10], [20, -10],   
            [-20, 7.5], [0, 7.5], [20, 7.5],  
            [-20, 25],  [0, 25],  [20, 25]    
        ])

        # # 5 x 5 landmarks
        # self.landmarks = np.array([
        #     [-20, -10], [-10, -10], [0, -10], [10, -10], [20, -10],   
        #     [-20, -1.25], [-10, -1.25], [0, -1.25], [10, -1.25], [20, -1.25],  
        #     [-20, 8.75], [-10, 8.75], [0, 8.75], [10, 8.75], [20, 8.75],  
        #     [-20, 17.5], [-10, 17.5], [0, 17.5], [10, 17.5], [20, 17.5],  
        #     [-20, 25], [-10, 25], [0, 25], [10, 25], [20, 25]   
        # ])

        # # 7 x 7 landmarks
        # self.landmarks = np.array([
        #     [-20, -10], [-13.33, -10], [-6.67, -10], [0, -10], [6.67, -10], [13.33, -10], [20, -10],  
        #     [-20, -4.17], [-13.33, -4.17], [-6.67, -4.17], [0, -4.17], [6.67, -4.17], [13.33, -4.17], [20, -4.17],  
        #     [-20, 1.67], [-13.33, 1.67], [-6.67, 1.67], [0, 1.67], [6.67, 1.67], [13.33, 1.67], [20, 1.67], 
        #     [-20, 7.5], [-13.33, 7.5], [-6.67, 7.5], [0, 7.5], [6.67, 7.5], [13.33, 7.5], [20, 7.5],  
        #     [-20, 13.33], [-13.33, 13.33], [-6.67, 13.33], [0, 13.33], [6.67, 13.33], [13.33, 13.33], [20, 13.33],  
        #     [-20, 19.17], [-13.33, 19.17], [-6.67, 19.17], [0, 19.17], [6.67, 19.17], [13.33, 19.17], [20, 19.17],  
        #     [-20, 25], [-13.33, 25], [-6.67, 25], [0, 25], [6.67, 25], [13.33, 25], [20, 25] 
        # ])        

        # # 生成 10 x 10 的地标矩阵
        # x_values = np.linspace(-25, 25, 10)
        # y_values = np.linspace(-25, 25, 10)
        # self.landmarks = np.array([[x, y] for x in x_values for y in y_values])

class RobotEstimator(object):

    def __init__(self, filter_config, map):
        # Variables which will be used
        self._config = filter_config
        self._map = map

    # This nethod MUST be called to start the filter
    def start(self):
        self._t = 0
        self._set_estimate_to_initial_conditions()

    def set_control_input(self, u):
        self._u = u

    # Predict to the time. The time is fed in to
    # allow for variable prediction intervals.
    def predict_to(self, time):
        # What is the time interval length?
        dt = time - self._t

        # Store the current time
        self._t = time

        # print("check dt",dt)
        # Now predict over a duration dT
        self._predict_over_dt(dt)

    # Return the estimate and its covariance
    def estimate(self):
        return self._x_est, self._Sigma_est

    # This method gets called if there are no observations
    def copy_prediction_to_estimate(self):
        self._x_est = self._x_pred
        self._Sigma_est = self._Sigma_pred

    # This method sets the filter to the initial state
    def _set_estimate_to_initial_conditions(self):
        # Initial estimated state and covariance
        self._x_est = self._config.x0
        self._Sigma_est = self._config.Sigma0

    # Predict to the time
    def _predict_over_dt(self, dt):
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._config.V

        # Predict the new state
        self._x_pred = self._x_est + np.array([
            v_c * np.cos(self._x_est[2]) * dt,
            v_c * np.sin(self._x_est[2]) * dt,
            omega_c * dt
        ])
        self._x_pred[-1] = np.arctan2(np.sin(self._x_pred[-1]),
                                      np.cos(self._x_pred[-1]))

        # Predict the covariance
        A = np.array([
            [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
            [0, 1,  v_c * np.cos(self._x_est[2]) * dt],
            [0, 0, 1]
        ])

        self._kf_predict_covariance(A, self._config.V * dt)

    # Predict the EKF covariance; note the mean is
    # totally model specific, so there's nothing we can
    # clearly separate out.
    def _kf_predict_covariance(self, A, V):
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    # Implement the Kalman filter update step.
    def _do_kf_update(self, nu, C, W):

        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)

        # State update
        self._x_est = self._x_pred + K @ nu

        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred

    def update_from_landmark_range_observations(self, y_range):

        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred
        for lm in self._map.landmarks:

            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            y_pred.append(range_pred)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred

        # Since we are oberving a bunch of landmarks
        # build the covariance matrix. Note you could
        # swap this to just calling the ekf update call
        # multiple times, once for each observation,
        # as well
        W_landmarks = self._config.W_range * np.eye(len(self._map.landmarks))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))

    def update_from_range_bearing_observations(self, y_range_bearing):
        # 初始化观测值和雅可比矩阵
        y_pred = []
        C = []
        x_pred = self._x_pred

        # 遍历所有地标，计算预测值和雅可比矩阵
        for lm in self._map.landmarks:
            # 计算预测的距离和方位角
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            bearing_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]
            bearing_pred = wrap_angle(bearing_pred)
            
            # 检查预测值的顺序和数值
            y_pred.extend([range_pred, bearing_pred])

            # 构建观测雅可比矩阵的一部分
            C_range_bearing = np.array([
                [-dx_pred / range_pred, -dy_pred / range_pred, 0],
                [dy_pred / (range_pred**2), -dx_pred / (range_pred**2), -1]
            ])
            C.append(C_range_bearing)

        # 将 y_pred 和 C 转换为数组形式，y_pred 包含所有地标的预测观测值
        y_pred = np.array(y_pred)
        C = np.vstack(C)  # 将所有地标的雅可比矩阵部分拼接到一起

        # print(f"y_range_bearing shape: {y_range_bearing.flatten().shape}, y_pred shape: {y_pred.shape}")

        # 计算创新向量
        nu = y_range_bearing.flatten() - y_pred
        for i in range(1, len(nu), 2):  # 对方位角差进行包裹
            nu[i] = wrap_angle(nu[i])
        # 设置观测噪声协方差矩阵
        W = np.kron(np.eye(len(self._map.landmarks)), np.diag([self._config.W_range, self._config.W_bearing]))

        # # 输出调试信息
        # print(f"Predicted measurements (y_pred): {y_pred}")
        # print(f"Actual measurements (y_range_bearing): {y_range_bearing.flatten()}")
        # print(f"Innovation vector (nu): {nu}")
        # print(f"Observation Jacobian (C): {C}")
        # print(f"Observation noise covariance (W): {W}")

        # 调用通用的 EKF 更新函数来计算卡尔曼增益和更新状态
        self._do_kf_update(nu, C, W)

        # 确保角度在 [-π, π] 范围内
        self._x_est[-1] = wrap_angle(self._x_est[-1])
