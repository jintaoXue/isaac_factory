import numpy as np
import matplotlib.pyplot as plt

class KfFatigue:
    def __init__(self, dt, num_steps, true_lambda, init_lambda, F0, Q, R, x0, P0):
        self.dt = dt  # 时间间隔
        self.num_steps = num_steps  # 时间步数
        self.true_lambda = true_lambda  # 真实的 lambda 值
        self.F0 = F0  # 初始 F(t) 值
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 测量噪声协方差
        self.x_hat = x0  # 初始状态估计 [F(0), lambda]
        self.P = P0  # 初始协方差矩阵
        self.H = np.array([[1, 0]])  # 测量矩阵 - 只测量F(t)
        # 改进的状态转移矩阵 - 考虑F(t)对lambda的依赖
        self.A = np.array([[1, self.dt * (1 - self.x_hat[0])], 
                          [0, 1]])  # 考虑F(t)对lambda的梯度
        self.prev_time_step = -2
        self.F_estimates = [F0]
        self.lambda_estimates = [init_lambda]
        self.measurements = []
        self.true_F = []
        self.times = []

    def reinit(self, time_step, F0, _lambda):
        self.prev_time_step = time_step
        self.F0 = F0
        self.x_hat = np.array([F0, _lambda])
        # 重新计算状态转移矩阵
        self.A = np.array([[1, self.dt * (1 - F0)], 
                          [0, 1]])

    def predict(self):
        # 动态更新状态转移矩阵 - 改进版本
        # 考虑疲劳模型的非线性特性：dF/dt = lambda * (1 - F)
        # 这导致lambda的估计依赖于F的当前值
        self.A = np.array([[1, self.dt * (1 - self.x_hat[0])], 
                          [0, 1]])
        self.x_pred = self.A @ self.x_hat
        self.P_pred = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x_pred  # 测量残差
        S = self.H @ self.P_pred @ self.H.T + self.R  # 残差协方差
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        self.x_hat = self.x_pred + K @ y  # 更新状态估计
        self.P = (np.eye(2) - K @ self.H) @ self.P_pred  # 更新协方差

    def step(self, measurement, true_F, time_step):
        if time_step != self.prev_time_step + 1:
            self.reinit(time_step, true_F, self.lambda_estimates[-1])
            return
        self.predict()
        z = np.array(measurement)
        self.update(z)
        self.F_estimates.append(self.x_hat[0])
        # self.x_hat[0] = measurement
        self.lambda_estimates.append(self.x_hat[1])
        self.measurements.append(measurement)
        self.true_F.append(true_F)
        self.times.append(self.prev_time_step)
        self.prev_time_step = time_step

    def run(self):
        # np.random.seed(42)
        self.times = np.arange(0, self.num_steps * self.dt, self.dt)
        self.true_F = 1 - (1 - self.F0) * np.exp(-self.true_lambda * self.times)
        self.measurements = self.true_F + np.random.normal(0, np.sqrt(self.R[0, 0]), size=self.true_F.shape)
        F_estimates = []
        lambda_estimates = []
        for i in range(self.num_steps):
            self.predict()
            z = np.array([self.measurements[i]])
            self.update(z)
            F_estimates.append(self.x_hat[0])
            lambda_estimates.append(self.x_hat[1])
        return self.times, F_estimates, lambda_estimates

    def plot_results(self, times, F_estimates, lambda_estimates):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(times, self.true_F, label='True F(t)', color='blue')
        plt.plot(times, self.measurements, 'x', label='Measurements', color='red', alpha=0.5)
        plt.plot(times, F_estimates, label='Estimated F(t)', color='green')
        plt.xlabel('Time')
        plt.ylabel('F(t)')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(times, lambda_estimates, label='Estimated lambda', color='green')
        plt.axhline(y=self.true_lambda, color='blue', linestyle='--', label='True lambda')
        plt.xlabel('Time')
        plt.ylabel('lambda')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('kf_class_results.png')

class KfRecover:
    def __init__(self, dt, num_steps, true_mu, init_mu, R0, Q, R, x0, P0):
        self.dt = dt  # 时间间隔
        self.num_steps = num_steps  # 时间步数
        self.true_mu = true_mu  # 真实的 mu 值
        self.R0 = R0  # 初始 R(t) 值
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 测量噪声协方差
        self.x_hat = x0  # 初始状态估计 [R(0), mu]
        self.P = P0  # 初始协方差矩阵
        self.H = np.array([[1, 0]])  # 测量矩阵 - 只测量R(t)
        # 改进的状态转移矩阵 - 考虑R(t)对mu的依赖
        self.A = np.array([[1, -self.dt * self.x_hat[0]], 
                          [0, 1]])  # 考虑R(t)对mu的梯度
        self.prev_time_step = -2
        self.R_estimates = [R0]
        self.mu_estimates = [init_mu]
        self.measurements = []
        self.true_R = []
        self.times = []

    def reinit(self, time_step, R0, mu):
        self.prev_time_step = time_step
        self.R0 = R0
        self.x_hat = np.array([R0, mu])
        # 重新计算状态转移矩阵
        self.A = np.array([[1, -self.dt * R0], 
                          [0, 1]])

    def predict(self):
        # 动态更新状态转移矩阵 - 改进版本
        # 考虑恢复模型的非线性特性：dR/dt = -mu * R
        # 这导致mu的估计依赖于R的当前值
        self.A = np.array([[1, -self.dt * self.x_hat[0]], 
                          [0, 1]])
        self.x_pred = self.A @ self.x_hat
        self.P_pred = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x_pred  # 测量残差
        S = self.H @ self.P_pred @ self.H.T + self.R  # 残差协方差
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        self.x_hat = self.x_pred + K @ y  # 更新状态估计
        self.P = (np.eye(2) - K @ self.H) @ self.P_pred  # 更新协方差

    def step(self, measurement, true_R, time_step):
        if time_step != self.prev_time_step + 1:
            self.reinit(time_step, true_R, self.mu_estimates[-1])
            return
        self.predict()
        z = np.array(measurement)
        self.update(z)
        self.R_estimates.append(self.x_hat[0])
        self.mu_estimates.append(self.x_hat[1])
        self.measurements.append(measurement)
        self.true_R.append(true_R)
        self.times.append(self.prev_time_step)
        self.prev_time_step = time_step

    def run(self):
        # np.random.seed(42)
        self.times = np.arange(0, self.num_steps * self.dt, self.dt)
        self.true_R = self.R0 * np.exp(-self.true_mu * self.times)
        self.measurements = self.true_R + np.random.normal(0, np.sqrt(self.R[0, 0]), size=self.true_R.shape)
        R_estimates = []
        mu_estimates = []
        for i in range(self.num_steps):
            self.predict()
            z = np.array([self.measurements[i]])
            self.update(z)
            R_estimates.append(self.x_hat[0])
            mu_estimates.append(self.x_hat[1])
        return self.times, R_estimates, mu_estimates

    def plot_results(self, times, R_estimates, mu_estimates):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(times, self.true_R, label='True R(t)', color='blue')
        plt.plot(times, self.measurements, 'x', label='Measurements', color='red', alpha=0.5)
        plt.plot(times, R_estimates, label='Estimated R(t)', color='green')
        plt.xlabel('Time')
        plt.ylabel('R(t)')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(times, mu_estimates, label='Estimated mu', color='green')
        plt.axhline(y=self.true_mu, color='blue', linestyle='--', label='True mu')
        plt.xlabel('Time')
        plt.ylabel('mu')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('kf_exponential_decay_results.png')

# 使用示例
if __name__ == "__main__":
    # Fatigue KF - 改进版本
    dt = 0.1
    num_steps = 100
    true_lambda = 0.5
    F0 = 0.0
    # 增加lambda的过程噪声，使其更容易被估计
    Q = np.diag([0.01, 0.01])  # 将lambda的过程噪声从0.0001增加到0.01
    R = np.array([[0.1]])
    x0 = np.array([F0, 0.1])
    # 增加lambda的初始不确定性
    P0 = np.diag([1.0, 10.0])  # 将lambda的初始协方差从1.0增加到10.0
    kf_fatigue = KfFatigue(dt, num_steps, true_lambda, F0, Q, R, x0, P0)
    times, F_estimates, lambda_estimates = kf_fatigue.run()
    kf_fatigue.plot_results(times, F_estimates, lambda_estimates)

    # Recover KF - 改进版本
    dt = 0.1
    num_steps = 100
    true_mu = 0.5
    R0 = 1.0
    # 增加mu的过程噪声
    Q = np.diag([0.01, 0.01])  # 将mu的过程噪声从0.0001增加到0.01
    R = np.array([[0.1]])
    x0 = np.array([R0, 0.1])
    # 增加mu的初始不确定性
    P0 = np.diag([1.0, 10.0])  # 将mu的初始协方差从1.0增加到10.0
    kf_recover = KfRecover(dt, num_steps, true_mu, R0, Q, R, x0, P0)
    times, R_estimates, mu_estimates = kf_recover.run()
    kf_recover.plot_results(times, R_estimates, mu_estimates) 