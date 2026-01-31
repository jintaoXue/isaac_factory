import numpy as np
import matplotlib.pyplot as plt

class ParticleFilter:
    def __init__(self, dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v, lamda_init, upper_bound, lower_bound, resample_method='systematic'):
        self.dt = dt  # 时间间隔
        self.num_steps = num_steps  # 时间步数
        self.true_lambda = true_lambda  # 真实的 lambda 值
        self.F0 = F0  # 初始 F(t) 值
        self.num_particles = num_particles  # 粒子数量
        self.sigma_w = sigma_w  # 过程噪声标准差
        self.sigma_v = sigma_v  # 测量噪声标准差
        self.resample_method = resample_method  # 重采样方法

        # 初始化粒子
        # np.random.seed(42)
        self.l_bound = lower_bound
        self.u_bound = upper_bound
        # self.particles = np.random.uniform(lower_bound, upper_bound, num_particles)  # 初始粒子分布
        self.particles = np.linspace(lower_bound, upper_bound, num_particles)  # 初始粒子分布
        self.weights = np.ones(num_particles) / num_particles  # 初始权重
        lamda_init = np.sum(self.particles * self.weights)
        self.prev_time_step = -2
        self.F_estimates = []
        self.lambda_estimates = [lamda_init]
        self.measurements = [F0]
        self.true_F = []
        self.times = []
    
    def reinit(self, time_step, F0):
        
        self.F_estimates.append(F0)
        self.measurements.append(F0)
        if len(self.lambda_estimates) > 0:
            self.lambda_estimates.append(self.lambda_estimates[-1])
        self.prev_time_step = time_step
        self.F0 = F0

    def state_transition(self):
        self.particles = self.particles + np.random.normal(0, self.sigma_w, self.num_particles)
        # self.particles = np.clip(self.particles, self.l_bound, self.u_bound)

    def update_weights(self, F_prev, measurement_t):
        F_pred = F_prev + (1 - F_prev) * (1 - np.exp(-self.particles * self.dt))
        # likelihood = np.exp(-0.5 * (measurement_t - F_pred)**2 / self.sigma_v**2)
        errors = np.abs(measurement_t - F_pred)
        error_min = np.min(errors)
        error_max = np.max(errors)
        #误差归一化到[0,1]
        normalized_errors = (errors - error_min) / (error_max - error_min)
        #使用指数函数放大误差
        likelihood = np.exp(-10 * normalized_errors**2)
        likelihood = likelihood/ sum(likelihood)
        # likelihood = np.exp(-0.5 * (measurement_t - F_pred)**2)
        # likelihood = likelihood/ sum(likelihood)
        likelihood = np.clip(likelihood, 0, 1)
        # if likelihood.any():
        self.weights = self.weights * likelihood
        self.weights = self.weights / np.sum(self.weights)  # 归一化权重

    def effective_sample_size(self):
        """计算有效样本大小"""
        return 1.0 / np.sum(self.weights**2)

    def systematic_resample(self):
        """系统重采样"""
        # 计算累积权重
        cumsum_weights = np.cumsum(self.weights)
        
        # 生成均匀分布的随机起始点
        u = np.random.uniform(0, 1.0 / self.num_particles)
        
        # 生成系统重采样的索引
        indices = np.zeros(self.num_particles, dtype=int)
        j = 0
        for i in range(self.num_particles):
            u_i = u + i / self.num_particles
            while u_i > cumsum_weights[j]:
                j += 1
            indices[i] = j
        
        return indices

    def stratified_resample(self):
        """分层重采样"""
        # 计算累积权重
        cumsum_weights = np.cumsum(self.weights)
        
        # 生成分层随机数
        u = np.random.uniform(0, 1.0 / self.num_particles, self.num_particles)
        u += np.arange(self.num_particles) / self.num_particles
        
        # 生成重采样索引
        indices = np.zeros(self.num_particles, dtype=int)
        j = 0
        for i in range(self.num_particles):
            while u[i] > cumsum_weights[j]:
                j += 1
            indices[i] = j
        
        return indices

    def residual_resample(self):
        """残差重采样"""
        # 计算期望的粒子数量
        expected_counts = self.weights * self.num_particles
        
        # 确定性重采样部分
        deterministic_counts = np.floor(expected_counts).astype(int)
        residual_weights = expected_counts - deterministic_counts
        
        # 归一化残差权重
        residual_weights = residual_weights / np.sum(residual_weights)
        
        # 随机重采样剩余部分
        remaining_particles = self.num_particles - np.sum(deterministic_counts)
        random_indices = np.random.choice(self.num_particles, remaining_particles, p=residual_weights)
        
        # 构建最终索引
        indices = []
        for i in range(self.num_particles):
            indices.extend([i] * deterministic_counts[i])
        indices.extend(random_indices)
        
        return np.array(indices)

    def resample(self):
        """改进的重采样方法"""
        N_eff = self.effective_sample_size()
        
        # 只有当有效样本大小小于阈值时才重采样
        if N_eff < self.num_particles / 2:
            if self.resample_method == 'systematic':
                indices = self.systematic_resample()
            elif self.resample_method == 'stratified':
                indices = self.stratified_resample()
            elif self.resample_method == 'residual':
                indices = self.residual_resample()
            else:  # 默认使用系统重采样
                indices = self.systematic_resample()
            
            # 重采样粒子和权重
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
            
            # 添加小的随机扰动以避免粒子退化
            self.particles += np.random.normal(0, self.sigma_w * 0.1, self.num_particles)
            self.particles = np.clip(self.particles, self.l_bound, self.u_bound)

    def step(self, measurement, true_F, time_step):
        self.true_F.append(true_F)
        self.times.append(time_step)
        if time_step != self.prev_time_step + 1:
            self.reinit(time_step, true_F)
            return
        
        self.state_transition()
        # 更新权重
        # F_prev = self.F_estimates[-1]
        F_prev = self.measurements[-1]
        self.update_weights(F_prev, measurement)
        self.measurements.append(measurement)

        # 重采样
        # self.resample()

        # 估计 lambda
        lambda_est = np.sum(self.particles * self.weights)
        self.lambda_estimates.append(lambda_est)

        # 估计 F(t) 用于下一时刻
        F_est = measurement + (1 - measurement) * (1 - np.exp(-lambda_est * self.dt))
        self.F_estimates.append(F_est)

    def run(self):
        F_estimates = [self.F0]  # F(t) 的估计值列表，初始值为 F0
        lambda_estimates = []
        # 生成模拟数据
        self.times = np.arange(0, self.num_steps * self.dt, self.dt)
        self.true_F = 1 - (1 - self.F0) * np.exp(-self.true_lambda * self.times)  # 真实的 F(t)
        self.measurements = self.true_F + np.random.normal(0, self.sigma_v, size=self.true_F.shape)  # 带噪声的测量

        for t in range(1, self.num_steps):
            # 预测步骤
            self.state_transition()

            # 更新权重
            F_prev = F_estimates[-1]
            self.update_weights(F_prev, self.measurements[t])

            # 重采样
            self.resample()

            # 估计 lambda
            lambda_est = np.sum(self.particles * self.weights)
            lambda_estimates.append(lambda_est)

            # 估计 F(t) 用于下一时刻
            F_est = F_prev + (1 - F_prev) * (1 - np.exp(-lambda_est * self.dt))
            F_estimates.append(F_est)

        return self.times, F_estimates, lambda_estimates

    def plot_results(self, times, F_estimates, lambda_estimates, name = ''):
        plt.figure(figsize=(10, 6))

        # 绘制 F(t)
        plt.subplot(2, 1, 1)
        # plt.plot(times, self.true_F, label='True F(t)', color='blue')
        plt.plot(times, self.measurements, 'x', label='Measurements', color='red', alpha=0.5)
        plt.plot(times, F_estimates, label='Estimated F(t)', color='green')
        plt.xlabel('Time')
        plt.ylabel('F(t)')
        plt.legend()
        plt.grid(True)

        # 绘制 lambda
        plt.subplot(2, 1, 2)
        plt.plot(times, lambda_estimates[1:], label='Estimated lambda', color='green')
        plt.axhline(y=self.true_lambda, color='blue', linestyle='--', label='True lambda')
        plt.xlabel('Time')
        plt.ylabel('lambda')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(name + '_pf_results.png')



class RecParticleFilter(ParticleFilter):

    def update_weights(self, F_prev, measurement_t):
        F_pred = F_prev*np.exp(-self.particles * self.dt)
        # likelihood = np.exp(-0.5 * (measurement_t - F_pred)**2 / self.sigma_v**2)
        errors = np.abs(measurement_t - F_pred)
        error_min = np.min(errors)
        error_max = np.max(errors)
        if error_max == error_min:
            return
        #误差归一化到[0,1]
        normalized_errors = (errors - error_min) / (error_max - error_min)
        #使用指数函数放大误差
        likelihood = np.exp(-100 * normalized_errors**2)
        likelihood = likelihood/ sum(likelihood)
        # likelihood = np.exp(-0.5 * (measurement_t - F_pred)**2)
        # likelihood = likelihood/ sum(likelihood)
        likelihood = np.clip(likelihood, 0, 1)
        # if likelihood.any():
        self.weights = self.weights * likelihood
        self.weights = self.weights / np.sum(self.weights)  # 归一化权重

# 使用示例
if __name__ == "__main__":
    # 参数设置
    dt = 0.1
    num_steps = 100
    true_lambda = 0.5
    F0 = 0.0
    num_particles = 1000
    sigma_w = 0.001
    sigma_v = 0.001

    # 创建粒子滤波实例，可以选择不同的重采样方法
    # resample_method 可以是 'systematic', 'stratified', 'residual'
    pf = ParticleFilter(dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v, 
                       lamda_init=0.5, upper_bound=2.0, lower_bound=0.0, resample_method='systematic')

    # 运行粒子滤波
    times, F_estimates, lambda_estimates = pf.run()

    # 绘制结果
    pf.plot_results(times, F_estimates, lambda_estimates)