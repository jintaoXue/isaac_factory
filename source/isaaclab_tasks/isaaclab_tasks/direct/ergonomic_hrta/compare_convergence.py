import numpy as np
import matplotlib.pyplot as plt
from pf_filter import ParticleFilter
from pf_filter_improved import ImprovedParticleFilter
import time

def compare_convergence_performance():
    """比较原始和改进粒子滤波器的收敛性能"""
    
    # 参数设置
    dt = 0.1
    num_steps = 100
    true_lambda = 0.5
    F0 = 0.0
    num_particles = 1000
    sigma_w = 0.001
    sigma_v = 0.001
    
    print("开始收敛性能比较...")
    
    # 测试原始粒子滤波器
    print("测试原始粒子滤波器...")
    start_time = time.time()
    pf_original = ParticleFilter(dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v,
                               lamda_init=0.5, upper_bound=2.0, lower_bound=0.0, resample_method='systematic')
    times_orig, F_estimates_orig, lambda_estimates_orig = pf_original.run()
    time_original = time.time() - start_time
    
    # 测试改进的粒子滤波器
    print("测试改进的粒子滤波器...")
    start_time = time.time()
    pf_improved = ImprovedParticleFilter(dt, num_steps, true_lambda, F0, num_particles, sigma_w, sigma_v,
                                       lamda_init=0.5, upper_bound=2.0, lower_bound=0.0, resample_method='systematic')
    times_imp, F_estimates_imp, lambda_estimates_imp = pf_improved.run()
    time_improved = time.time() - start_time
    
    # 计算收敛指标
    def calculate_convergence_metrics(lambda_estimates, true_lambda):
        """计算收敛指标"""
        if len(lambda_estimates) < 10:
            return None
        
        # 计算最后20个估计的统计量
        recent_estimates = lambda_estimates[-20:]
        mean_est = np.mean(recent_estimates)
        std_est = np.std(recent_estimates)
        variance = np.var(recent_estimates)
        
        # 收敛指标
        convergence_metric = variance / (true_lambda ** 2)
        relative_error = abs(mean_est - true_lambda) / true_lambda * 100
        
        return {
            'mean': mean_est,
            'std': std_est,
            'variance': variance,
            'convergence_metric': convergence_metric,
            'relative_error': relative_error
        }
    
    # 计算指标
    metrics_orig = calculate_convergence_metrics(lambda_estimates_orig, true_lambda)
    metrics_imp = calculate_convergence_metrics(lambda_estimates_imp, true_lambda)
    
    # 打印结果
    print("\n=== 收敛性能比较结果 ===")
    print(f"原始粒子滤波器运行时间: {time_original:.4f} 秒")
    print(f"改进粒子滤波器运行时间: {time_improved:.4f} 秒")
    
    if metrics_orig and metrics_imp:
        print(f"\n原始粒子滤波器:")
        print(f"  Lambda估计均值: {metrics_orig['mean']:.4f}")
        print(f"  Lambda估计标准差: {metrics_orig['std']:.4f}")
        print(f"  收敛指标: {metrics_orig['convergence_metric']:.6f}")
        print(f"  相对误差: {metrics_orig['relative_error']:.2f}%")
        
        print(f"\n改进粒子滤波器:")
        print(f"  Lambda估计均值: {metrics_imp['mean']:.4f}")
        print(f"  Lambda估计标准差: {metrics_imp['std']:.4f}")
        print(f"  收敛指标: {metrics_imp['convergence_metric']:.6f}")
        print(f"  相对误差: {metrics_imp['relative_error']:.2f}%")
        
        # 计算改进程度
        convergence_improvement = (metrics_orig['convergence_metric'] - metrics_imp['convergence_metric']) / metrics_orig['convergence_metric'] * 100
        error_improvement = (metrics_orig['relative_error'] - metrics_imp['relative_error']) / metrics_orig['relative_error'] * 100
        
        print(f"\n改进程度:")
        print(f"  收敛指标改进: {convergence_improvement:.2f}%")
        print(f"  相对误差改进: {error_improvement:.2f}%")
    
    # 绘制比较图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # F(t) 估计比较
    axes[0, 0].plot(times_orig, pf_original.measurements, 'x', label='Measurements', color='gray', alpha=0.5, markersize=2)
    axes[0, 0].plot(times_orig, F_estimates_orig, label='Original PF', color='blue', linewidth=2)
    axes[0, 0].plot(times_imp, F_estimates_imp, label='Improved PF', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('F(t)')
    axes[0, 0].set_title('F(t) Estimation Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Lambda 估计比较
    axes[0, 1].plot(times_orig, lambda_estimates_orig[1:], label='Original PF', color='blue', linewidth=2)
    axes[0, 1].plot(times_imp, lambda_estimates_imp[1:], label='Improved PF', color='red', linewidth=2)
    axes[0, 1].axhline(y=true_lambda, color='black', linestyle='--', label='True lambda')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('lambda')
    axes[0, 1].set_title('Lambda Estimation Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 有效样本大小比较
    if hasattr(pf_improved, 'ess_history') and len(pf_improved.ess_history) > 0:
        ess_times = range(len(pf_improved.ess_history))
        axes[1, 0].plot(ess_times, pf_improved.ess_history, 'r-', linewidth=2, label='Improved PF')
        axes[1, 0].axhline(y=num_particles * 0.3, color='gray', linestyle='--', label='Resampling threshold')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Effective Sample Size')
        axes[1, 0].set_title('Effective Sample Size (Improved PF)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 收敛指标比较
    if hasattr(pf_improved, 'convergence_history') and len(pf_improved.convergence_history) > 0:
        conv_times = range(len(pf_improved.convergence_history))
        axes[1, 1].plot(conv_times, pf_improved.convergence_history, 'g-', linewidth=2, label='Improved PF')
        axes[1, 1].axhline(y=0.01, color='gray', linestyle='--', label='Convergence threshold')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Convergence Metric')
        axes[1, 1].set_title('Convergence Analysis (Improved PF)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_different_scenarios():
    """测试不同场景下的收敛性能"""
    
    scenarios = [
        {
            'name': '低噪声场景',
            'sigma_w': 0.0005,
            'sigma_v': 0.0005,
            'num_particles': 500
        },
        {
            'name': '高噪声场景',
            'sigma_w': 0.002,
            'sigma_v': 0.002,
            'num_particles': 2000
        },
        {
            'name': '少粒子场景',
            'sigma_w': 0.001,
            'sigma_v': 0.001,
            'num_particles': 200
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n测试场景: {scenario['name']}")
        
        # 参数设置
        dt = 0.1
        num_steps = 100
        true_lambda = 0.5
        F0 = 0.0
        
        # 测试原始粒子滤波器
        pf_original = ParticleFilter(dt, num_steps, true_lambda, F0, scenario['num_particles'], 
                                   scenario['sigma_w'], scenario['sigma_v'],
                                   lamda_init=0.5, upper_bound=2.0, lower_bound=0.0, 
                                   resample_method='systematic')
        times_orig, F_estimates_orig, lambda_estimates_orig = pf_original.run()
        
        # 测试改进的粒子滤波器
        pf_improved = ImprovedParticleFilter(dt, num_steps, true_lambda, F0, scenario['num_particles'], 
                                           scenario['sigma_w'], scenario['sigma_v'],
                                           lamda_init=0.5, upper_bound=2.0, lower_bound=0.0, 
                                           resample_method='systematic')
        times_imp, F_estimates_imp, lambda_estimates_imp = pf_improved.run()
        
        # 计算收敛指标
        def calculate_metrics(lambda_estimates):
            if len(lambda_estimates) < 10:
                return None
            recent_estimates = lambda_estimates[-20:]
            mean_est = np.mean(recent_estimates)
            variance = np.var(recent_estimates)
            convergence_metric = variance / (true_lambda ** 2)
            relative_error = abs(mean_est - true_lambda) / true_lambda * 100
            return convergence_metric, relative_error
        
        metrics_orig = calculate_metrics(lambda_estimates_orig)
        metrics_imp = calculate_metrics(lambda_estimates_imp)
        
        if metrics_orig and metrics_imp:
            conv_orig, error_orig = metrics_orig
            conv_imp, error_imp = metrics_imp
            
            improvement_conv = (conv_orig - conv_imp) / conv_orig * 100
            improvement_error = (error_orig - error_imp) / error_orig * 100
            
            results.append({
                'scenario': scenario['name'],
                'original_conv': conv_orig,
                'improved_conv': conv_imp,
                'original_error': error_orig,
                'improved_error': error_imp,
                'conv_improvement': improvement_conv,
                'error_improvement': improvement_error
            })
            
            print(f"  原始收敛指标: {conv_orig:.6f}")
            print(f"  改进收敛指标: {conv_imp:.6f}")
            print(f"  收敛改进: {improvement_conv:.2f}%")
            print(f"  原始相对误差: {error_orig:.2f}%")
            print(f"  改进相对误差: {error_imp:.2f}%")
            print(f"  误差改进: {improvement_error:.2f}%")
    
    # 绘制场景比较结果
    if results:
        scenarios_names = [r['scenario'] for r in results]
        conv_improvements = [r['conv_improvement'] for r in results]
        error_improvements = [r['error_improvement'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 收敛指标改进
        bars1 = ax1.bar(scenarios_names, conv_improvements, color='skyblue')
        ax1.set_ylabel('Convergence Improvement (%)')
        ax1.set_title('Convergence Metric Improvement')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, conv_improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 相对误差改进
        bars2 = ax2.bar(scenarios_names, error_improvements, color='lightcoral')
        ax2.set_ylabel('Error Improvement (%)')
        ax2.set_title('Relative Error Improvement')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars2, error_improvements):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("开始收敛性能比较...")
    compare_convergence_performance()
    
    print("\n开始不同场景测试...")
    test_different_scenarios() 