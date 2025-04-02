import mujoco
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from time import sleep
from typing import Callable, Optional
import os
import mujoco.viewer

class RobotController:
    def __init__(self, model_path: str, urdf_path: str):
        """
        初始化机器人控制器
        Args:
            model_path: MuJoCo模型文件路径
            urdf_path: URDF文件路径
        """
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            
        # 加载MuJoCo模型
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {str(e)}")
        
        # 创建Pinocchio模型
        try:
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
            self.pin_data = self.pin_model.createData()
        except Exception as e:
            raise RuntimeError(f"Failed to load Pinocchio model: {str(e)}")
        gravity_vector: np.ndarray = np.array([0, 0, 0])
        self.pin_model.gravity.linear = gravity_vector
        
        # 控制参数
        self.Kp = 100.0  # P增益
        self.Kd = 20.0   # D增益
        
        # 机器人参数
        self.nq = self.pin_model.nq  # 关节数量
        self.joint_limits = {
            'lower': self.pin_model.lowerPositionLimit,
            'upper': self.pin_model.upperPositionLimit,
            'velocity': self.pin_model.velocityLimit,
            'torque': self.pin_model.effortLimit
        }
        
        # 记录数据用于绘图
        self.reset_logs()

    def reset_logs(self):
        """重置记录数据"""
        self.time_log = []
        self.q_desired_log = []
        self.q_actual_log = []
        self.dq_actual_log = []
        self.tau_log = []
        self.error_log = []
        
    def forward_dynamics(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        计算前向动力学
        返回：关节加速度
        """
        pin.computeAllTerms(self.pin_model, self.pin_data, q, dq)
        ddq = pin.aba(self.pin_model, self.pin_data, q, dq, tau)
        return ddq
        
    def inverse_dynamics(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """
        计算逆动力学
        返回：关节力矩
        """
        tau = pin.rnea(self.pin_model, self.pin_data, q, dq, ddq)
        return tau
        
    def clamp_torque(self, tau: np.ndarray) -> np.ndarray:
        """限制力矩在允许范围内"""
        return np.clip(tau, -self.joint_limits['torque'], self.joint_limits['torque'])
        
    def compute_control(self, q_desired: np.ndarray, dq_desired: np.ndarray, 
                       ddq_desired: np.ndarray, q_current: np.ndarray, 
                       dq_current: np.ndarray) -> np.ndarray:
        """计算控制输出"""
        # 检查输入维度
        if any(arr.shape != (self.nq,) for arr in [q_desired, dq_desired, ddq_desired, q_current, dq_current]):
            raise ValueError("Input arrays must match the number of joints")
            
        # 计算误差
        q_error = q_desired - q_current
        dq_error = dq_desired - dq_current
        
        # 计算期望加速度（PD控制）
        ddq = ddq_desired + self.Kp * q_error + self.Kd * dq_error
        
        # 计算所需关节力矩
        #tau = self.inverse_dynamics(q_current, dq_current, ddq)
        tau = self.inverse_dynamics(q_desired, dq_desired, ddq_desired)
        return tau

    def generate_quintic_trajectory(self, t: float, t0: float, tf: float, 
                                  q0: np.ndarray, qf: np.ndarray,
                                  v0: np.ndarray = None, vf: np.ndarray = None,
                                  a0: np.ndarray = None, af: np.ndarray = None) -> tuple:
        """
        生成五次多项式轨迹
        Args:
            t: 当前时间点
            t0: 起始时间
            tf: 终止时间
            q0: 起始位置
            qf: 终止位置
            v0: 起始速度，默认为0
            vf: 终止速度，默认为0
            a0: 起始加速度，默认为0
            af: 终止加速度，默认为0
        Returns:
            tuple: (位置，速度，加速度)
        """
        # 设置默认值
        if v0 is None:
            v0 = np.zeros_like(q0)
        if vf is None:
            vf = np.zeros_like(qf)
        if a0 is None:
            a0 = np.zeros_like(q0)
        if af is None:
            af = np.zeros_like(qf)

        # 归一化时间
        T = tf - t0
        if T <= 0:
            raise ValueError("Final time must be greater than initial time")
        
        # 计算归一化当前时间
        s = (t - t0) / T
        if s < 0:
            return q0, v0, a0
        elif s > 1:
            return qf, vf, af

        # 五次多项式系数计算
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 2, 6, 12, 20]
        ])

        # 计算每个关节的轨迹
        q = np.zeros_like(q0)
        dq = np.zeros_like(q0)
        ddq = np.zeros_like(q0)

        for i in range(len(q0)):
            # 边界条件向量
            b = np.array([
                q0[i],
                v0[i] * T,
                a0[i] * T * T,
                qf[i],
                vf[i] * T,
                af[i] * T * T
            ])

            # 求解系数
            x = np.linalg.solve(A, b)

            # 计算位置
            s_vec = np.array([1, s, s**2, s**3, s**4, s**5])
            q[i] = np.dot(s_vec, x)

            # 计算速度
            ds_vec = np.array([0, 1, 2*s, 3*s**2, 4*s**3, 5*s**4]) / T
            dq[i] = np.dot(ds_vec, x)

            # 计算加速度
            dds_vec = np.array([0, 0, 2, 6*s, 12*s**2, 20*s**3]) / (T * T)
            ddq[i] = np.dot(dds_vec, x)

        return q, dq, ddq

    def generate_trajectory(self, t: float, trajectory_type: str = 'sine') -> tuple:
        """
        生成不同类型的轨迹
        Args:
            t: 时间点
            trajectory_type: 轨迹类型 ('sine', 'circle', 'square', 'quintic')
        Returns:
            tuple: (位置，速度，加速度)
        """
        if trajectory_type == 'quintic':
            # 定义五次多项式轨迹的起始和终止状态
            t0 = 0.0
            tf = 5.0  # 轨迹总时间
            q0 = np.zeros(self.nq)  # 起始位置
            qf = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2,0.1])  # 终止位置
            v0 = np.zeros(self.nq)  # 起始速度
            vf = np.zeros(self.nq)  # 终止速度
            a0 = np.zeros(self.nq)  # 起始加速度
            af = np.zeros(self.nq)  # 终止加速度

            return self.generate_quintic_trajectory(t, t0, tf, q0, qf, v0, vf, a0, af)
            
        elif trajectory_type == 'sine':
            q = np.array([
                0.5 * np.sin(2 * np.pi * t),
                0.3 * np.sin(2 * np.pi * t + np.pi/4),
                0.4 * np.sin(2 * np.pi * t + np.pi/3),
                0.3 * np.sin(2 * np.pi * t + np.pi/2),
                0.2 * np.sin(2 * np.pi * t + 2*np.pi/3),
                0.3 * np.sin(2 * np.pi * t + 3*np.pi/4),
                0.4 * np.sin(2 * np.pi * t + np.pi)
            ])
            dq = np.zeros_like(q)  # 可以计算实际的速度
            ddq = np.zeros_like(q)  # 可以计算实际的加速度
            return q, dq, ddq
        
        elif trajectory_type == 'circle':
            q = np.array([
                0.3 * np.cos(2 * np.pi * t),
                0.3 * np.sin(2 * np.pi * t),
                0.2 * np.cos(2 * np.pi * t),
                0.2 * np.sin(2 * np.pi * t),
                0.1 * np.cos(2 * np.pi * t),
                0.1 * np.sin(2 * np.pi * t),
               # 0.1 * np.cos(2 * np.pi * t)
            ])
            dq = np.zeros_like(q)
            ddq = np.zeros_like(q)
            return q, dq, ddq
            
    def run_simulation(self, trajectory_type: str = 'quintic', 
                      duration: float = 5.0, dt: float = 0.001,
                      render: bool = False):
        """运行仿真"""
        steps = int(duration / dt)
        self.reset_logs()

        self.data.qpos[:7] = np.zeros(self.nq)
        self.data.qvel[:7] = np.zeros(self.nq)
        
        
        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        for i in range(steps):
            t = i * dt
            
            # 获取期望轨迹
            q_d, dq_d, ddq_d = self.generate_trajectory(t, trajectory_type)
            
            # 获取当前状态
            q = self.data.qpos[:7]
            dq = self.data.qvel[:7]
            
            # 计算控制输出
            tau = self.compute_control(q_d, dq_d, ddq_d, q, dq)
            
            # 应用控制
            self.data.ctrl[:7] = tau
            
            # 记录数据
            self.time_log.append(t)
            self.q_desired_log.append(q_d.copy())
            self.q_actual_log.append(q.copy())
            self.dq_actual_log.append(dq.copy())
            self.tau_log.append(tau.copy())
            self.error_log.append(np.linalg.norm(q_d - q))
            
            # 推进仿真
            mujoco.mj_step(self.model, self.data)
            
            # 渲染（如果启用）
            if render and i % 10 == 0:  # 每10步渲染一次
                viewer.sync()

    def plot_results(self, save_path: Optional[str] = None):
        """
        绘制结果对比图
        Args:
            save_path: 如果提供，将图像保存到指定路径
        """
        time_array = np.array(self.time_log)
        q_desired_array = np.array(self.q_desired_log)
        q_actual_array = np.array(self.q_actual_log)
        tau_array = np.array(self.tau_log)
        error_array = np.array(self.error_log)
        
        # 创建子图
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2)
        
        # 角度轨迹图
        ax1 = fig.add_subplot(gs[0:2, 0])
        for i in range(self.nq):
            ax1.plot(time_array, q_desired_array[:, i], '--', label=f'Desired j{i+1}')
            ax1.plot(time_array, q_actual_array[:, i], '-', label=f'Actual j{i+1}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Joint Angles (rad)')
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 跟踪误差图
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.plot(time_array, error_array, 'r-', label='Tracking Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error Norm')
        ax2.grid(True)
        ax2.legend()
        # 控制力矩图
        ax3 = fig.add_subplot(gs[:, 1])
        for i in range(self.nq):
            ax3.plot(time_array, tau_array[:, i], '-', label=f'Joint {i+1}')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Control Torque (Nm)')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()



def main():
    # 创建控制器实例
    controller = RobotController(
        model_path="kuka_xml_urdf/iiwa14_dock.xml",
        urdf_path="kuka_xml_urdf/iiwa14_dock.urdf"
    )
    
    # 运行不同轨迹的仿真
    trajectories = ['quintic']
    for traj in trajectories:
        print(f"\nRunning simulation with {traj} trajectory...")
        controller.run_simulation(
            trajectory_type=traj,
            duration=5.0,
            render=True
        )
        controller.plot_results(save_path=f"results_{traj}.png")

if __name__ == "__main__":
    main()