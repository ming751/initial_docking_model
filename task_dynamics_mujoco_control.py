import mujoco
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from time import sleep
from typing import Callable, Optional
import os
import mujoco.viewer
from typing import Tuple, List

from Relate_class import TaskSpaceController,TaskSpaceTrajectory,DecoupledQuinticTrajectory



class Log:
    def __init__(self):

        self.nq = 7

    def reset_logs(self):
        """重置记录数据"""

        self.t_list = []

        self.joint_angles = []
        self.joint_velocities = []

        self.pos_actual = []
        self.vel_actual = []

        self.error =[]

        self.pos_desired = []
        self.vel_desired = []
        self.acc_desired = []

        self.tau_hist = []
        self.force_externals = []
        self.torque_externals = []



    def store_data(self, t: float, q: np.ndarray, 
                   v: np.ndarray, pos_actual: np.ndarray, 
                   vel_actual: np.ndarray, error: float,
                   pos_desired: np.ndarray,vel_desired: np.ndarray,
                   acc_desired: np.ndarray,tau: np.ndarray,
                   external_force: np.ndarray,external_torque: np.ndarray):
        """
        存储数据
        Args:
            t: 时间
            q_d: 期望的位置
            q: 实际的位置
            dq: 实际的速度
            tau: 实际的控制力矩
            error: 跟踪误差
        """

        self.t_list.append(t)

        self.joint_angles.append(q)
        self.joint_velocities.append(v)

        self.pos_actual.append(pos_actual)
        self.vel_actual.append(vel_actual)

        self.error.append(error)

        self.pos_desired.append(pos_desired)
        self.vel_desired.append(vel_desired)
        self.acc_desired.append(acc_desired)

        self.tau_hist.append(tau)
        self.force_externals.append(external_force)
        self.torque_externals.append(external_torque)



    def plot_results(self):
        
        """绘制仿真结果"""
        pos_actual = np.array(self.pos_actual).reshape(-1, 3)
        pos_desired = np.array(self.pos_desired)
        joint_angles = np.array(self.joint_angles)
        joint_velocities = np.array(self.joint_velocities)

        virtual_force_list = np.zeros_like(pos_actual)
        # array(self.virtual_force_list)


        tau_hist = np.array(self.tau_hist)
        external_forces = np.array(self.force_externals)
        external_torques = np.array(self.torque_externals)

        if len(self.force_externals) < 1:
            external_forces = np.zeros_like(pos_actual)
            external_torques = np.zeros_like(pos_actual)
            
        
        # 创建三个子图
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2)
        
        # 1. 位置跟踪
        ax1 = fig.add_subplot(gs[0, :])
        labels = ['X', 'Y', 'Z']
        color = ['r', 'g', 'b']
        for i in range(3):
            ax1.plot(self.t_list, pos_actual[:, i], '-', label=f'Actual {labels[i]}', color=color[i])
            ax1.plot(self.t_list, pos_desired[:, i], '--', label=f'Desired {labels[i]}', color=color[i])
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position [m]')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('End-effector Position Tracking')
        
        # 2. 跟踪误差
        ax2 = fig.add_subplot(gs[1, :])
        for i in range(3):
            error = pos_desired[:, i] - pos_actual[:, i]
            ax2.plot(self.t_list, error, label=f'{labels[i]} Error')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Error [m]')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Position Tracking Error')
        
        # 3. 关节角度和速度
        ax3 = fig.add_subplot(gs[2, 0])
        for i in range(3):
            ax3.plot(self.t_list[:], external_forces[:, i], label=f'axis {i+1}')
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Virtual Force [N]')
            ax3.legend()
            ax3.grid(True)
            ax3.set_title('Impedance Force')

        ax4 = fig.add_subplot(gs[2, 1])
        for i in range(3):
            ax4.plot(self.t_list[:], external_torques[:, i], label=f'axis {i+1}')
            ax4.set_xlabel('Time [s]')
            ax4.set_ylabel('Virtual Force [N]')
            ax4.legend()
            ax4.grid(True)
            ax4.set_title('Impedance Torque')

        # for i in range(6):
        #     ax3.plot(self.t_list, np.rad2deg(joint_angles[:, i]), label=f'Joint {i+1}')
        # ax3.set_xlabel('Time [s]')
        # ax3.set_ylabel('Joint Angle [deg]')
        # ax3.legend()
        # ax3.grid(True)
        # ax3.set_title('Joint Angles')

        # ax4 = fig.add_subplot(gs[2, 1])


        plt.show()
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(7, 1)

        for i in range(7):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(self.t_list, tau_hist[:, i], label=f'Joint {i+1}')
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Torque [Nm]')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title('Joint Torques')

        plt.show()


class MujRobot:
    def __init__(self, model_path: str,render:bool= True,dt:float=0.001,target_pos =np.zeros(3)):

        self.model_path = model_path
        self.dt = dt
        self.target_pos = target_pos

        self.setup_mujoco()
        self.viewer = None

        self.render = render
        
        self.setup_viewer()

    def setup_mujoco(self):
        """设置MuJoCo环境"""
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model.opt.tolerance = 0.001
        # self.model.nconmax = 100  # 根据需要调整数值
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = self.dt

        self.eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "dock1")  # 替换为实际末端执行器名称
        self.eef_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_marker")
        self.vis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "vis")

        # self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'track_cam')

    def init_simulators(self,init_qpos:np.ndarray):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, init_qpos, mujoco.mjtState.mjSTATE_QPOS)
        mujoco.mj_forward(self.model, self.data)



    def setup_viewer(self):
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # self.viewer.cam.trackingcamid = self.camera_id
            # self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    #def 

    def step(self,tau:np.ndarray ):

        self.data.ctrl[:7] = tau
        mujoco.mj_step(self.model, self.data)

        
        qpos = self.data.qpos[:7]
        qvel = self.data.qvel[:7]

        eef_pos = self.data.xpos[self.eef_id]
        self.data.site_xpos[self.vis_id] = self.target_pos
        #self.data.site_xpos[self.eef_marker_id] = eef_pos
        if self.render:
            # with self.viewer.lock():
            #      self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
            #      self.viewer.opt.flags[mujoco.mjtVisFlag. mjVIS_CONTACTFORCE] = True

            self.viewer.sync()

        return qpos,qvel,eef_pos
    
    def get_ee_state(self):

        ee_pos = self.data.xpos[self.eef_id]
        print(dir(self.data))
        ee_vel = self.data.body_xvelp[self.eef_id]
        return ee_pos,ee_vel
    
    def get_joint_state(self):
        qpos = self.data.qpos[:7]
        qvel = self.data.qvel[:7]
        return qpos,qvel
    
def compute_ik(pin_model, pin_data, target_pose, initial_q=np.ones(7)*0.3, max_iters=3000, eps=1e-7):
    """
    Compute inverse kinematics using Pinocchio's functions
    
    Args:
        pin_model: Pinocchio model
        pin_data: Pinocchio data
        target_pose: pin.SE3, target end-effector pose
        initial_q: Initial joint configuration, if None uses neutral configuration
        max_iters: Maximum iterations for the IK solver
        eps: Convergence threshold
        
    Returns:
        q: Solved joint angles
        success: Whether IK converged successfully
    """
    initial_q = np.array([1.07746776 , 0.28602036, -2.47939995 , 1.29702927 , 2.96705973 , 1.6159687,-1.4298983 ])
    if initial_q is None:
        q = pin.neutral(pin_model)
    else:
        q = initial_q.copy()
        
    # Get end effector frame ID
    ee_frame_id = pin_model.getFrameId("cylinder_link")
    
    # Damping factor for numerical stability
    damp = 1e-8
    
    for i in range(max_iters):
        # Update robot kinematics
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        
        # Get current end-effector pose
        current_pose = pin_data.oMf[ee_frame_id]
        
        # Compute position error
        error_pos = target_pose.translation - current_pose.translation
        
        # Compute orientation error using matrix logarithm
        error_rot = pin.log3(target_pose.rotation @ current_pose.rotation.T)
        
        # Combine errors
        error = np.concatenate([error_pos, error_rot])
        
        # Check convergence
        if np.linalg.norm(error) < eps:
            print(f"IK converged in {i+1} iterations")
            return q, True
            
        # Compute task Jacobian
        pin.computeJointJacobians(pin_model, pin_data, q)
        J = pin.getFrameJacobian(pin_model, pin_data, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        # Compute joint update using damped least squares
        Jt = J.T
        JJt = J @ Jt
        lambda_eye = damp * np.eye(6)  # 6 DOF task space
        
        # Solve using damped least squares
        v = np.linalg.solve(JJt + lambda_eye, error)
        dq = Jt @ v
        
        # Update joint positions
        q = pin.integrate(pin_model, q, dq)
        
        # Joint limits handling (if needed)
        q = np.clip(q, pin_model.lowerPositionLimit, pin_model.upperPositionLimit)
    
    print("IK failed to converge")
    return q, False

        

def run_simulation(muj_robot:MujRobot,
                   task_dynamics:TaskSpaceController,
                   trajector_planner:TaskSpaceTrajectory,
                   log:Log,duration:float,
                   dt:float,q_init:np.ndarray):
    
    log.reset_logs()

    muj_robot.init_simulators(q_init)



    q = q_init
    v = np.zeros(7)

    current_pos, current_vel, _ = task_dynamics.get_task_space_state(q, v)

    force_external = np.zeros(3)
    torque_external = np.zeros(3)


    while muj_robot.data.time < duration:
   # while True:
        t = muj_robot.data.time
        pos_des, vel_des, acc_des = trajector_planner.get_state(t)
        #print(f"Current time: {t}, Desired position: {pos_des}, Desired velocity: {vel_des}, Desired acceleration: {acc_des}")

        tau = task_dynamics.compute_control_task_space_with_orientation_and_imp(q, v, pos_des, vel_des, acc_des ,current_pos,current_vel,force_external,torque_external)
                                                                
        q,v,eef_pos = muj_robot.step(tau)
        #print(f"Current time: {t}, End-effector position: {q},tau: {v}",)

        current_pos, current_vel ,current_ori = task_dynamics.get_task_space_state(q, v)

        current_ori_log = pin.log3(current_ori)

        current_pos = np.array(current_pos)
        tau = np.array(tau)

        force_sensor = -muj_robot.data.sensor("force_sensor").data

        torque_sensor = -muj_robot.data.sensor("torque_sensor").data


        force_external = current_ori @ force_sensor 

        torque_external = current_ori @ torque_sensor

        #print(f"current_time:{t},force_external:{force_exteral},current_ori:{current_ori_log}")

        #print('current:',current_pos,"current_ori:",current_ori)

        # print('far:',np.linalg.norm(eef_pos-current_pos))

        #log.store_data(t, pos_des, q, v, tau, np.linalg.norm(current_pos - pos_des))

        log.store_data(t, q, v, current_pos, current_vel, np.linalg.norm(current_pos - pos_des),pos_des,vel_des,acc_des,tau ,force_external,torque_external)
                #    v: np.ndarray, pos_actual: np.ndarray, 
                #    vel_actual: np.ndarray, error: float,
                #    pos_desired: np.ndarray,vel_desired: np.ndarray,
                #    acc_desired: np.ndarray)

    log.plot_results()



def main():

    dt = 0.001

    
    log = Log()

    pin_model = pin.buildModelFromUrdf("kuka_xml_urdf/iiwa14_dock.urdf")
    #pin_model = pin.buildModelFromMJCF("kuka_xml_urdf/iiwa14_dock.xml")
    pin_data = pin_model.createData()


    task_dynamics = TaskSpaceController(pin_model, pin_data)


    init_pos = np.array([0.0, 0.5, 0.5])
    init_ori = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    # init_ori = np.array([
    #     [1,  0,  0],
    #     [0, -1,  0],
    #     [0,  0, 1]
    # ])
    init_pose = pin.SE3(init_ori, init_pos)

    q_init , success = compute_ik(pin_model, pin_data, init_pose)

    pin.forwardKinematics(pin_model, pin_data, q_init)
    pin.updateFramePlacements(pin_model, pin_data)
    H_init = pin_data.oMf[pin_model.getFrameId("cylinder_link")]
    print(f"Initial end-effector position: {H_init.translation}")

    

    print(f"Initial joint positions: {q_init}",success)

    
    # 设置目标位置（相对运动）
    target_pos = init_pos + np.array([0.00, -0.00,-0.18])

    muj_robot = MujRobot(model_path="kuka_xml_urdf/iiwa14_dock.xml",render=True,dt=dt ,target_pos = target_pos)
    print(f"Target position: {target_pos}")
    #target_pos = np.array([0.25,0.25,0.5,])
    traj_duration = 15.0
    trajector_planner = DecoupledQuinticTrajectory(init_pos ,target_pos ,traj_duration)

    run_simulation(muj_robot,task_dynamics,trajector_planner,log,duration=20,dt=dt,q_init = q_init)

if __name__ == '__main__':
    main()

    



