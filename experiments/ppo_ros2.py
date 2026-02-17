import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
import numpy as np
from stable_baselines3 import PPO
import math
import pickle

class BlueRovFinalController(Node):
    def __init__(self):
        super().__init__('bluerov_final_controller')
        
        self.thruster_pubs = []
        for i in range(1, 7):
            pub = self.create_publisher(Float64, f'/bluerov2/cmd_thruster{i}', 10)
            self.thruster_pubs.append(pub)
            
        self.pose_sub = self.create_subscription(Pose, '/bluerov2/pose_gt', self.pose_callback, 10)
            
        self.model = PPO.load("bluerov_ppo")
        with open("bluerov_vec_normalize.pkl", "rb") as f:
            self.stats = pickle.load(f)
        
        self.stats.training = False
        self.stats.norm_reward = False
        
        self.current_pos = np.zeros(4) 
        self.prev_pos = np.zeros(4)
        self.velocity = np.zeros(4) 
        
        self.setpoint = np.array([0.0, 0.0, -10.0, 0.0])
        self.pose_received = False
        self.last_time = self.get_clock().now()
        
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Correccion final: X normal, Z invertido.')

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def pose_callback(self, msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        
        if dt > 0:
            self.current_pos[0] = msg.position.x
            self.current_pos[1] = msg.position.y
            self.current_pos[2] = msg.position.z
            self.current_pos[3] = self.get_yaw_from_quaternion(msg.orientation)
            
            v_global = (self.current_pos - self.prev_pos) / dt
            theta = self.current_pos[3]
            
            self.velocity[0] = v_global[0] * np.cos(theta) + v_global[1] * np.sin(theta)
            self.velocity[1] = -v_global[0] * np.sin(theta) + v_global[1] * np.cos(theta)
            self.velocity[2] = v_global[2]
            self.velocity[3] = v_global[3]
            
            self.prev_pos = np.copy(self.current_pos)
            self.last_time = now
            self.pose_received = True

    def map_forces_to_thrusters(self, action_flat):
        wx, wy, wz, w_omega = action_flat
        
        # Mapeo basado en observacion de logs:
        wx_mapped = wx    # X normal
        wz_mapped = -wz   # Z invertido (mantiene estabilidad)
        
        c = 0.707
        t = np.zeros(6)
        
        t[0] =  wx_mapped * c + wy * c + w_omega 
        t[1] =  wx_mapped * c - wy * c - w_omega 
        t[2] = -wx_mapped * c + wy * c - w_omega 
        t[3] = -wx_mapped * c - wy * c + w_omega 
        
        t[4] = -wz_mapped                  
        t[5] = -wz_mapped                  
        
        return np.clip(t * 40.0, -40.0, 40.0)

    def control_loop(self):
        if not self.pose_received: return

        error_pos = self.setpoint - self.current_pos
        
        obs_dict = {
            "x": np.array([error_pos[0]], dtype=np.float32),
            "y": np.array([error_pos[1]], dtype=np.float32),
            "z": np.array([error_pos[2]], dtype=np.float32),
            "theta": np.array([error_pos[3]], dtype=np.float32),
            "vx": np.array([self.velocity[0]], dtype=np.float32),
            "vy": np.array([self.velocity[1]], dtype=np.float32),
            "vz": np.array([self.velocity[2]], dtype=np.float32),
            "omega": np.array([self.velocity[3]], dtype=np.float32),
        }

        obs_norm = self.stats.normalize_obs(obs_dict)
        action, _ = self.model.predict(obs_norm, deterministic=True)
        
        action_flat = action.flatten()
        thruster_cmds = self.map_forces_to_thrusters(action_flat)
        
        for i in range(6):
            msg = Float64()
            msg.data = float(thruster_cmds[i])
            self.thruster_pubs[i].publish(msg)

        print(f"Erro X: {error_pos[0]:.2f} | Erro Z: {error_pos[2]:.2f} | Vz: {self.velocity[2]:.4f}")

def main(args=None):
    rclpy.init(args=args)
    node = BlueRovFinalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()