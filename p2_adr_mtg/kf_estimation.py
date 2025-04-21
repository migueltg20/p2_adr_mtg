import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from irobot_create_msgs.msg import WheelVels

import numpy as np
import math
import sys

from .sensor_utils import odom_to_pose2D, get_normalized_pose2D, Odom2DDriftSimulator, generate_noisy_measurement
from .motion_models import velocity_motion_model
from .observation_models import odometry_observation_model
from .filters.kalman_filter import KalmanFilter

visualization_available = False
try:
    np_version = np.__version__
    if np_version.startswith('2.'):
        print(f"Warning: Running with NumPy {np_version}, visualization might not work properly")
    from .visualization import Visualizer
    visualization_available = True
    print("Visualization module imported successfully")
except (ImportError, AttributeError, Exception) as e:
    print(f"Visualization module could not be imported: {e}")
    print("Running without visualization capabilities.")
    visualization_available = False

class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        initial_state = np.zeros(3)
        initial_covariance = np.eye(3) * 0.1
        
        self.Q = np.diag([0.1, 0.1, 0.05])
        self.R = np.diag([0.2, 0.2, 0.1])

        self.kf = KalmanFilter(initial_state, initial_covariance)
        self.kf.Q = self.Q
        self.kf.R = self.R
        self.last_timestamp = None
        
        self.visualizer = None
        if visualization_available:
            try:
                self.visualizer = Visualizer()
                self.get_logger().info("Visualization initialized successfully")
            except Exception as e:
                self.get_logger().warn(f"Failed to initialize visualizer: {e}")
                self.get_logger().warn("Running without visualization")

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/kf_estimate',
            10
        )
        
        self.get_logger().info("Kalman Filter node initialized")

    def odom_callback(self, msg):
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_timestamp is None:
            delta_t = 0.1
        else:
            delta_t = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny, cosy)
        
        theta_current = self.kf.mu[2]
        
        A_func, B_func = velocity_motion_model()
        A = A_func()
        B = B_func(np.array([v, omega, theta_current]), delta_t)
        
        u = np.array([v, omega])
        
        self.kf.mu = A @ self.kf.mu + B @ u
        self.kf.Sigma = A @ self.kf.Sigma @ A.T + self.kf.Q
        
        if self.visualizer is not None:
            try:
                self.visualizer.update(None, self.kf.mu, self.kf.Sigma, step="predict")
            except Exception as e:
                self.get_logger().warn(f"Visualization error: {e}")
                self.visualizer = None
        
        original_pose = (x, y, theta)
        
        noise_std = [0.05, 0.05, 0.02, 0.03, 0.03]
        noisy_measurement = generate_noisy_measurement(original_pose, v, omega, noise_std)
        z = noisy_measurement[:3]
        
        H = odometry_observation_model()
        
        y = z - H @ self.kf.mu
        S = H @ self.kf.Sigma @ H.T + self.kf.R
        K = self.kf.Sigma @ H.T @ np.linalg.inv(S)
        self.kf.mu = self.kf.mu + K @ y
        self.kf.Sigma = (np.eye(3) - K @ H) @ self.kf.Sigma
        
        if self.visualizer is not None:
            try:
                self.visualizer.update(original_pose, self.kf.mu, self.kf.Sigma, step="update")
            except Exception as e:
                self.get_logger().warn(f"Visualization error: {e}")
                self.visualizer = None
        
        estimate_msg = PoseWithCovarianceStamped()
        estimate_msg.header = msg.header
        
        estimate_msg.pose.pose.position.x = self.kf.mu[0]
        estimate_msg.pose.pose.position.y = self.kf.mu[1]
        estimate_msg.pose.pose.orientation.z = math.sin(self.kf.mu[2] / 2.0)
        estimate_msg.pose.pose.orientation.w = math.cos(self.kf.mu[2] / 2.0)
        
        full_covariance = np.zeros((6, 6))
        
        full_covariance[0:2, 0:2] = self.kf.Sigma[0:2, 0:2]
        full_covariance[5, 5] = self.kf.Sigma[2, 2]
        full_covariance[0, 5] = full_covariance[5, 0] = self.kf.Sigma[0, 2]
        full_covariance[1, 5] = full_covariance[5, 1] = self.kf.Sigma[1, 2]
        
        estimate_msg.pose.covariance = full_covariance.flatten().tolist()
        self.publisher.publish(estimate_msg)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = KalmanFilterNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
