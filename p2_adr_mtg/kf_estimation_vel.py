import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

import numpy as np
import math

from .sensor_utils import odom_to_pose2D, get_normalized_pose2D, generate_noisy_measurement_2
from .filters.kalman_filter import KalmanFilter_2
from .simple_visualizer import SimpleVisualizer
from .motion_models import velocity_motion_model_2
from .observation_models import odometry_observation_model_2

class KalmanFilterPureNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_pure_node')

        initial_state = np.zeros(6)
        initial_covariance = np.eye(6) * 0.1

        self.kf = KalmanFilter_2(initial_state, initial_covariance)
        self.visualizer = SimpleVisualizer(self)
        self.last_timestamp = None

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/kf2_estimate',
            10
        )

    def odom_callback(self, msg):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny, cosy)

        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_timestamp is None:
            dt = 0.1 
        else:
            dt = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp
        
        A_func, B_func = velocity_motion_model_2()
        A = A_func()
        B = B_func(np.array([v, omega]), dt)
        
        u = np.array([v, omega])
        self.kf.predict(u, dt)
        
        self.visualizer.update(None, self.kf.mu, self.kf.Sigma, step="predict")
        
        original_pose = (x, y, theta)
        
        noise_std = [0.05, 0.05, 0.02, 0.03, 0.03, 0.02]  
        vx = v * np.cos(theta)  
        vy = v * np.sin(theta)
        z = generate_noisy_measurement_2(original_pose, vx, vy, omega, noise_std)
        
        H = odometry_observation_model_2()

        self.kf.update(z)  
        
        self.visualizer.update(original_pose, self.kf.mu, self.kf.Sigma, step="update")
        
        estimate_msg = PoseWithCovarianceStamped()
        estimate_msg.header = msg.header  
        
        estimate_msg.pose.pose.position.x = self.kf.mu[0]
        estimate_msg.pose.pose.position.y = self.kf.mu[1]
        estimate_msg.pose.pose.orientation.z = math.sin(self.kf.mu[2] / 2.0)
        estimate_msg.pose.pose.orientation.w = math.cos(self.kf.mu[2] / 2.0)
        
        estimate_msg.pose.covariance = self.kf.Sigma.flatten().tolist()
        self.publisher.publish(estimate_msg)

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilterPureNode()
    rclpy.spin(node)
    rclpy.shutdown()

