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

# Initialize visualization flag first
visualization_available = False
# Try to import visualization in a safer way
try:
    # Check NumPy version and warn but don't fail
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
        
        # Define process and measurement noise covariance matrices
        self.Q = np.diag([0.1, 0.1, 0.05])  # Process noise covariance
        self.R = np.diag([0.2, 0.2, 0.1])   # Measurement noise covariance

        self.kf = KalmanFilter(initial_state, initial_covariance)
        self.kf.Q = self.Q  # Set the process noise covariance directly as an attribute
        self.kf.R = self.R  # Set the measurement noise covariance
        self.last_timestamp = None
        
        # Only initialize visualizer if available
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
        # Extract velocities from odometry
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        
        # Calculate delta_t from timestamps
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_timestamp is None:
            delta_t = 0.1  # Default value for first iteration
        else:
            delta_t = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp
        
        # Extract position and orientation directly from odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Convert quaternion to yaw angle (consistent with kf_estimation_vel.py)
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny, cosy)
        
        # Retrieve current heading from filter state
        theta_current = self.kf.mu[2]
        
        # Get motion model matrices
        A_func, B_func = velocity_motion_model()
        A = A_func()
        B = B_func(np.array([v, omega, theta_current]), delta_t)
        
        # Predict step with control input
        u = np.array([v, omega])
        
        # Update the filter with the matrices and control input
        # Directly update the prediction using the matrices instead of passing a function
        self.kf.mu = A @ self.kf.mu + B @ u
        self.kf.Sigma = A @ self.kf.Sigma @ A.T + self.kf.Q
        
        # Visualize prediction step only if visualizer is available
        if self.visualizer is not None:
            try:
                self.visualizer.update(None, self.kf.mu, self.kf.Sigma, step="predict")
            except Exception as e:
                self.get_logger().warn(f"Visualization error: {e}")
                self.visualizer = None  # Disable visualizer if it fails
        
        # Original measurement directly from odometry
        original_pose = (x, y, theta)
        
        # Using consistent noise parameters instead of hardcoded values
        noise_std = [0.05, 0.05, 0.02, 0.03, 0.03]
        noisy_measurement = generate_noisy_measurement(original_pose, v, omega, noise_std)
        z = noisy_measurement[:3]  # We only need x, y, theta for the basic KF
        
        # Define observation model matrix
        H = odometry_observation_model()
        
        # Update step directly without using the function-based approach
        y = z - H @ self.kf.mu
        S = H @ self.kf.Sigma @ H.T + self.kf.R
        K = self.kf.Sigma @ H.T @ np.linalg.inv(S)
        self.kf.mu = self.kf.mu + K @ y
        self.kf.Sigma = (np.eye(3) - K @ H) @ self.kf.Sigma
        
        # Visualize update step - show both real and noisy measurements
        if self.visualizer is not None:
            try:
                self.visualizer.update(original_pose, self.kf.mu, self.kf.Sigma, step="update")
            except Exception as e:
                self.get_logger().warn(f"Visualization error: {e}")
                self.visualizer = None  # Disable visualizer if it fails
        
        # Publish estimated state
        estimate_msg = PoseWithCovarianceStamped()
        estimate_msg.header = msg.header  # Copy timestamp and frame_id
        
        # Populate PoseWithCovarianceStamped message with state and covariance
        estimate_msg.pose.pose.position.x = self.kf.mu[0]
        estimate_msg.pose.pose.position.y = self.kf.mu[1]
        estimate_msg.pose.pose.orientation.z = math.sin(self.kf.mu[2] / 2.0)
        estimate_msg.pose.pose.orientation.w = math.cos(self.kf.mu[2] / 2.0)
        
        # Create a 6x6 covariance matrix (required by ROS)
        # Initialize with zeros
        full_covariance = np.zeros((6, 6))
        
        # Copy the 3x3 KF covariance into the appropriate positions
        # First 2x2 block for x, y position
        full_covariance[0:2, 0:2] = self.kf.Sigma[0:2, 0:2]
        # Theta goes into the yaw component (5,5)
        full_covariance[5, 5] = self.kf.Sigma[2, 2]
        # Cross-correlations between position and orientation
        full_covariance[0, 5] = full_covariance[5, 0] = self.kf.Sigma[0, 2]
        full_covariance[1, 5] = full_covariance[5, 1] = self.kf.Sigma[1, 2]
        
        # Fill the covariance matrix in the ROS message (36 elements)
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
