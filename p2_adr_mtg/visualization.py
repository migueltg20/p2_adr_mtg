import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import matplotlib.pyplot as plt

class Visualizer(Node):
    def __init__(self):
        super().__init__('kalman_filter_visualizer')
        self.estimated_pose_pub = self.create_publisher(PoseStamped, 'kf_estimated_pose', 10)
        self.estimated_path_pub = self.create_publisher(Path, 'kf_estimated_path', 10)
        self.real_path_pub = self.create_publisher(Path, 'real_path', 10)
        self.estimated_path = Path()
        self.real_path = Path()
        self.fig, self.ax = plt.subplots()
        self.estimated_positions = []
        self.real_positions = []
    
    def update(self, real_pose, estimated_pose, covariance, step="predict"):
        self.publish_to_rviz(real_pose, estimated_pose)
        self.update_matplotlib(real_pose, estimated_pose, covariance, step)
    
    def publish_to_rviz(self, real_pose, estimated_pose):
        now = self.get_clock().now().to_msg()
        est_pose_msg = PoseStamped()
        est_pose_msg.header.stamp = now
        est_pose_msg.header.frame_id = "map"
        est_pose_msg.pose.position.x = estimated_pose[0]
        est_pose_msg.pose.position.y = estimated_pose[1]
        real_pose_msg = PoseStamped()
        real_pose_msg.header.stamp = now
        real_pose_msg.header.frame_id = "map"
        real_pose_msg.pose.position.x = real_pose[0]
        real_pose_msg.pose.position.y = real_pose[1]
        self.estimated_path.header = est_pose_msg.header
        self.estimated_path.poses.append(est_pose_msg)
        self.real_path.header = real_pose_msg.header
        self.real_path.poses.append(real_pose_msg)
        self.estimated_pose_pub.publish(est_pose_msg)
        self.estimated_path_pub.publish(self.estimated_path)
        self.real_path_pub.publish(self.real_path)
    
    def update_matplotlib(self, real_pose, estimated_pose, covariance, step):
        self.real_positions.append(real_pose[:2])
        self.estimated_positions.append(estimated_pose[:2])
        self.ax.clear()
        self.ax.set_title(f"Kalman Filter {step} step")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        real_x, real_y = zip(*self.real_positions)
        est_x, est_y = zip(*self.estimated_positions)
        self.ax.plot(real_x, real_y, 'g-', label="Real Path")
        self.ax.plot(est_x, est_y, 'b-', label="Estimated Path")
        self.ax.scatter(est_x[-1], est_y[-1], c='red', label="Latest Estimate")
        self.ax.legend()
        plt.pause(0.01)
    
    def plot_covariance(self, mean, covariance):
        if covariance.shape == (3, 3):
            covariance = covariance[:2, :2]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = plt.matplotlib.patches.Ellipse((mean[0],mean[1]), width, height, np.degrees(angle), edgecolor='red', facecolor='none')
        self.ax.add_patch(ellipse)
    
    def show_plot(self):
        plt.show()

if __name__ == "__main__":
    rclpy.init()
    visualizer = Visualizer()
    rclpy.spin(visualizer)
    visualizer.destroy_node()
    rclpy.shutdown()

