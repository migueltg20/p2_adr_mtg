import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

class SimpleVisualizer:
    """
    A simple visualizer that doesn't depend on matplotlib, 
    avoiding NumPy 2.x compatibility issues.
    """
    
    def __init__(self, node=None):
        self.node = node
        if self.node:
            self.marker_publisher = self.node.create_publisher(
                MarkerArray,
                '/kf_visualization',
                10
            )
            self.pose_publisher = self.node.create_publisher(
                PoseArray,
                '/kf_poses',
                10
            )
        self.counter = 0
    
    def update(self, measurement, state, covariance, step=None):
        """
        Update visualization with current state and covariance
        
        Parameters:
        -----------
        measurement : tuple or None
            Original measurement (x, y, theta)
        state : np.ndarray
            Estimated state vector
        covariance : np.ndarray
            Covariance matrix
        step : str
            Label for the step (e.g. "predict", "update")
        """
        if not self.node:
            return
            
        self.counter += 1
        
        # Skip too frequent visualization to reduce load
        if self.counter % 5 != 0:
            return
            
        # Create marker for the state
        marker_array = MarkerArray()
        
        # State position marker
        position_marker = Marker()
        position_marker.header.frame_id = "odom"
        position_marker.header.stamp = self.node.get_clock().now().to_msg()
        position_marker.ns = "kf_state"
        position_marker.id = 0
        position_marker.type = Marker.SPHERE
        position_marker.action = Marker.ADD
        
        # Position from state
        position_marker.pose.position.x = state[0]
        position_marker.pose.position.y = state[1]
        position_marker.pose.position.z = 0.1
        
        # Orientation from state
        position_marker.pose.orientation.z = np.sin(state[2] / 2.0)
        position_marker.pose.orientation.w = np.cos(state[2] / 2.0)
        
        # Size and color
        position_marker.scale.x = 0.2
        position_marker.scale.y = 0.2
        position_marker.scale.z = 0.2
        position_marker.color.a = 1.0
        position_marker.color.r = 0.0
        position_marker.color.g = 1.0
        position_marker.color.b = 0.0
        
        marker_array.markers.append(position_marker)
        
        # Display covariance as ellipse if needed
        if covariance is not None and covariance.shape[0] >= 2:
            # Simple representation of covariance as a marker
            cov_marker = Marker()
            cov_marker.header.frame_id = "odom"
            cov_marker.header.stamp = self.node.get_clock().now().to_msg()
            cov_marker.ns = "kf_covariance"
            cov_marker.id = 1
            cov_marker.type = Marker.SPHERE
            cov_marker.action = Marker.ADD
            
            cov_marker.pose.position.x = state[0]
            cov_marker.pose.position.y = state[1]
            cov_marker.pose.position.z = 0.05
            
            # Use the covariance to scale the marker
            # Just using a simplified approach
            cov_x = max(0.1, np.sqrt(covariance[0, 0]))
            cov_y = max(0.1, np.sqrt(covariance[1, 1]))
            
            cov_marker.scale.x = cov_x
            cov_marker.scale.y = cov_y
            cov_marker.scale.z = 0.01
            
            cov_marker.color.a = 0.3
            cov_marker.color.r = 1.0
            cov_marker.color.g = 0.0
            cov_marker.color.b = 0.0
            
            marker_array.markers.append(cov_marker)
        
        # If we have a measurement, show it as well
        if measurement is not None:
            meas_marker = Marker()
            meas_marker.header.frame_id = "odom"
            meas_marker.header.stamp = self.node.get_clock().now().to_msg()
            meas_marker.ns = "measurement"
            meas_marker.id = 2
            meas_marker.type = Marker.SPHERE
            meas_marker.action = Marker.ADD
            
            meas_marker.pose.position.x = measurement[0]
            meas_marker.pose.position.y = measurement[1]
            meas_marker.pose.position.z = 0.15
            
            # Size and color
            meas_marker.scale.x = 0.15
            meas_marker.scale.y = 0.15
            meas_marker.scale.z = 0.15
            meas_marker.color.a = 1.0
            meas_marker.color.r = 0.0
            meas_marker.color.g = 0.0
            meas_marker.color.b = 1.0
            
            marker_array.markers.append(meas_marker)
        
        self.marker_publisher.publish(marker_array)
