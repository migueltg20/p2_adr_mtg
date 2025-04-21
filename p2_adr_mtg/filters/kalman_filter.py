import numpy as np 
import logging

logging.basicConfig(level=logging.DEBUG)

from ..motion_models import velocity_motion_model, velocity_motion_model_2
from ..observation_models import odometry_observation_model, odometry_observation_model_2

class KalmanFilter:

    def __init__(self, initial_state, initial_covariance, proc_noise_std = [0.02, 0.02, 0.01], obs_noise_std = [0.02, 0.02, 0.01]):
        self.mu = initial_state
        self.Sigma = initial_covariance

        self.A, self.B = velocity_motion_model()

        self.proc_noise_std = np.array(proc_noise_std)
        self.R = np.diag(self.proc_noise_std ** 2)

        self.C = odometry_observation_model()

        self.obs_noise_std = np.array(obs_noise_std)
        self.Q = np.diag(self.obs_noise_std ** 2)
            
    def predict(self, u, dt):
        logging.debug(f"State (mu): {self.mu}, Shape: {self.mu.shape}")
        logging.debug(f"Delta time (dt): {dt}")
        logging.debug(f"Control input (u): {u}, Shape: {u.shape}")

        self.mu = np.atleast_2d(self.mu).T
        u = np.atleast_2d(u).T

        try:
            B_result = self.B(self.mu, dt)
            logging.debug(f"B(self.mu, dt): {B_result}, Shape: {B_result.shape}")

            if B_result.ndim == 1:
                B_result = B_result[:, np.newaxis]

            if u.ndim == 1:
                u = u[:, np.newaxis]

            self.mu = (self.A @ self.mu + B_result @ u).flatten()
        except Exception as e:
            logging.error(f"Error in predict step: {e}")
            raise

        self.Sigma = self.A @ self.Sigma @ self.A.T + self.R

        logging.debug(f"Updated state (mu): {self.mu}, Shape: {self.mu.shape}")
        logging.debug(f"Updated covariance (Sigma): {self.Sigma}, Shape: {self.Sigma.shape}")

    def update(self, z):
        S = self.C @ self.Sigma @ self.C.T + self.Q
        K = self.Sigma @ self.C.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ (z - self.C @ self.mu)
        self.Sigma = (np.eye(len(self.mu)) - K @ self.C) @ self.Sigma

class KalmanFilter_2:
    def __init__(self, initial_state, initial_covariance,
                 proc_noise_std=[0.02]*6, obs_noise_std=[0.02]*6):

        self.mu = initial_state
        self.Sigma = initial_covariance

        self.A, self.B = velocity_motion_model_2()

        self.proc_noise_std = np.array(proc_noise_std)
        self.R = np.diag(self.proc_noise_std ** 2)

        self.C = odometry_observation_model_2()
        self.obs_noise_std = np.array(obs_noise_std)
        self.Q = np.diag(self.obs_noise_std ** 2)

    def predict(self, u=None, dt=1.0):
        self.mu = self.A() @ self.mu + self.B(self.mu, dt) @ (u if u is not None else np.zeros(2))
        self.Sigma = self.A() @ self.Sigma @ self.A().T + self.R

    def update(self, z):
        S = self.C @ self.Sigma @ self.C.T + self.Q
        K = self.Sigma @ self.C.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ (z - self.C @ self.mu)
        self.Sigma = (np.eye(len(self.mu)) - K @ self.C) @ self.Sigma
