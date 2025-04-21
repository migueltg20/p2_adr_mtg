import numpy as np

def odometry_observation_model():
    return np.eye(3)

def odometry_observation_model_2():
    return np.eye(6)
