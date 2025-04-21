import numpy as np

def odometry_observation_model():
    # Return identity matrix (3x3) for fully observable state
    return np.eye(3)

def odometry_observation_model_2():
    # Return identity matrix (6x6) for fully observed 6D state
    return np.eye(6)
