import numpy as np

def velocity_motion_model():
    def state_transition_matrix_A():
        # Define and return the 3x3 identity matrix A
        A = np.eye(3)
        return A

    def control_input_matrix_B(mu, delta_t):
        # Extract theta as a scalar
        theta = float(mu[2])
        B = np.array([
            [np.cos(theta)*delta_t, 0],
            [np.sin(theta)*delta_t, 0],
            [0, delta_t]
        ])
        return B

    return state_transition_matrix_A, control_input_matrix_B

def velocity_motion_model_2():
    def A():
        # Define and return the 6x6 constant velocity model transition matrix with dt=1.0
        dt = 1.0
        A_mat = np.eye(6)
        A_mat[0,3] = dt
        A_mat[1,4] = dt
        A_mat[2,5] = dt
        return A_mat

    def B(mu, dt):
        # Return 6x2 zero matrix (no control input used in pure KF)
        return np.zeros((6,2))

    return A, B
