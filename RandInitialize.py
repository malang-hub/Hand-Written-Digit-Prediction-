import numpy as np

def initialise(L_out, L_in):
    epsilon_init = 0.12
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init