

import numpy as np
import torch
states = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

s = np.arange(states.shape[0]) != 1
print(states.shape[0])
print(states[states.shape[0] != 1])