import numpy as np
import matplotlib.pyplot as plt  

# Scalar Field Model
def scalar_field(x, y, z, t, V, E, K):
    def E(t):
        # your code here
        return t**2
    return V + K*E(t)

x, y, z = 0, 1, 0
t = np.linspace(0, 10, 100)
V = x**2 + y**2 + z**2   # Define V as an array
E = np.sin(t)  
K = 0.5

pot = scalar_field(x, y, z, t, V, E, K) 