import numpy as np
import matplotlib.pyplot as plt

# Auric Sensation as Energy Flow

def energy_flow_model(E_a, P_a, S_a):
    A_s = k1*E_a + k2*P_a + k3*S_a
    return A_s

E_a = 2  
P_a = 0.5
S_a = 0.8
k1, k2, k3 = 0.3, 0.4, 0.5

A_s = energy_flow_model(E_a, P_a, S_a)
print(f"Auric Sensation (Energy Flow): {A_s}")
