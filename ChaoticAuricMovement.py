import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Energy Flow Model
def energy_flow(E_a, P_a, dE_b, dP_b):
    return k1*E_a + k2*P_a + k3*dE_b + k4*dP_b

E_a, P_a = 2, 0.5  
dE_b, dP_b = 1.5, 0.3 
k1, k2, k3, k4 = 0.3, 0.4, 0.2, 0.5

dA_m = energy_flow(E_a, P_a, dE_b, dP_b) 
print(f"Change in Auric Movement: {dA_m}")


# Oscillation Model
time = np.linspace(0, 20, 1000)
A_0, ω, φ = 5, 0.5, np.pi/4  
noise = np.random.normal(0, 1, len(time))

def oscillation(t, A_0, ω, φ, noise):
    return A_0*np.sin(ω*t + φ) + noise

A_m = oscillation(time, A_0, ω, φ, noise)

plt.plot(time, A_m)
plt.title("Auric Movement Oscillation")
plt.show()


# Chaotic System Model 
# Simple example, can make more complex

def chaotic(A_m, E_b, P_b):
    return k1*A_m + k2*E_b + k3*P_b - k4*A_m**2

A_m0 = 0.5  
E_b = 0.7
P_b = 0.3
k1, k2, k3, k4 = 2, 4, 3, 1

time = np.linspace(0, 10, 2000)
A_m = np.empty_like(time)
A_m[0] = A_m0

for i in range(1, len(time)):
    rate = chaotic(A_m[i-1], E_b, P_b)
    A_m[i] = A_m[i-1] + rate*0.01
