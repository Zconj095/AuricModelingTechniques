import numpy as np
import matplotlib.pyplot as plt

# Energy Flow Model
E_a = 2  
dE_b = 3 
k = 0.5

dA_s = -k*(E_a + dE_b)
print(f"Change in Auric Stillness: {dA_s}")


# Oscillation Model  
time = np.linspace(0,20,100)
A_0, ω = 10, 0.1 
noise = np.random.normal(0,3,len(time))  

def oscillation(t, A0, ω, noise):
    return A0*np.exp(-ω*t) + noise

A_s = oscillation(time, A_0, ω, noise)