import numpy as np
import matplotlib.pyplot as plt

# Hormonal Influence Model
def hormonal_model(hormones, em_change):
    return sum(hormones) + em_change

time = np.linspace(0, 10, 30)  
h1 = np.sin(time) 
h2 = np.cos(time)
em_change = np.random.rand(len(time))

mood = hormonal_model([h1, h2], em_change)