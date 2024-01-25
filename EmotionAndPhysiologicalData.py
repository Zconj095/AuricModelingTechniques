import numpy as np

# Discrete Level-Based
emotions = {"happy": 3, "sad": 5, "angry": 8} 

def level_based(emotion):
    return emotions[emotion]

print(level_based("happy"))


# Continuous Intensity Score

def intensity_score(emotion, intensity):
    return intensity  

emotion = "happy" 
intensity = 0.7

print(intensity_score(emotion, intensity))


# Physiological Response

def physiological(parameters):
    return np.mean(parameters)

parameters = [70, 16, 0.7] # [heart rate, skin conductance, facial expression score]

print(physiological(parameters))


# Multi-dimensional

def multi_dimensional(valence, arousal, dominance):
    return (valence + arousal + dominance)/3

valence = 0.6  
arousal = 0.8
dominance = 0.3

print(multi_dimensional(valence, arousal, dominance))

import numpy as np
from math import log2

# Frequency-Based
time_frame = 60 # 1 minute
num_transitions = 10
emotional_throughput = num_transitions / time_frame
print(emotional_throughput)


# Intensity-Weighted Frequency
time_frame = 60
intensities = [0.3, 0.8, 0.6, 0.4, 0.9]
transitions = [3, 2, 1, 4, 2] 

weighted_throughput = sum([intensity*transitions[i] for i, intensity in enumerate(intensities)]) / time_frame
print(weighted_throughput)


# Entropy-Based
p = [0.3, 0.1, 0.4, 0.05, 0.15] # Distribution

entropy = -sum([pi*log2(pi) for pi in p])
print(entropy)


# Physiological Response Rate 
rates = [0.02, -0.05, 0.03] # Sample parameter change rates

throughput = sum(np.abs(rates)) 
print(throughput)