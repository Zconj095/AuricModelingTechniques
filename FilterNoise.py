# Noise reduction using a simple high-pass filter
import numpy as np

def filter_noise(data):
    # Implement a simple high-pass filter to remove low-frequency noise
    filtered_data = {}

    cutoff_frequency = 10  # Set the cutoff frequency for noise reduction
    for key, value in data.items():
        if value < cutoff_frequency:
            filtered_data[key] = 0
        else:
            filtered_data[key] = value - cutoff_frequency

    return filtered_data
