import numpy as np
from scipy.signal import spectrogram

def get_eeg_features(eeg_data):
    """Extracts spectral features from EEG"""
    f, t, Sxx = spectrogram(eeg_data, fs=100)

    features = {}

    # Sum power in alpha band (8-12hz)
    alpha_band = (f > 8) & (f < 12)
    features['alpha_power'] = np.sum(Sxx[alpha_band, :])

    # Calculate peak alpha frequency 
    i, j = np.unravel_index(np.argmax(Sxx[:, alpha_band]), Sxx.shape)
    features['alpha_peak'] = f[i]

    # Add more features...

    return features

# Generate synthetic sample data  
new_features = {
    'alpha_power': np.random.rand(), 
    'alpha_peak': 10 + np.random.randn(),
}
