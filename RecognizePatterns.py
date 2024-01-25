# Pattern recognition using extracted features
def recognize_patterns(features):
    # Determine the dominant brainwave state based on feature values
    dominant_state = None
    max_power = 0
    for feature_name, feature_value in features.items():
        if feature_name.startswith('mean_') and feature_value > max_power:
            dominant_state = feature_name.split('_')[1]
            max_power = feature_value

    recognized_patterns = {}
    if dominant_state:
        recognized_patterns['dominant_brainwave_state'] = dominant_state

    # Identify additional patterns based on specific feature combinations
    if features['mean_alpha_power'] > 0.5 * features['mean_beta_power']:
        recognized_patterns['relaxed_state'] = True

    if features['mean_theta_power'] > features['mean_alpha_power'] and features['mean_theta_power'] > features['mean_beta_power']:
        recognized_patterns['deep_relaxation'] = True        
    return recognized_patterns



