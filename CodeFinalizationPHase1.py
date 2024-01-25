import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Assuming the LSTM expects sequences of length 'n_steps'
n_steps = 5  # This can be adjusted based on your model's training

def preprocess_data(data):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

    # Reshape data into sequences
    X = []
    for i in range(len(data_scaled) - n_steps):
        X.append(data_scaled[i:i + n_steps, 0])
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM [samples, time steps, features]
    
    return X, scaler

# Example usage
# real_time_data = [..]  # This should be your real-time data array
# processed_data, scaler = preprocess_data(real_time_data)
def predict(model, data):
    prediction = model.predict(data)
    return prediction

# Example usage
# lstm_model = build_lstm_model(...)  # Assuming the model is built and trained
# real_time_processed_data = preprocess_data(real_time_data)
# prediction = predict(lstm_model, real_time_processed_data)

def inverse_transform_prediction(scaler, prediction):
    # Reshape prediction to the format expected by scaler
    prediction_reshaped = prediction.reshape(-1, 1)
    # Inverse transform
    inverted_prediction = scaler.inverse_transform(prediction_reshaped)
    return inverted_prediction

# Example usage
# prediction_scaled = predict(lstm_model, real_time_processed_data)
# prediction_original_scale = inverse_transform_prediction(scaler, prediction_scaled)

# real_time_data is your incoming real-time data

# Preprocess the data
processed_data, scaler = preprocess_data(real_time_data)

# Make prediction
prediction_scaled = predict(lstm_model, processed_data)

# Inverse transform the prediction
prediction_original_scale = inverse_transform_prediction(scaler, prediction_scaled)

# Now, prediction_original_scale contains the prediction in its original scale
