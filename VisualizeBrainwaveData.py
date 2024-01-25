import matplotlib as plt
def visualize_brainwave_data(data):
    # Extract pulse frequency and wavelength data
    pulse_frequencies = []
    wavelengths = []

    data = [{'pulseFrequency': 10}, {'pulseFrequency': 15}, {'pulseFrequency': 20}]

    
    
    for data_point in data:
        pulse_frequency = data_point['pulseFrequency']
        wavelength = 45 / pulse_frequency

        pulse_frequencies.append(pulse_frequency)
        wavelengths.append(wavelength)

    # Create a line chart for pulse frequency
    plt.figure(figsize=(10, 6))
    plt.plot(pulse_frequencies, label='Pulse Frequency (Hz)')
    plt.xlabel('Time')
    plt.ylabel('Pulse Frequency (Hz)')
    plt.title('Pulse Frequency Over Time')
    plt.grid(True)
    plt.legend()

    # Create a line chart for wavelength
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, label='Wavelength (meters)')
    plt.xlabel('Time')
    plt.ylabel('Wavelength (meters)')
    plt.title('Wavelength Over Time')
    plt.grid(True)
    plt.legend()

    # Show the generated charts
    plt.show()
data = {
    'pulseAmplitude': 1.0,
    'pulseFrequency': 10.0,
    'magneticFieldDirection': 5.0
}

process_data(data)  # This actually calls the function and executes its code
