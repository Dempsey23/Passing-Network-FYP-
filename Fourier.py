import matplotlib.pyplot as plt
import numpy as np

import TOT_VS_STOKE

# Define the passing matrix
passing_matrix = TOT_VS_STOKE.A_TOT

# Perform Fourier transform
fourier_transform = np.fft.fft2(passing_matrix)

# Magnitude and phase of Fourier coefficients
magnitude = np.abs(fourier_transform)
phase = np.angle(fourier_transform)

# Print the magnitude and phase of Fourier coefficients
print("Magnitude of Fourier coefficients:")
print(magnitude)
print("\nPhase of Fourier coefficients (radians):")
print(phase)
# Visualize magnitude matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(magnitude, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Magnitude')
plt.title('Magnitude of Fourier Coefficients')
plt.xlabel('Receiver Player')
plt.ylabel('Sender Player')
plt.xticks(range(passing_matrix.shape[1]), [f'Player {i+1}' for i in range(passing_matrix.shape[1])])
plt.yticks(range(passing_matrix.shape[0]), [f'Player {i+1}' for i in range(passing_matrix.shape[0])])
plt.grid(visible=False)
plt.show()

# Visualize phase matrix using quiver plot
plt.figure(figsize=(8, 6))
plt.quiver(np.arange(passing_matrix.shape[1]), np.arange(passing_matrix.shape[0]),
           np.cos(phase), np.sin(phase), magnitude, pivot='mid', cmap='hsv')
plt.colorbar(label='Magnitude')
plt.title('Phase of Fourier Coefficients')
plt.xlabel('Receiver Player')
plt.ylabel('Sender Player')
plt.xticks(range(passing_matrix.shape[1]), [f'Player {i+1}' for i in range(passing_matrix.shape[1])])
plt.yticks(range(passing_matrix.shape[0]), [f'Player {i+1}' for i in range(passing_matrix.shape[0])])
plt.grid(visible=False)
plt.show()

freq = np.fft.fftfreq(passing_matrix.shape[0])
plt.figure(figsize=(8, 6))
plt.plot(freq, np.mean(magnitude, axis=0), marker='o')
plt.title('Frequency Spectrum of Passing Network')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(fourier_transform.real, fourier_transform.imag, c=magnitude, cmap='viridis')
plt.colorbar(label='Magnitude')
plt.title('2D Fourier Space Plot')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.show()

time_intervals = np.arange(0, 10, 1)  # Example time intervals
plt.figure(figsize=(8, 6))
for i in range(len(time_intervals)):
    plt.plot(np.mean(magnitude, axis=0), label=f'Time Interval {i}')
plt.title('Temporal Evolution of Fourier Coefficients')
plt.xlabel('Receiver Player')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.show()