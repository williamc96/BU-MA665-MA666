# Load modules we'll need.

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
# Load the data.

data = loadmat('EEG-2.mat')    # Load the EEG data
EEG = data['EEG'].reshape(-1)  # Extract the EEG variable
t = data['t'][0]               # ... and the t variablea
# Look at it.

plt.figure(1);
plt.plot(t,EEG)
plt.xlabel('Time [s]')
plt.ylabel('EEG');


# Compute the power spectrum using the FFT function.

#Define useful quantities
dt = t[2]-t[1]
f0 = 1/dt
N = np.size(EEG)
T = N*dt

x  = EEG
xf = np.fft.fft(x)                        # Compute Fourier transform of x
Sxx = 2 * dt ** 2 / T * (xf * xf.conj())  # Compute spectrum
Sxx = Sxx[0:int(N / 2)].real              # Ignore negative frequencies

# Define the frequency axis
df = 1/T                     # Determine frequency resolution
fNQ = f0 / 2                 # Determine Nyquist frequency
faxis = np.arange(0,fNQ,df)  # Construct frequency axis

# Plot the spectrum versus frequency.
plt.figure(2);
plt.plot(faxis, Sxx)
plt.xlim([0, 100])                          # Select frequency range
plt.xlabel('Frequency [Hz]')                # Label the axes
plt.ylabel('Power [$\mu V^2$/Hz]');

# Apply the Hanning taper and look at the data.

plt.figure(3);
x_tapered  = np.hanning(N) * x              # Apply the Hanning taper to the data.
plt.plot(t,x)
plt.plot(t,x_tapered);

# Apply the Hanning taper and look at the spectrum.
xf_tapered  = np.fft.fft(x_tapered)               # Compute Fourier transform of x.
Sxx_tapered = 2 * dt ** 2 / T * (xf_tapered * xf_tapered.conj())               # Compute the spectrum,
Sxx_tapered = np.real(Sxx_tapered[:int(N / 2)])  # ... and ignore negative frequencies.

plt.figure(4)
plt.semilogx(faxis,10*np.log10(Sxx))         # Plot spectrum of untapered signal.  
plt.semilogx(faxis,10*np.log10(Sxx_tapered)) # Plot spectrum vs tapered signal.
plt.xlim([faxis[1], 100])                    # Select frequency range,
plt.ylim([-70, 20])                          # ... and the power range.
plt.xlabel('Frequency [Hz]')                 # Label the axes
plt.ylabel('Power [$\mu V^2$/Hz]')

# Plot the spectrogram.

plt.figure(5)
Fs = 1 / dt               # Define the sampling frequency,
interval = int(Fs)        # ... the interval size,
overlap = int(Fs * 0.95)  # ... and the overlap intervals

                          # Compute the spectrogram
f0, t0, Sxx0 = spectrogram(
    EEG,                  # Provide the signal,
    fs=Fs,                # ... the sampling frequency,
    nperseg=interval,     # ... the length of a segment,
    noverlap=overlap)     # ... the number of samples to overlap,
plt.pcolormesh(t0, f0, 10 * np.log10(Sxx0),
               cmap='jet',
               shading='auto')# Plot the result
plt.colorbar()            # ... with a color bar,
# plt.ylim([0, 70])             # ... set the frequency range,
plt.xlabel('Time [s]')       # ... and label the axes
plt.ylabel('Frequency [Hz]');

plt.show()