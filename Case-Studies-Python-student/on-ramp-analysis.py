import scipy.io as sio               # Import packages to read data, do analysis, and plot it.
import pylab as pl
# %matplotlib inline

mat = sio.loadmat('matfiles/sample_data.mat') # Load the example data set.
t   = mat['t'][0]                    # Get the values associated with the key 't' from the dictorary.
LFP = mat['LFP'][0]                  # Get the values associated with the key 'LFP' from the dictorary

                                     # Print useful information about the data.
print("Sampling frequency is " + str( 1/(t[2]-t[1]))  + ' Hz.')
print("Total duration of recording is " + str(t[-1]) + ' s.')
print("Dimensions of data are " + str(shape(LFP)) + ' data points.')

pl.initial_time_interval = t < 5        # Choose an initial interval of time, from onset to 5 s,
                                     # ... and plot it.
pl.plot(t[initial_time_interval], LFP[initial_time_interval])
pl.xlabel('Time [s]')
pl.ylabel('LFP')
pl.title('Initial interval of LFP data');
pl.show()
