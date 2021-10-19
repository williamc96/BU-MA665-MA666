#######################################################################################
# 1.	Load the file:
#
#           EEG-2.mat, available on the this repository into Python. You’ll find two variables:
#
#  EEG = the EEG data, consisting of 1,000 trials, each observed for 1 s;
#  t = the time axis, which ranges from 0 s to 1 s.
# 
# These data have a similar structure to the data studied in class. To collect these data, a stimulus was presented to the subject at 0.25 s. Analyze these data for the presence of an evoked response. To do so, answer the following questions:
#  a.	What is the time between samples (dt) in seconds?
#  b.	Examine these data. Explain what you observe in pictures and words.  From your visual inspection, do you expect to find an ERP in these data?
#  c.	Compute the ERP for these data, and plot the results. Do you observe an ERP (i.e., times at which the 95% confidence intervals do not include zero)? Include 95% confidence intervals in your ERP plot, and label the axes. Explain in a few sentences the results of your analysis, as you would to a collaborator who collected these data.

import numpy as np                  # Import functions for calculations.
import matplotlib.pyplot as plt     # Import functions for plotting
from scipy.io import loadmat        # Import function to read data.

data = loadmat('data/EEG-2.mat')         # Load the data for this example.

ntrials= 1000
EEG = data['EEG']
t    = data['t'][0]

# a. What is the time between samples (dt) in seconds?
dt = [0.02]

# b. Examine these data. Explain what you observe in pictures and words. From your visual inspection, do you expect to find an ERP in these data?

trialOne = EEG[0]

plt.plot(t,trialOne)

plt.xlabel('Time [s]')           # Label the axes
plt.ylabel('Voltage [$\mu V$]')
plt.title('Trial run 1, File 1 (EEG-2)')  # ... provide a title
plt.ylim([-3, 3])
plt.plot([0.25, 0.25], [-4,4], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
#plt.show()
plt.figure()

plt.imshow(EEG,                                   # Image the data from condition A.
            #cmap='BuPu',                            # ... set the colormap (optional)
            extent=[t[0], t[-1], 1, ntrials],       # ... set axis limits (t[-1] represents the last element of t)
            aspect='auto',                          # ... set aspect ratio 
            origin='lower')                         # ... put origin in lower left corner
plt.xlabel('Time[s]')                              # Label the axes
plt.ylabel('Trial #')
plt.title('EEG data, file 1 (EEG-2)')  # ... provide a title
colorbar = plt.colorbar()                                     # Show voltage to color mapping
colorbar.set_label('Voltage (mV)')
plt.vlines(0.25, 1, 1000, 'k', lw=2)        # Indicate stimulus onset with line
#plt.show()
plt.figure()
# 
# #  c.	Compute the ERP for these data, and plot the results. Do you observe an ERP (i.e., times at which the 95% confidence intervals do not include zero)? Include 95% confidence intervals in your ERP plot, and label the axes. Explain in a few sentences the results of your analysis, as you would to a collaborator who collected these data.
# 
ERP = np.mean(EEG,0)
sd = np.std(EEG,0)    # Compute the std across trials
sdmn = sd / np.sqrt(ntrials)  # Compute the std of the mean
plt.plot(t,ERP)
plt.hlines(0, 0, 1, 'b')
plt.plot([0.25, 0.25], [-4,4], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
plt.fill_between(t,ERP-2*sdmn,ERP+2*sdmn, alpha=0.5, facecolor='#FF5733')
# plt.plot(t, ERP + 2 * sdmn, 'k:', lw=1)  # ... and include the upper CI\n
# plt.plot(t, ERP - 2 * sdmn, 'k:', lw=1)  # ... and the lower CI

plt.xlabel('Time [s]')           # Label the axes
plt.ylabel('Voltage [$\mu V$]')
plt.title('ERP of condition A, file 1 (EEG-2)')  # ... provide a title
#plt.show()
# 
#######################################################################################
# 2.	Load the file:
#
#           EEG-3.mat, available on the this repository into Python. You’ll find two variables:
#
#  EEG = the EEG data, consisting of 1,000 trials, each observed for 1 s;
#  t = the time axis, which ranges from 0 s to 1 s.
# 
# These data have a similar structure to the data studied in class. To collect these data, a stimulus was presented to the subject at 0.25 s. Analyze these data for the presence of an evoked response. To do so, answer the following questions:
#  a.	What is the time between samples (dt) in seconds?
#  b.	Examine these data. Explain what you observe in pictures and words.  From your visual inspection, do you expect to find an ERP in these data?
#  c.	Compute the ERP for these data, and plot the results. Do you observe an ERP (i.e., times at which the 95% confidence intervals do not include zero)? Include 95% confidence intervals in your ERP plot, and label the axes. Explain in a few sentences the results of your analysis, as you would to a collaborator who collected these data.


data = loadmat('data/EEG-3.mat')         # Load the data for this example.
ntrials= 1000
# print(data)
EEG = data['EEG']
t    = data['t'][0]

# a. What is the time between samples (dt) in seconds?
dt = [0.02]

# b. Examine these data. Explain what you observe in pictures and words. From your visual inspection, do you expect to find an ERP in these data?

trialOne = EEG[0]
plt.figure()
plt.plot(t,trialOne)

plt.xlabel('Time [s]')           # Label the axes
plt.ylabel('Voltage [$\mu V$]')
plt.title('Trial run 1, File 2 (EEG-3)')  # ... provide a title
plt.ylim([-10, 10])
plt.plot([0.25, 0.25], [-4,4], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
#plt.show()
plt.figure()

plt.imshow(EEG,                                   # Image the data from condition A.
            #cmap='BuPu',                            # ... set the colormap (optional)
            extent=[t[0], t[-1], 1, ntrials],       # ... set axis limits (t[-1] represents the last element of t)
            aspect='auto',                          # ... set aspect ratio 
            origin='lower')                         # ... put origin in lower left corner
plt.xlabel('Time[s]')                              # Label the axes
plt.ylabel('Trial #')
plt.title('EEG data, file 2 (EEG-3)')  # ... provide a title
plt.colorbar()                                     # Show voltage to color mapping
plt.vlines(0.25, 1, 1000, 'k', lw=2)        # Indicate stimulus onset with line
#plt.show()
# 
# #  c.	Compute the ERP for these data, and plot the results. Do you observe an ERP (i.e., times at which the 95% confidence intervals do not include zero)? Include 95% confidence intervals in your ERP plot, and label the axes. Explain in a few sentences the results of your analysis, as you would to a collaborator who collected these data.
# 
ERP = np.mean(EEG,0)
sd = np.std(EEG,0)    # Compute the std across trials
sdmn = sd / np.sqrt(ntrials)  # Compute the std of the mean
plt.figure()
plt.plot(t,ERP)
plt.hlines(0, 0, 1, 'b')
plt.plot([0.25, 0.25], [-4,4], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
plt.fill_between(t,ERP-2*sdmn,ERP+2*sdmn, alpha=0.5, facecolor='#FF5733')

plt.xlabel('Time [s]')           # Label the axes
plt.ylabel('Voltage [$\mu V$]')
plt.title('ERP of condition A, file 2 (EEG-3)')  # ... provide a title
plt.plot([0.25, 0.25], [-4,4], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
#plt.show()

# -------- Now, re-use your code from above! --------

#######################################################################################
# 3.	Load the file:
#
#           EEG-4.mat, available on the this repository into Python. You’ll find two variables:
#
#  EEG = the EEG data, consisting of 1,000 trials, each observed for 1 s;
#  t = the time axis, which ranges from 0 s to 1 s.
# 
# These data have a similar structure to the data studied in class. To collect these data, a stimulus was presented to the subject at 0.25 s. Analyze these data for the presence of an evoked response. To do so, answer the following questions:
#  a.	What is the time between samples (dt) in seconds?
#  b.	Examine these data. Explain what you observe in pictures and words.  From your visual inspection, do you expect to find an ERP in these data?
#  c.	Compute the ERP for these data, and plot the results. Do you observe an ERP (i.e., times at which the 95% confidence intervals do not include zero)? Include 95% confidence intervals in your ERP plot, and label the axes. Explain in a few sentences the results of your analysis, as you would to a collaborator who collected these data.


data = loadmat('data/EEG-4.mat')         # Load the data for this example.

# -------- Now, re-use your code from above! -------
ntrials= 1000
EEG = data['EEG']
t    = data['t'][0]

# a. What is the time between samples (dt) in seconds?
dt = [0.02]

# b. Examine these data. Explain what you observe in pictures and words. From your visual inspection, do you expect to find an ERP in these data?

trialOne = EEG[0]
plt.figure()
plt.plot(t,trialOne)

plt.xlabel('Time [s]')           # Label the axes
plt.ylabel('Voltage [$\mu V$]')
plt.title('Trial run 1, File 3 (EEG-4)')  # ... provide a title
plt.ylim([-10, 10])
plt.plot([0.25, 0.25], [-4,4], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
#plt.show()
plt.figure()

plt.imshow(EEG,                                   # Image the data from condition A.
            #cmap='BuPu',                            # ... set the colormap (optional)
            extent=[t[0], t[-1], 1, ntrials],       # ... set axis limits (t[-1] represents the last element of t)
            aspect='auto',                          # ... set aspect ratio 
            origin='lower')                         # ... put origin in lower left corner
plt.xlabel('Time[s]')                              # Label the axes
plt.ylabel('Trial #')
plt.colorbar()                                     # Show voltage to color mapping
plt.title('EEG data, file 3 (EEG-4)')  # ... provide a title
plt.vlines(0.25, 1, 1000, 'k', lw=2)        # Indicate stimulus onset with line
#plt.show()
plt.figure()
# 
# #  c.	Compute the ERP for these data, and plot the results. Do you observe an ERP (i.e., times at which the 95% confidence intervals do not include zero)? Include 95% confidence intervals in your ERP plot, and label the axes. Explain in a few sentences the results of your analysis, as you would to a collaborator who collected these data.
# 
ERP = np.mean(EEG,0)    # Compute the mean across trials (the ERP)
sd = np.std(EEG,0)    # Compute the std across trials
sdmn = sd / np.sqrt(ntrials)  # Compute the std of the mean
# plt.shadedErrorBar()

plt.plot(t,ERP)
plt.hlines(0, 0, 1, 'b')
plt.plot([0.25, 0.25], [-4,4], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
plt.fill_between(t,ERP-2*sdmn,ERP+2*sdmn, alpha=0.5, facecolor='#FF5733')

plt.xlabel('Time [s]')           # Label the axes
plt.ylabel('Voltage [$\mu V$]')
plt.title('ERP of condition A, file 3 (EEG-4)')  # ... provide a title
plt.plot([0.25, 0.25], [-4,4], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
#plt.show()

#######################################################################################
# 4. In the previous question, you considered the dataset EEG-4.mat and analyzed these data for the presence of an ERP.  
# To do so, you (presumably) averaged the EEG data across trials.  The results may have surprised you . . . 
# Modify your analysis of these data (in some way) to better illustrate the appearance (or lack thereof) of an evoked response.  
# Explain “what's happening” in these data as you would to a colleague or experimental collaborator.
plt.figure()
data = loadmat('data/EEG-4.mat')         # Load the data for this example.
EEGa = data['EEG']
nsamples = 500

EEGa = EEG[:,:250] #contains all the ERP
EEGb = EEG[:,250:] #our null hypothesis which doesn't contain ERP

EEG = np.vstack((EEGa, EEGb))  # Step 1. Merge EEG data from all trials

nTrials = 1000

N_resample = 3000

ERP0 = np.zeros((N_resample, int(nsamples/2)))

for k in range(0,N_resample):
    
    # i = np.random.randint(0, nTrials, size=nTrials)   # Create resampled indices.
    # EEG0_A = EEGa[i,:]    # Create a resampled EEG for "condition A".
    # absERP0_A = np.abs(EEG0_A).mean(0)    # Create a resampled ERP for "condition A".
    
    i = np.random.randint(0, nTrials*2, size=nTrials*2)   # Create resampled indices.
    EEG0_B = EEG[i,:]    # Create a resampled EEG for "condition A".
    ERP0[k] = np.abs(EEG0_B).mean(0)    # Create a resampled ERP for "condition A".
    
    # absDiff = absERP0_A - absERP0_B                # Compute the differenced ERP

    # ERP0[k] = np.max(np.abs(absDiff))  # Compute the stat from the resampled ERPs

# Compute 95% CI from the resampled ERPs
ERP0_sorted = np.sort(ERP0,0)             # Sort each column of the resampled ERP
ciL = ERP0_sorted[int(np.ceil(N_resample*0.05))]                    # Determine the lower CI
middle = ERP0_sorted[int(np.ceil(N_resample*0.5))]                    # Determine the lower CI
ciU = ERP0_sorted[int(np.ceil(N_resample*0.95))]                    # ... and the upper CI
mnA = np.abs(EEGa).mean(0)                                 # Determine the ERP for condition A
plt.plot(t[:250], mnA, 'k', lw=3)               # ... and plot it
plt.plot(t[:250], middle, 'k')               # ... and plot it
plt.fill_between(t[:250],ciL,ciU, alpha=0.5, facecolor='#FF5733')
# plt.plot(t[:250], ciL, 'k:')                    # ... and plot the lower CI
# plt.plot(t[:250], ciU, 'k:')                    # ... and the upper CI
plt.xlabel('Time [s]')                    # ... and label the axes
plt.ylabel('Voltage');
plt.show()


#######################################################################################
# 5. Compare the datasets EEG-3.mat and EEG-4.mat studied in the previous problems.  
# Use a bootstrap procedure to test the hypothesis that the evoked response is significantly different in the two datasets.

data3 = loadmat('data/EEG-3.mat')         # Load the data for this example.
EEG3 = data3['EEG']

data4 = loadmat('data/EEG-4.mat')         # Load the data for this example.
EEG4 = data4['EEG']
ntrials_3 = 1000
ntrials_4 = 1000


EEG = np.vstack((EEG3, EEG4))  # Step 1. Merge EEG data from all trials
np.random.seed(123)            # Fix seed for reproducibility

N_resample = 3000;
stat0 = np.zeros(N_resample)
for k in range(0,N_resample):
    
    i = np.random.randint(0, ntrials_3, size=ntrials_3)   # Create resampled indices.
    EEG0_A = EEG3[i,:]    # Create a resampled EEG for "condition A".
    absERP0_A = EEG0_A.mean(0)    # Create a resampled ERP for "condition A".
    
    i = np.random.randint(0, ntrials_3+ntrials_4, size=ntrials_3+ntrials_4)   # Create resampled indices.
    EEG0_B = EEG[i,:]    # Create a resampled EEG for "condition A".
    absERP0_B = EEG0_B.mean(0)    # Create a resampled ERP for "condition A".
    
    absDiff = absERP0_A - absERP0_B                # Compute the differenced ERP

    stat0[k] = np.max(np.abs(absDiff))  # Compute the stat from the resampled ERPs
    
mnA = np.mean(EEG4,0)          # Determine ERP for condition A
mnB = np.mean(EEG3,0)          # Determine ERP for condition B
mnD = mnA - mnB                # Compute the differenced ERP
stat = np.max(np.abs(mnD))

plt.figure()
y, x, _ = plt.hist(stat0, bins='auto')

# plt.vlines(stat, 0, 100)
plt.plot([stat, stat], [0,y.max()], 'k', lw=2)   # Add a vertical line to indicate the stimulus time
print(np.size(np.where(stat0>stat)) / stat0.size)
plt.show()
