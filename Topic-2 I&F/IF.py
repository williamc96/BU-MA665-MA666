#######################################################################################
# The *IF* model.

import numpy as np
import matplotlib.pyplot as plt

def LIF(I,C,plot,R=0):                #Input important parameters, and an option to plot V vs t.

    Vth = 1;                     #Define the voltage threshold.
    Vreset = 0;                  #Define the reset voltage.
    dt=0.01                      #Set the timestep.
    timeSpan = 1000
    V = np.zeros([timeSpan,1])       #Initialize V.
    V[0]=0.2;                    #Set the initial condition.

    for k in range(0,timeSpan-1):       #March forward in time,
        V[k+1] = V[k] + dt*(I/C)-dt*V[k]*R/C*V[k]*R #Update the voltage,
        if V[k+1] > Vth:         #... and check if the voltage exceeds the threshold.
            V[k+1] = Vreset      #... if so, reset the voltage

    t = np.arange(0,len(V))*dt   #Define the time axis.

    if plot:                     #If the plot flag is on, plot V vs t.
        plt.plot(t,V)
        plt.show()

def izhikevich(I,c1,c2,c3,c4,c5):                #Input important parameters, and an option to plot V vs t.

    Vth = 1;                     #Define the voltage threshold.
    Vreset = 0;                  #Define the reset voltage.
    dt=0.01                      #Set the timestep.
    num_time_steps = 1000
    V = np.zeros([num_time_steps,1])       #Initialize V.
    V[0]=0.2;                    #Set the initial condition.
    u = 0

    for currTime in range(0,num_time_steps-1):       #March forward in time,
        V[currTime+1] = c1*(V[currTime]**2) + \
            c2*V[currTime] + \
            c3 - \
            u + \
            I
        u = u+c4*(c5*V[currTime]-u)
        if V[currTime+1] > Vth:         #... and check if the voltage exceeds the threshold.
            V[currTime+1] = Vreset      #... if so, reset the voltage

    t = np.arange(0,num_time_steps)*dt   #Define the time axis.

    plt.plot(t,V)
    plt.show()

izhikevich(1,.04,5,140,.2,.2)