import numpy as np
import matplotlib.pyplot as plt

def alphaM(V):
    return (2.5-0.1*(V+65)) / (np.exp(2.5-0.1*(V+65)) -1)

def betaM(V):
    return 4*np.exp(-(V+65)/18)

def alphaH(V):
    return 0.07*np.exp(-(V+65)/20)

def betaH(V):
    return 1/(np.exp(3.0-0.1*(V+65))+1)

def alphaN(V):
    return (0.1-0.01*(V+65)) / (np.exp(1-0.1*(V+65)) -1)

def betaN(V):
    return 0.125*np.exp(-(V+65)/80)

def HH(I0,T0,gNa0=120,gK0=36):
    dt = 0.01;
    T  = int(np.ceil(T0/dt))  # [ms]
    ENa  = 115;  # [mV]
    EK   = -12;  # [mV]
    gL0  = 0.3;  # [mS/cm^2]
    EL   = 10.6; # [mV]
    total_fires = 0

    t = np.arange(0,T)*dt
    V = np.zeros([T,1])
    m = np.zeros([T,1])
    h = np.zeros([T,1])
    n = np.zeros([T,1])

    V[0]=-70.0
    m[0]=0.05
    h[0]=0.54
    n[0]=0.34

    for i in range(0,T-1):
        V[i+1] = V[i] + dt*(gNa0*m[i]**3*h[i]*(ENa-(V[i]+65)) + gK0*n[i]**4*(EK-(V[i]+65)) + gL0*(EL-(V[i]+65)) + I0);
        m[i+1] = m[i] + dt*(alphaM(V[i])*(1-m[i]) - betaM(V[i])*m[i]);
        h[i+1] = h[i] + dt*(alphaH(V[i])*(1-h[i]) - betaH(V[i])*h[i]);
        n[i+1] = n[i] + dt*(alphaN(V[i])*(1-n[i]) - betaN(V[i])*n[i]);
        total_fires = total_fires+(V[i]<V[i-1] and V[i-1]>V[i-2] and V[i]>0)
    return V,m,h,n,t,total_fires

slope = .25
max_current = 20
currents = [k*slope for k in range(0,int(np.ceil(max_current/slope)))]
fires = [HH(current,5)[5][0] for current in currents]

plt.plot(currents,fires)
plt.xlabel('Current (I)')
plt.ylabel('Frequency (spike/ms)')
plt.title('f-I curve')
plt.grid(True, linewidth=0.5, color='k', linestyle='-')
plt.show()