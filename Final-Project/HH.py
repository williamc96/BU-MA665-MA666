#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:53:18 2020

@author: nt
"""

import numpy as np

def HH(A_tonic, A_sin, A_noise, T, K): # T input in [s]
    T0 = T*1000                        # Convert [s] --> [ms]
    dt = 0.01                          # Units of [ms]
    t = np.arange(0, T0, dt)           # [ms]
    
    N = t.size
    
    I = np.zeros([K,N])
    I_sin = A_sin * np.sin(2 * np.pi * 8 * t/1000)
    
    V = np.zeros([K,N])
    
    for i in range(K):
        I_noise = np.random.normal(0, A_noise, N)
        I_trial = I_noise + I_sin + A_tonic
        I[i,] = I_trial
        
        [v, _, _, _, _] = singleTrial(I_trial, T0, gM=0)
        V[i,] = v.flatten()
        
    spike_train = count_spikes(V)
    
    return [V, spike_train, I, t/1000]  # Output in [s]


def singleTrial(I0,T0,gM=0):
    dt = 0.01;
    T  = int(np.ceil(T0/dt))  # [ms]
    gNa0 = 120   # [mS/cm^2]
    ENa  = 115;  # [mV]
    gK0  = 36;   # [mS/cm^2]
    EK   = -12;  # [mV]
    gL0  = 0.3;  # [mS/cm^2]
    EL   = 10.6; # [mV]

    t = np.arange(0,T)*dt
    V = np.zeros([T,1])
    m = np.zeros([T,1])
    h = np.zeros([T,1])
    n = np.zeros([T,1])
    mKM = np.zeros([T,1])

    V[0]=-70.0
    m[0]=0.05
    h[0]=0.54
    n[0]=0.34
    mKM[0] = 0

    for i in range(0,T-1):
        V[i+1] = V[i] + dt*(-gM*mKM[i]*(V[i]+95) + gNa0*m[i]**3*h[i]*(ENa-(V[i]+65)) + gK0*n[i]**4*(EK-(V[i]+65)) + gL0*(EL-(V[i]+65)) + I0[i]);
        m[i+1] = m[i] + dt*(alphaM(V[i])*(1-m[i]) - betaM(V[i])*m[i]);
        h[i+1] = h[i] + dt*(alphaH(V[i])*(1-h[i]) - betaH(V[i])*h[i]);
        n[i+1] = n[i] + dt*(alphaN(V[i])*(1-n[i]) - betaN(V[i])*n[i]);
        mKM[i+1] = mKM[i] + dt*(alphaKM(V[i])*(1-mKM[i]) - betaKM(V[i])*mKM[i])
    return V,m,h,n,t



def count_spikes(V):
    [K, N] = V.shape
    spikes = np.zeros(V.shape)
    threshold = 20
    
    for j in range(K):
        for i in range(N-1):
            if (V[j,i]<threshold) & (V[j,i+1]>threshold):
                spikes[j,i] = 1
    return spikes



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

def alphaKM(V):
    return 1.5 * 0.02 / (1 + np.exp((-V-20)/5))

def betaKM(V):
    return 1.25* 0.01 * np.exp((-V-43)/18)







