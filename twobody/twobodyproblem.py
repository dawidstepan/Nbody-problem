# -*- coding: utf-8 -*-
"""
@author: Dawid Stepanovic
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm 
np.set_printoptions(precision=20)

def twobodyproblem(step_val,e_val,a_val,norbit_val,simulation=False):
    
    # general setting
    nbody   = 2                                 # number of celestial objects 
    dt      = step_val                          # time step 
    e       = e_val                             # eccentricity
    a       = a_val                             # semi-major axis [au]
    norbits = norbit_val                        # number of orbits
    tperiod = 2*np.pi                           # orbital period
    solar   = 1.988544*10**30                   # in kg
    jup     = 1898.13*10**24                    # in kg
    au      = 1.49597870700*10**11              # in m
    G_SI    = 6.67408*10**-11                   # gravitational constant [in m^3/(kg*s^2)]
    k_const = (2*np.pi/365.2568983263281)       # 2pi/(year[in days]) 
    yearsec = 86400.*365.2568983263281          # in seconds
    daysec  = 86400.                            # in seconds
    earth   = 5.97219*10**24                    # in kg
    msun    = 1.                                # mass [in solar mass]
    mplanet = jup/solar                         # jupiter mass [in solar mass]
    
    # convert the gravitational constant G to 1 to scale the period to 2*pi 
    G_new  = G_SI/au**3*solar*(daysec)**2/k_const**2
    #G_new = G_SI/au**3*solar*(yearsec)**2/(2*np.pi)**2

    
    m  = G_new*np.array([msun,mplanet])         # gravitational constant*mass 
    r0 = a*(1+e)                                # initial position [au]
    v0 = np.sqrt(((m[0]+m[1])/a)*((1-e)/(1+e))) # initial velocity [au]
     
    # variable setting
    steps = norbits*int(tperiod/dt)    
    t = np.zeros(steps); energy = np.zeros(steps); moment  = np.zeros(steps)
    # r array stores the steps, the number of bodies in the system and the (x,y,z) array
    r = np.zeros((steps,nbody,3))               
    v = np.zeros_like(r)
    
    # initial position and velocity 
    r[0,1,0]  = r0; 
    v[0,1,1]  = v0;
    
    # acceleration for j != i
    def acceleration(i,r):
      a     = np.zeros_like(r[0])
      r_ij  = np.zeros_like(r[0])
      l = [k for k in range(nbody) if k != i]
      for j in l:                                   
        r_ij = r[j]-r[i]                                                                   
        a += m[j]/(norm(r_ij))**3*r_ij                                                             
      return a
    
    # total (kinetic + potential) energy 
    def calc_energy(r,v):
      kin = 0.;pot = 0.
      r_ij  = np.zeros_like(r[0])
      for i in range(nbody):
        kin += 1/2*m[i]*(norm(v[i])**2)
        l = [k for k in range(nbody) if k > i]
        for j in l:
          r_ij = r[j]-r[i]   
          pot  -= (m[i]*m[j])/norm(r_ij)
      return (kin+pot)/G_new
    
    # total angular momentum 
    def calc_moment(r,v):
      L = np.zeros_like(r[0])
      for i in range(nbody):   
        L += m[i]/G_new*np.cross(r[i],v[i])       
      return norm(L)
    
    # initial total energy
    energy[0] = calc_energy(r[0],v[0])
    
    # initial angular momentum 
    moment[0] = calc_moment(r[0],v[0])
    
    # first step of leapfrog integration - acceleration for initial velocity by RK 2
    for i in range(nbody):                        
      acc = acceleration(i,r[0]+1/2*dt*(-1/2*dt*v[0]))
    
      # velocity at t[n-1/2] for n=1
      v[0,i] -= 1/2*dt*acc 
        
    # regular steps of leapfrog integration
    for n in range(steps-1):
      
      # velocities from n-1/2 to n+1/2 
      for i in range(nbody):
        # acceleration
        acc = acceleration(i,r[n])
        v[n+1,i] = v[n,i] + dt*acc
        # position change from n to n+1/2 by velocities at n+1/2
        r[n+1,i] = r[n,i] + 1/2*dt*v[n+1,i]
        
      # compute total energy of system
      energy[n+1] = calc_energy(r[n+1],v[n+1])
    
      # compute angular momentum of system
      moment[n+1] = calc_moment(r[n+1],v[n+1])
    
      # finishing leapfrog integration step to the position n+1 using velocities at n+1/2
      for i in range(nbody):
        r[n+1,i] += 1/2*dt*v[n+1,i]
      
      # time step
      t[n+1] = t[n] + dt

    # check total energy and total anuglar momentum 
    mu = msun*mplanet/(msun+mplanet)            # reduced mass
    M  = (msun + mplanet)                       # total mass
    
    # specific orbital energy
    energy_check = -mu*G_new*M/(2*a)
    # specific angular momentum
    moment_check =  mu*np.sqrt((1-e**2)*G_new*M*a)

    # exit the simulation by closing the figure
    def handle_close(evt):
	    raise SystemExit('Closed figure, exit program.')
    
    # simulation
    if simulation: 	
        fig = plt.figure(figsize=(7,5))
        fig.canvas.mpl_connect('close_event', handle_close)
        for i in range(nbody):
            x = r[:,i,0]-r[:,0,0]
            y = r[:,i,1]-r[:,0,1]
        for m in range(steps):
            plt.cla()
            plt.ylim([-1.5,1.5])
            plt.xlim([-1.2,1.75])
            plt.title(r"leapfrog method - orbit""\n"\
                      "step size = %1.2f [orbital period (2$\pi$)], " %dt +\
                      "\n""a = %1.2f [au], " %a + "e = %1.2f " %e,fontsize=11)
            plt.xlabel(r"$x~[in~au]$",fontsize=11)
            plt.ylabel(r"$y~[in~au]$",fontsize=11)
            plt.grid()
            plt.scatter(x[1:m],y[1:m],s=1,color='blue')
            plt.scatter(x[m],y[m],s=40,color='darkblue')
            plt.pause(0.00001)           
    else:        
        # plots
        # figure 1 - orbit
        plt.figure(figsize=(20,5))
        plt.subplots_adjust(wspace = 0.275,left=1/8, right=1-1/8, bottom=1/4.8, top=1-1/7.5)
        plt.subplot(131)
        for i in range(nbody):
            plt.scatter((r[:,i,0]-r[:,0,0]),(r[:,i,1]-r[:,0,1]),s=1,color='b')
        plt.title(r"leapfrog method - orbit""\n"\
                  "step size = %1.2f [orbital period (2$\pi$)], " %dt +\
                  "\n""a = %1.2f [au], " %a + "e = %1.2f " %e,fontsize=11)
        plt.xlabel(r"$x~[in~au]$",fontsize=11)
        plt.ylabel(r"$y~[in~au]$",fontsize=11)
        plt.grid()
        
        col = 'dimgrey' 
        plt.annotate('Mean total energy: '+str(np.mean(energy))\
                     , xy=(0.0, -0.25), xycoords='axes fraction',color=col)
        plt.annotate('Specific orbital energy: '+str(energy_check)\
                     , xy=(0.0, -0.3), xycoords='axes fraction',color=col)        
        plt.annotate('Mean total angular momentum: '+str(np.mean(moment))
                     , xy=(1.1, -0.25), xycoords='axes fraction',color=col)
        plt.annotate('Specific angular momentum: '+str(moment_check)\
                     , xy=(1.1, -0.3), xycoords='axes fraction',color=col)       
        
        # figure 2 - total energy
        plt.subplot(132)
        plt.semilogy(t/(2*np.pi),abs(energy-energy[0])/abs(energy[0]),'k',color='b')
        plt.title(r"leapfrog method - total energy""\n"\
                  "step size = %1.2f [orbital period (2$\pi$)], " %dt +\
                  "\n""a = %1.2f [au], " %a + "e = %1.2f " %e,fontsize=11)
        plt.xlabel(r"$time~[{\rm in~orbital~period~(2\pi)}]$",fontsize=11)
        plt.ylabel(r"$\frac{|E-E(0)|}{|E(0)|}$",fontsize=13.5)
        plt.ylim([1e-10,2])
        plt.grid()

        # figure 3 - total angular momentum
        plt.subplot(133)
        plt.semilogy(t/(2*np.pi),abs(moment-moment[0])/abs(moment[0]),'k',color='b')
        plt.title(r"leapfrog method - total angular momentum""\n"\
                  "step size = %1.2f [orbital period (2$\pi$)], " %dt +\
                  "\n""a = %1.2f [au], " %a + "e = %1.2f " %e,fontsize=11)
        plt.xlabel(r"$time~[{\rm in~orbital~period~(2\pi)}]$",fontsize=11)
        plt.ylabel(r"$\frac{|L-L(0)|}{|L(0)|}$",fontsize=13.5)
        plt.ylim([1e-10,2])
        plt.grid()
        # save figures in one pdf
        plt.savefig("leapfrog_"+str(step_val)+"_"+str(e_val)+".pdf")
        plt.show()
        
# run the function twobodyproblem(step_val,e_val,a_val,norbit_val)
for steps in [1/10,1/20,1/100]:
    for e_value in [0.,0.2,0.5]:
            twobodyproblem(steps,e_value,a_val=1,norbit_val=10,simulation=False)

# run the simulation for e = 0.5 and s = 0.10 
twobodyproblem(0.1,0.5,a_val=1,norbit_val=10,simulation=True)

