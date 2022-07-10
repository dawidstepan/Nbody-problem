# -*- coding: utf-8 -*-
"""
@author: Dawid Stepanovic - N-body Plummer
"""
from matplotlib import pyplot as plt
import numpy as np
import plummer_modul as pl
from numpy.linalg import norm 
from math import pi
np.set_printoptions(precision=20)

method     = "rk4"                             # implemented methods: "rk4", "rk4_acc", "rk4_vel"
nbody      = 100                               # number of bodies        
period     = 1                                 # period
step_val   = 1/500                             # initial step size
norbit_val = 1                                 # number of orbits

def Nbodyproblem_Plumer(method,nbody,period,step_val,norbit_val):
    
    # general setting
    dt      = step_val                         # time step 
    norbits = norbit_val                       # number of orbits
    au      = 1.49597870700*10**11             # in m
    G_SI    = 6.67408*10**-11                  # gravitational constant [in m^3/(kg*s^2)]
    daysec  = 86400.                           # in seconds
    solar   = 1.988544*10**30                  # in kg
    
    # convert the gravitational constant
    G_new = G_SI/au**3*solar*(daysec)**2/np.square(2*np.pi/365.2568983263281) # ~ 1
     
    # variable setting 
    if method == "rk4":
        steps = int(norbits*(period/dt))
    else:
        steps = int(norbits*(period/dt))*100
    
    t = np.zeros(steps); dtstore = np.zeros(steps); energy = np.zeros(steps); moment  = np.zeros(steps) 
    a_time = np.zeros(nbody)
    
    # r array stores the steps, the number of bodies in the system and the (x,y,z) array
    r  = np.zeros((steps,nbody,3))               
    v  = np.zeros_like(r)
    
    # gravitational constant*mass
    mass,pos,vel = pl.plummer(nbody,0.99,16.0/(3*pi))
    m = mass
    # initial position and velocity 
    for i in range(nbody):
            r[0,i,:] = pos[i]; 
            v[0,i,:] = vel[i];
    
    # acceleration for j != i
    def acceleration(i,r):
      a     = np.zeros_like(r[0])
      r_ij  = np.zeros_like(r[0])
      l = [k for k in range(nbody) if k != i]
      for j in l:                                   
        r_ij = r[j]-r[i]                                                                   
        a += m[j]/(norm(r_ij))**3*r_ij                                                             
      return a
    
    # jerk for j != i
    def acceleration_dot(i,r,v):
      a_dot = np.zeros_like(r[0])
      r_ij  = np.zeros_like(r[0])
      v_ij  = np.zeros_like(r[0])
      l = [k for k in range(nbody) if k != i]
      for j in l:                                   
        r_ij = r[j]-r[i] 
        v_ij = v[j]-v[i]                                                                                                                                   
        a_dot += m[j]/(norm(r_ij))**3*v_ij-(3*np.dot(r_ij,v_ij)/(norm(r_ij))**5*r_ij)                                                             
      return a_dot
    
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
    
    def diff(i,vr):
      vr_ij  = np.zeros_like(vr[0])
      l = [k for k in range(nbody) if k != i]
      for j in l:                                   
        vr_ij += vr[j]-vr[i]                                                                   
      return vr_ij  
    
    # Barycenter
    def barycenter(r,v):
        rbar = np.zeros(3)
        vbar = np.zeros(3)
        for i in range(nbody):
            rbar += m[i]*r[i] 
            vbar += m[i]*v[i]     
        for i in range(nbody):
                r[i] = r[i]-(rbar/G_new)/np.sum(m/G_new)
                v[i] = v[i]-(vbar/G_new)/np.sum(m/G_new)
        return rbar, vbar
    
    # initial total energy
    energy[0] = calc_energy(r[0],v[0])
    
    # initial angular momentum 
    moment[0] = calc_moment(r[0],v[0])
    
    # general settings for the RK-methods
    n = 0
    t[n] = dt
    dtstore[n] = dt
      
    k1 = np.zeros((nbody,6))          
    k2 = np.zeros_like(k1)          
    k3 = np.zeros_like(k1)              
    k4 = np.zeros_like(k1)           
    
    # initial barycenter values 
    barycenter(r[0],v[0])
    
    if(method == "rk4"):
        for n in range(steps-1):
            for i in range(nbody):
                k1[:,0:3] = dt*v[n,:] 
                k1[i,3:6] = dt* acceleration(i,r[n]) 
                
                k2[:,0:3] = dt*(v[n,:]+0.5*k1[:,3:6])
                k2[i,3:6] = dt*acceleration(i,r[n]+0.5*k1[:,0:3])
                 
                k3[:,0:3] = dt*(v[n,:]+0.5*k2[:,3:6])
                k3[i,3:6] = dt*acceleration(i,r[n]+0.5*k2[:,0:3])
                
                k4[:,0:3] = dt*(v[n,:]+k3[:,3:6])
                k4[i,3:6] = dt*acceleration(i,r[n]+k3[:,0:3])
                
                # next acceleration and position 
                r[n+1,i]  = r[n,i]+1/6*(k1[i,0:3]+2*k2[i,0:3]+2*k3[i,0:3]+k4[i,0:3])
                v[n+1,i]  = v[n,i]+1/6*(k1[i,3:6]+2*k2[i,3:6]+2*k3[i,3:6]+k4[i,3:6])
        
            # barycenter
            barycenter(r[n+1],v[n+1])       
            # compute total energy of system
            energy[n+1] = calc_energy(r[n+1],v[n+1])         
            # compute angular momentum of system
            moment[n+1] = calc_moment(r[n+1],v[n+1])         
            # time step
            t[n+1] = t[n]+dt
            dtstore[n+1] = dt
            n = n+1 
            print('t:',t[n])
            
    elif(method == "rk4_vel"):
        while(t[n]<period*norbits): 
            for i in range(nbody):
                k1[:,0:3] = dt*v[n,:] 
                k1[i,3:6] = dt* acceleration(i,r[n]) 
                
                k2[:,0:3] = dt*(v[n,:]+0.5*k1[:,3:6])
                k2[i,3:6] = dt*acceleration(i,r[n]+0.5*k1[:,0:3])
                 
                k3[:,0:3] = dt*(v[n,:]+0.5*k2[:,3:6])
                k3[i,3:6] = dt*acceleration(i,r[n]+0.5*k2[:,0:3])
                
                k4[:,0:3] = dt*(v[n,:]+k3[:,3:6])
                k4[i,3:6] = dt*acceleration(i,r[n]+k3[:,0:3])
                
                # next acceleration and position 
                r[n+1,i]  = r[n,i]+1/6*(k1[i,0:3]+2*k2[i,0:3]+2*k3[i,0:3]+k4[i,0:3])
                v[n+1,i]  = v[n,i]+1/6*(k1[i,3:6]+2*k2[i,3:6]+2*k3[i,3:6]+k4[i,3:6])
        
            # barycenter
            barycenter(r[n+1],v[n+1])       
            # compute total energy of system
            energy[n+1] = calc_energy(r[n+1],v[n+1])         
            # compute angular momentum of system
            moment[n+1] = calc_moment(r[n+1],v[n+1])         
            # time step
            for i in range(nbody):
                a_time[i] = norm(diff(i,r[n+1]))/norm(diff(i,v[n+1]))
    
            multiplier = 10
            if((np.min(step_val*a_time[a_time > 0])*multiplier)/(t[n]-t[n-1])> 10):
                dt = dt
            else:
                dt = np.min(step_val*a_time[a_time > 0])*multiplier
            t[n+1] = t[n]+dt
            dtstore[n+1] = dt
            n = n+1 
            print('t:',t[n])
    
    elif(method == "rk4_acc"):
        while(t[n]<period*norbits): 
            for i in range(nbody):
                k1[:,0:3] = dt*v[n,:] 
                k1[i,3:6] = dt* acceleration(i,r[n]) 
                
                k2[:,0:3] = dt*(v[n,:]+0.5*k1[:,3:6])
                k2[i,3:6] = dt*acceleration(i,r[n]+0.5*k1[:,0:3])
                 
                k3[:,0:3] = dt*(v[n,:]+0.5*k2[:,3:6])
                k3[i,3:6] = dt*acceleration(i,r[n]+0.5*k2[:,0:3])
                
                k4[:,0:3] = dt*(v[n,:]+k3[:,3:6])
                k4[i,3:6] = dt*acceleration(i,r[n]+k3[:,0:3])
                
                # next acceleration and position 
                r[n+1,i]  = r[n,i]+1/6*(k1[i,0:3]+2*k2[i,0:3]+2*k3[i,0:3]+k4[i,0:3])
                v[n+1,i]  = v[n,i]+1/6*(k1[i,3:6]+2*k2[i,3:6]+2*k3[i,3:6]+k4[i,3:6])
        
            # barycenter
            barycenter(r[n+1],v[n+1])       
            # compute total energy of system
            energy[n+1] = calc_energy(r[n+1],v[n+1])         
            # compute angular momentum of system
            moment[n+1] = calc_moment(r[n+1],v[n+1])         
            # time step
            for i in range(nbody):
                a_time[i] = norm(acceleration(i,r[n+1]))/norm(acceleration_dot(i,r[n+1],v[n+1]))
                
            multiplier = 1e8
            if((np.min(step_val*a_time[a_time > 0])*multiplier)/(t[n]-t[n-1])> 2):
                dt = dt
            else:
                dt = np.min(step_val*a_time[a_time > 0])*multiplier
            t[n+1] = t[n]+dt
            dtstore[n+1] = dt
            n = n+1 
            print('t:',t[n])
    
    # extract relevant non-zero values   
    r=r[:n+1] 
    t=t[:n+1]
    dtstore=dtstore[:n+1]
    energy=energy[:n+1]
    moment=moment[:n+1]
    
    # plots
    # figure 1 - orbit
    plt.figure(figsize=(18,5))
    plt.subplots_adjust(wspace = 0.45,left=1/9, right=1-1/9, bottom=1/4.8, top=1-1/7.5)
    plt.subplot(141)
    for i in range(nbody):
        plt.plot(r[0,i,0],r[0,i,1],color=(0,i/nbody,0.6,1),linewidth=.5,\
                 marker='h',markersize=1, markevery=1)
    
    plt.title(r"N-body Plummer model - initial position""\n"\
              "init. step size = %1.3f, " %step_val +\
               "N = %1.f " %nbody,fontsize=11)
    plt.xlabel(r"$x~[in~au]$",fontsize=11)
    plt.ylabel(r"$y~[in~au]$",fontsize=11)
    
    plt.ylim([-3,3])
    plt.xlim([-3,3])
    plt.grid()
    
    col = 'dimgrey' 
    plt.annotate('mean total energy: '+str(np.mean(energy))\
                 , xy=(0.0, -0.25), xycoords='axes fraction',color=col)
    plt.annotate('mean total angular momentum: '+str(np.mean(moment))\
                 , xy=(0.0, -0.3), xycoords='axes fraction',color=col)
    plt.annotate('steps: '+str(n),xy=(1.6, -0.25), xycoords='axes fraction',color=col) 
    plt.annotate('min. step size: '+str(np.min(dtstore))+\
                 ', max. step size: '+str(np.max(dtstore)),\
                 xy=(1.6, -0.3), xycoords='axes fraction',color=col) 
    
    plt.subplot(142)
    for i in range(nbody):
        plt.plot(r[n,i,0],r[n,i,1],color=(0,i/nbody,0.6,1),linewidth=.5,\
                 marker='h',markersize=1, markevery=1)
    plt.title(r"N-body Plummer model - end position""\n"\
              "init. step size = %1.3f, " %step_val +\
               "N = %1.f " %nbody,fontsize=11)
    plt.xlabel(r"$x~[in~au]$",fontsize=11)
    plt.ylabel(r"$y~[in~au]$",fontsize=11)
    
    plt.ylim([-3,3])
    plt.xlim([-3,3])
    plt.grid()
    
    # figure 3 - total energy
    plt.subplot(143)
    plt.semilogy(t/period,abs(energy-energy[0])/abs(energy[0]),'k',color='b')
    plt.title(r"N-body Plummer model - energy""\n"\
              "init. step size = %1.3f, " %step_val +\
               "N = %1.f " %nbody,fontsize=11)
    plt.xlabel(r"$time~[{\rm in~orbital~period}]$",fontsize=11)
    plt.ylabel(r"$\mathcal{log}~\frac{|E-E(0)|}{|E(0)|}$",fontsize=13.5)
    plt.ylim([1e-10,1000])
    plt.grid()
    
    # figure 4 - total angular momentum
    plt.subplot(144)
    plt.semilogy(t/period,abs(moment-moment[0])/abs(moment[0]),'k',color='b')
    plt.title(r"N-body Plummer model - angular momentum""\n"\
              "init. step size = %1.3f, " %step_val +\
               "N = %1.f " %nbody,fontsize=11)
    plt.xlabel(r"$time~[{\rm in~orbital~period}]$",fontsize=11)
    plt.ylabel(r"$\mathcal{log}~\frac{|L-L(0)|}{|L(0)|}$",fontsize=13.5)
    plt.ylim([1e-10,1000])
    plt.grid()
        
    # save figures as pdf
    plt.savefig("Nbody_plummer.png")
    plt.show()
       
    # 3D plots
    # figure 1 - initial position
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1, projection='3d')
    for i in range(nbody):
        ax.scatter3D(r[0,i,0],r[0,i,1],r[0,i,2],lw=0,s=4)
    plt.title(r"N-body Plummer model - initial position""\n"\
              "init. step size = %1.3f, " %step_val +\
               "N = %1.f " %nbody,fontsize=11)
    plt.xlabel(r"$x~[in~au]$",fontsize=11)
    plt.ylabel(r"$y~[in~au]$",fontsize=11)
    
    # figure 2 - end position
    ax = fig.add_subplot(1,2,2, projection='3d')
    for i in range(nbody):
        ax.scatter3D(r[n,i,0],r[n,i,1],r[n,i,2],lw=0,s=4)
    plt.title(r"N-body Plummer model - end position""\n"\
          "init. step size = %1.3f, " %step_val +\
           "N = %1.f " %nbody,fontsize=11)
    plt.xlabel(r"$x~[in~au]$",fontsize=11)
    plt.ylabel(r"$y~[in~au]$",fontsize=11)
    plt.savefig("Nbody_plummer_3D.pdf")
    plt.show()
    
Nbodyproblem_Plumer(method,nbody,period,step_val,norbit_val)