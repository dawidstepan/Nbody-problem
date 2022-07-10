# -*- coding: utf-8 -*-
"""
@author: Dawid Stepanovic
"""
from matplotlib import pyplot as plt
import numpy as np
import coconv as con
from numpy.linalg import norm 
np.set_printoptions(precision=20)

# sidereal orbit parameter retrieved from: https://astronomy.swin.edu.au/cosmos/s/Sidereal+Period
solar      = 1.988544*10**30                   # in kg
jup        = 1898.13*10**24                    # in kg
ear        = 5.97219*10**24                    # in kg
sat        = 568.319*10**24                    # in kg
ven        = 4.86732*10**24                    # in kg

# parameter for assignment 2a:
venus     = np.array((0.72333199,0.00677323,0.,0.,0.,0.,ven/solar,224.701))
earth     = np.array((1.00000011,0.01671022,0.,0.,180.,0.,ear/solar,365.2568983263281))
jupiter   = np.array((5.20336301,0.04839266,0.,0.,0.,0.,jup/solar,4332.589))
saturn	   = np.array((9.53707032,0.05415060,0.,0.,180.,0.,sat/solar,10759.22))

# parameter for assignment 2b:
#venus      = np.array((0.72333199,0.00677323,3.39471,76.68069,131.53298,181.97973,ven/solar,224.701))
#earth      = np.array((1.00000011,0.01671022,0.00005,-11.26064,102.94719,100.46435,ear/solar,365.2568983263281))
#jupiter    = np.array((5.20336301,0.04839266,1.30530,100.55615,14.75385,34.40438,jup/solar,4332.589))
#saturn	   = np.array((9.53707032,0.05415060,2.48446,113.71504,92.43194,49.94432,sat/solar,10759.22))

method     = "rk4"                             # implemented methods: "rk4", "rk4_acc", "rk4_vel", "rk45_fehlberg"
nbody      = 3                                 # number of bodies        
planet     = np.zeros((nbody-1,8))             # define storage for two planets since the sun is in the program include 
planet[0]  = venus                             # venus parameter
planet[1]  = earth                             # earth parameter 
period     = planet[0,-1]                      # sidereal orbit period
step_val   = 0.01                              # initial step size
norbit_val = 10                                # number of orbits
hillradius = "hill"                            # implemented methods: "hill" (Hill radius), "laplace" (Laplace radius), False (neither of both)
shift_hill = 1                                 # 0 = change radius for inner planet (Hill radius) 1 = change radius for outer planet (Hill radius)
e_val      = 0.3                               # eccentricity
shift_eval = 1                                 # 0 = change ecc. for inner planet, 1 = change ecc. for outer planet
simulation = False                             # de/activating simulation
epsilon = 1e-6                                 # accuracy for the Runge-Kutta-Fehlberg method 

def Nbodyproblem(method,nbody,planet,period,step_val,norbit_val,hillradius,shift_hill,e_val,shift_eval,epsilon,simulation):
    
    # general setting
    dt      = period*step_val                   # time step 
    norbits = norbit_val                        # number of orbits
    au      = 1.49597870700*10**11              # in m
    G_SI    = 6.67408*10**-11                   # gravitational constant [in m^3/(kg*s^2)]
    daysec  = 86400.                            # in seconds
    msun    = 1.                                # mass [in solar mass]
    sun     = np.array((0,0,0,0,0,0,1))
    
    # convert the gravitational constant
    G_new = G_SI/au**3*solar*(daysec)**2
    #G_new = np.square(2*np.pi/365.2568983263281)
    
    # Hill/Laplace radius
    if hillradius == "laplace":
        radius = planet[0,0]*(planet[0,-2])**(2/5) 
        radius = 3*radius
        planet[shift_hill,0] = planet[0,0]+radius
    elif hillradius == "hill":
        radius = planet[0,0]*(planet[0,-2]/3)**(1/3)
        radius = 3*radius
        planet[shift_hill,0] = planet[0,0]+radius
    else:
        radius = 0;
        planet[shift_hill,0] = planet[shift_hill,0]+radius
    
    # shift eccentricity
    if e_val != 0:
        planet[shift_eval,1] = planet[shift_eval,1]+e_val
              
    # variable setting
    steps = int(100*period*norbits) #int(norbits*(period/dt))
    t = np.zeros(steps); dtstore = np.zeros(steps); energy = np.zeros(steps); moment  = np.zeros(steps) 
    rungelenz = np.zeros(steps); a_time = np.zeros(3)
    
    # r array stores the steps, the number of bodies in the system and the (x,y,z) array
    r  = np.zeros((steps,nbody,3))               
    v  = np.zeros_like(r)
    r0 = np.zeros_like(r)
    r1 = np.zeros_like(r)
    v0 = np.zeros_like(r)
    v1 = np.zeros_like(r)
    
    # gravitational constant*mass
    m = []
    m.append(G_new*msun)
    for i in range(np.shape(planet)[0]):  
         m.append(G_new*planet[i,-2])          
    m = np.array(m)
    
    # initial position and velocity 
    for i in range(np.shape(planet)[0]):
            r[0,i+1,:] = con.kepToCart(planet[i,:-1],sun[-1])[0]; 
            v[0,i+1,:] = con.kepToCart(planet[i,:-1],sun[-1])[1];
    
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
    
    # Runge-Lenz vector
    def diff(i,vr):
      vr_ij  = np.zeros_like(vr[0])
      l = [k for k in range(nbody) if k != i]
      for j in l:                                   
        vr_ij += vr[j]-vr[i]                                                                   
      return vr_ij  
    
    # min Runge-Lenz vector
    def diffmin(i,vr):
      vr_ij  = np.zeros(nbody)
      l = [k for k in range(nbody) if k != i]
      for j in l:                                   
        vr_ij[j] = norm(vr[j]-vr[i])                                                                   
      return vr_ij
    
    # Runge-Lenz vector
    def calc_rungelenz(r,v):
        RL = np.zeros_like(r[0])
        L  = np.zeros_like(r[0])
        for i in range(nbody):
            L = m[i]/G_new*np.cross(r[i],v[i])
            RL += np.cross(L,diff(i,v))-m[i]*diff(i,r)/norm(diff(i,r))
        return norm(RL)
    
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
    
    # initial Runge-Lenz vector
    rungelenz[0] = calc_rungelenz(r[0],v[0])
    
    # general settings for the RK-methods
    n = 0
    t[n] = dt
    dtstore[n] = dt
    
    
    k1 = np.zeros((nbody,6))          
    k2 = np.zeros_like(k1)          
    k3 = np.zeros_like(k1)              
    k4 = np.zeros_like(k1)           
    k5 = np.zeros_like(k1)           
    k6 = np.zeros_like(k1)  
    delta_r = np.zeros(nbody)       
    delta_v = np.zeros(nbody)
    
    # initial barycenter values 
    barycenter(r[0],v[0])
    
    if(method == "rk4"):
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
            # compute Runge-Lenz vector
            rungelenz[n+1] = calc_rungelenz(r[n+1],v[n+1])  
            # time step
            t[n+1] = t[n]+dt
            dtstore[n+1] = dt
            n = n+1 
            
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
            # compute Runge-Lenz vector
            rungelenz[n+1] = calc_rungelenz(r[n+1],v[n+1])  
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
            # compute Runge-Lenz vector
            rungelenz[n+1] = calc_rungelenz(r[n+1],v[n+1])  
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
            
    elif(method == "rk45_fehlberg"):
        while(t[n]<period*norbits):
                for i in range(0,nbody):
                    k1[:,0:3]  = dt*v[n,:] 
                    k1[i,3:6]  = dt* acceleration(i,r[n])
                
                    k2[:,0:3]  = dt*(v[n,:]+k1[:,3:6]/4)
                    k2[i,3:6]  = dt*acceleration(i,r[n]+k1[:,0:3]/4)
                     
                    k3[:,0:3]  = dt*(v[n,:]+3/32*k1[:,3:6]+9/32*k2[:,3:6])
                    k3[i,3:6]  = dt*acceleration(i,r[n]+3/32*k1[:,0:3]+9/32*k1[:,0:3])
                    
                    k4[:,0:3]  = dt*(v[n,:]+1932/2197*k1[:,3:6]-7200/2197*k2[:,3:6]+7296/2197*k3[:,3:6])
                    k4[i,3:6]  = dt*acceleration(i,r[n]+1932/2197*k1[:,0:3]-7200/2197*k2[:,0:3]+7296/2197*k3[:,0:3])
                    
                    k5[:,0:3]  = dt*(v[n,:]+439/216*k1[:,3:6]-8*k2[:,3:6]+3680/513*k3[:,3:6]-845/4104*k4[:,3:6])
                    k5[i,3:6]  = dt*acceleration(i,r[n]+439/216*k1[:,0:3]-8*k2[:,0:3]+3680/513*k3[:,0:3]-845/4104*k4[:,0:3])
                    
                    k6[:,0:3]  = dt*(v[n,:]-8/27*k1[:,3:6]+2*k2[:,3:6]-3544/2565*k3[:,3:6]+1859/4104*k4[:,3:6]-11/40*k4[:,3:6])
                    k6[i,3:6]  = dt*acceleration(i,r[n]-8/27*k1[:,0:3]+2*k2[:,0:3]-3544/2565*k3[:,0:3]+1859/4104*k4[:,0:3]-11/40*k4[:,0:3])
                                    
                    # next acceleration and position 
                    r0[n+1,i]  = r[n,i]+25/216*k1[i,0:3]+1408/2565*k3[i,0:3]+2197/4104*k4[i,0:3]-k5[i,0:3]/5
                    v0[n+1,i]  = v[n,i]+25/216*k1[i,3:6]+1408/2565*k3[i,3:6]+2197/4104*k4[i,3:6]-k5[i,3:6]/5
                    
                    r1[n+1,i]  = r[n,i]+16/135*k1[i,0:3]+6656/12825*k3[i,0:3]+28561/56430*k4[i,0:3]-9/50*k5[i,0:3]+2/55*k5[i,0:3]
                    v1[n+1,i]  = v[n,i]+16/135*k1[i,3:6]+6656/12825*k3[i,3:6]+28561/56430*k4[i,3:6]-9/50*k5[i,3:6]+2/55*k5[i,3:6]
                        
                    delta_r[i] = 0.84*((epsilon*dt)/norm(r0[n+1,i]-r1[n+1,i]))**(1/4)
                    delta_v[i] = 0.84*((epsilon*dt)/norm(v0[n+1,i]-v1[n+1,i]))**(1/4)
                    r[n+1,i]   = r0[n+1,i]
                    v[n+1,i]   = v0[n+1,i]
        
                # time step
                if(norm(v0[n+1,i]-v1[n+1,i])/dt <= epsilon):
                    r[n+1,i] = r0[n+1,i]
                    v[n+1,i] = v0[n+1,i]
                    barycenter(r[n+1],v[n+1])
                    t[n+1] = t[n]+dt
                    dtstore[n+1] = dt
                    n = n+1 
                    dt = np.min(delta_v[1:nbody])*dt
                else:  
                    dt = np.min(delta_v[1:nbody])*dt
    
                # compute total energy of system
                energy[n] = calc_energy(r[n],v[n])
                # compute angular momentum of system
                moment[n] = calc_moment(r[n],v[n])
                # compute Runge-Lenz vector
                rungelenz[n] = calc_rungelenz(r[n],v[n]) 
                
    # extract relevant non-zero values   
    r=r[:n+1] 
    t=t[:n+1]
    dtstore=dtstore[:n+1]
    energy=energy[:n+1]
    moment=moment[:n+1,]
    rungelenz=rungelenz[:n+1]
    
    # exit the simulation by closing the figure
    def handle_close(evt):
    	    raise SystemExit('Closed figure, exit program.')
    
    # simulation
    if simulation: 	
        fig = plt.figure(figsize=(5,5))
        fig.canvas.mpl_connect('close_event', handle_close)
                
        for m in range(n):
                plt.cla()
                plt.title(r"Runge-Kutta method - orbit""\n"\
                          "step size = %1.2f [orbital period (2$\pi$)], " %dt +\
                          "\n""$3\Delta_c$ = %1.2f [au], " %radius + "$\Delta$e =\
                          %1.2f " %e_val,fontsize=11)
                plt.xlabel(r"$x~[in~au]$",fontsize=11)
                plt.ylabel(r"$y~[in~au]$",fontsize=11)
                plt.grid()
                plt.scatter(r[1:m,1,0],r[1:m,1,1],s=1,color='blue')
                plt.scatter(r[1:m,1,0],r[1:m,1,1],s=1,color='blue')
                plt.scatter(r[1:m,2,0],r[1:m,2,1],s=1,color='blue')
                plt.scatter(r[1:m,2,0],r[1:m,2,1],s=1,color='blue')
                plt.scatter(r[m,0,0],r[m,0,1],s=40,color='yellow')
                plt.scatter(r[m,1,0],r[m,1,1],s=40,color='darkblue')
                plt.scatter(r[m,2,0],r[m,2,1],s=40,color='darkblue')
                plt.pause(0.00000000001)           
    else:        
        # plots
        # figure 1 - orbit
        plt.figure(figsize=(22,5))
        plt.subplots_adjust(wspace = 0.45,left=1/9, right=1-1/9, bottom=1/4.8, top=1-1/7.5)
        plt.subplot(141)
        for i in range(nbody):
            plt.plot(r[:,i,0],r[:,i,1],color=(0,i/nbody,0.6,1),linewidth=.5,\
                     marker='h',markersize=1, markevery=1)
        plt.title(r"Runge-Kutta - orbit""\n"\
                  "init. step size = %1.2f [orbital period (2$\pi$)], " %step_val +\
                  "\n""$3\Delta_c$ = %1.2f [au], " %radius + "$\Delta$e = %1.2f " %e_val,fontsize=11)
        plt.xlabel(r"$x~[in~au]$",fontsize=11)
        plt.ylabel(r"$y~[in~au]$",fontsize=11)
        #plt.ylim([-.02,.02])
        #plt.xlim([-.02,.02])
        plt.ylim([-1.4,1.1])
        plt.xlim([-1.1,1.4])
        plt.grid()
        
        col = 'dimgrey' 
        plt.annotate('mean total energy: '+str(np.mean(energy))\
                     , xy=(0.0, -0.25), xycoords='axes fraction',color=col)
        plt.annotate('mean total angular momentum: '+str(np.mean(moment))\
                     , xy=(0.0, -0.3), xycoords='axes fraction',color=col)
        plt.annotate('steps: '+str(n)+ ', min. step [in days]: '+str(np.min(dtstore))+\
                     ', max. step [in days]: '+str(np.max(dtstore)),\
                     xy=(1.5, -0.25), xycoords='axes fraction',color=col) 
        plt.annotate('period of inner orbit [in days]: '+str(period)\
                     , xy=(1.5, -0.3), xycoords='axes fraction',color=col)
        
        # figure 2 - total energy
        plt.subplot(142)
        plt.semilogy(t/period,abs(energy-energy[0])/abs(energy[0]),'k',color='b')
        plt.title(r"Runge-Kutta - total energy""\n"\
                  "init. step size = %1.2f [orbital period (2$\pi$)], " %step_val +\
                 "\n""$3\Delta_c$ = %1.2f [au], " %radius + "$\Delta$e = %1.2f " %e_val,fontsize=11)
        plt.xlabel(r"$time~[{\rm in~orbital~period}]$",fontsize=11)
        plt.ylabel(r"$\mathcal{log}~\frac{|E-E(0)|}{|E(0)|}$",fontsize=13.5)
        plt.ylim([1e-10,1000])
        plt.grid()
    
        # figure 3 - total angular momentum
        plt.subplot(143)
        plt.semilogy(t/period,abs(moment-moment[0])/abs(moment[0]),'k',color='b')
        plt.title(r"Runge-Kutta method - total angular momentum""\n"\
                  "init. step size = %1.2f [orbital period (2$\pi$)], " %step_val +\
                 "\n""$3\Delta_c$ = %1.2f [au], " %radius + "$\Delta$e = %1.2f " %e_val,fontsize=11)
        plt.xlabel(r"$time~[{\rm in~orbital~period}]$",fontsize=11)
        plt.ylabel(r"$\mathcal{log}~\frac{|L-L(0)|}{|L(0)|}$",fontsize=13.5)
        plt.ylim([1e-10,1000])
        plt.grid()
        
        # figure 4 - total Runge-Lenz vector
        plt.subplot(144)
        plt.semilogy(t/period,rungelenz,'k',color='b')
        plt.title(r"Runge-Kutta method - total Runge-Lenz vector""\n"\
                  "init. step size = %1.2f [orbital period (2$\pi$)], " %step_val +\
                  "\n""$3\Delta_c$ = %1.2f [au], " %radius + "$\Delta$e = %1.2f " %e_val,fontsize=11)
        plt.xlabel(r"$time~[{\rm in~orbital~period}]$",fontsize=11)
        plt.ylabel(r"$\mathcal{log~Runge-Lenz~vector}$",fontsize=13.5)
        plt.ylim([1e-10,1000])
        plt.grid()
        
        # save figures as pdf
        plt.savefig("rungekutta_"+str(step_val)+"_"+str(e_val)+"_"+str(hillradius)+"_"+str(method)+".pdf")
        plt.show()
   
Nbodyproblem(method,nbody,planet,period,step_val,norbit_val,hillradius,shift_hill,e_val,shift_eval,epsilon,simulation)
    