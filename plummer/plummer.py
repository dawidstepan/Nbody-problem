import numpy.random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import pi
from numpy.linalg import norm 

# Lab 1)
# Plummer model - position 
def plummer_position(number_of_particles,mass_cutoff):
    
    def calculate_radius():
        random_mass_fraction = np.random.uniform(1/number_of_particles,mass_cutoff,(number_of_particles,1))
        radius = 1.0/pow(pow(random_mass_fraction,-2.0/3.0)-1.0,1.0/2.0)
        return radius
    
    def position_spherical():
        radius = calculate_radius()
        theta  = numpy.arccos(np.random.uniform(-1.0,1.0,(number_of_particles,1)))
        phi    = np.random.uniform(0.0,2*pi,(number_of_particles,1))
        return (radius,theta,phi)   
    
    def spherical_to_cartesian(vec):
        x = vec[0]*np.sin(vec[1])*np.cos(vec[2])
        y = vec[0]*np.sin(vec[1])*np.sin(vec[2])
        z = vec[0]*np.cos(vec[1])
        return (x,y,z)
                
    return spherical_to_cartesian(position_spherical())

# Plots - mass cut off = 0.99
N = [100,1000,10000,100000]
fig = plt.figure(figsize=(20,5))
for i  in enumerate(N):
    ax = fig.add_subplot(1,len(N),i[0]+1, projection='3d')
    x,y,z = plummer_position(i[1],0.99) 
    ax.scatter3D(x,y,z,lw=0,s=4)
    plt.title("Plummer model for N = %1.0f"%i[1],fontsize=12,x=0.5,y=1.05)
plt.savefig("plummer_1.png")
plt.show()

# Plots mass cut off = 1/N
fig = plt.figure(figsize=(20,5))
for i  in enumerate(N):
    ax = fig.add_subplot(1,len(N),i[0]+1, projection='3d')
    x,y,z = plummer_position(i[1],1/i[1])
    ax.scatter3D(x,y,z,lw=0,s=4)
    plt.title("Plummer model for N = %1.0f"%i[1],fontsize=12,x=0.5,y=1.05)
plt.savefig("plummer_2.png")
plt.show()


# Plummer model
def plummer(number_of_particles,mass_cutoff, scaling = 1.):
    def calculate_radius():
        random_mass_fraction = np.random.uniform(1/number_of_particles,mass_cutoff,(number_of_particles,1))
        radius = 1.0/pow(pow(random_mass_fraction,-2.0/3.0)-1.0,1.0/2.0)
        return radius
    
    def position_spherical():
        radius = calculate_radius()
        theta  = numpy.arccos(np.random.uniform(-1.0,1.0,(number_of_particles,1)))
        phi    = np.random.uniform(0.0,2*pi,(number_of_particles,1))
        return (radius,theta,phi)   
    
    def spherical_to_cartesian(vec):
        x = vec[0]*np.sin(vec[1])*np.cos(vec[2])
        y = vec[0]*np.sin(vec[1])*np.sin(vec[2])
        z = vec[0]*np.cos(vec[1])
        return (x,y,z)
    
    def velocity_dens(number_of_particles):
        Nstore = 0
        x_value = numpy.zeros(0)
        y_value = numpy.zeros(0)
        while (Nstore < number_of_particles):
            x = np.random.uniform(0,1.0,(number_of_particles-Nstore))
            y = np.random.uniform(0,0.1,(number_of_particles-Nstore))
            p = (x**2)*pow(1.0-x**2.0,7.0/2.0)
            compare = y <= p
            x_value = np.concatenate((x_value,x.compress(compare)))
            y_value = np.concatenate((y_value,y.compress(compare)))
            Nstore  = len(x_value)
        return (np.atleast_2d(x_value).transpose(),np.atleast_2d(y_value).transpose())
        
    def velocity_spherical(radius):
        x,y = velocity_dens(number_of_particles)
        velocity = x*np.sqrt(2.0)*pow(1.0+radius*radius,-1.0/4.0)
        theta = np.arccos(np.random.uniform(-1.0,1.0, (number_of_particles,1)))
        phi = np.random.uniform(0.0,2*pi,(number_of_particles,1))
        return (velocity,theta,phi)
    
    def centering(vec,mass):
        weight = (vec*mass)/sum(mass)
        center = vec - weight
        return center

    mass = np.zeros((number_of_particles,1))+(1.0/number_of_particles)
    radius, theta, phi = position_spherical()
    position = np.hstack(spherical_to_cartesian(position_spherical()))
    velocity = np.hstack(spherical_to_cartesian(velocity_spherical(radius)))
    position = position/scaling
    velocity = velocity/np.sqrt(1.0/scaling)
    
    # centering
    position = centering(position,mass)
    velocity = centering(velocity,mass)   
    return (mass, position, velocity)

# Lab 1 - g)
# Energy analysis
def calc_energy(N,m,x,v):
      kin = 0.;pot = 0.
      x_ij  = np.zeros_like(x[0])
      for i in range(N):
        kin += 1/2*m[i]*(norm(v[i])**2)
        l = [k for k in range(N) if k > i]
        for j in l:
          x_ij = x[j]-x[i]   
          pot  -= (m[i]*m[j])/norm(x_ij)
      return (kin+pot)
  
def hist_energy(N,repeat):    
    result = np.zeros(repeat)
    for i in range(repeat):
        m,x,v = plummer(N,0.99,16.0/(3*pi))
        result[i] = calc_energy(N,m,x,v)
    return result

# Histogram of energies # it takes some minutes 
nbody = 5
obs   = 1000000 
result = hist_energy(nbody,obs) 

# log-log plot
import pylab as pl
data = -result
pl.figure(figsize=(8,5))
pl.hist(data, bins=np.logspace(np.log(0.001),np.log(4.0),100))
pl.gca().set_xscale("log")
pl.gca().set_yscale("log")
pl.title('log-log histogram of sampled energies E for a scaled \n' +\
         'Plummer model of N = %1.f' %nbody + ' and %1.f' %obs + ' observations')
pl.ylabel("log N")
pl.xlabel("log Energy E")
plt.grid()
plt.savefig("histogram_plummer.png")
pl.show()

# semi-log plot
import pylab as pl
data = result
pl.figure(figsize=(8,5))
pl.hist(data,100)
pl.gca().set_yscale("log")
pl.title('semi-log histogram of sampled energies E for a scaled \n' +\
         'Plummer model of N = %1.f' %nbody + ' and %1.f' %obs + ' observations')
pl.ylabel("log N")
pl.xlabel("Energy E")
plt.grid()
plt.savefig("histogram_plummer_semi.png")
pl.show()


