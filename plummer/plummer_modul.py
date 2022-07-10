# -*- coding: utf-8 -*-
"""
@author: Dawid Stepanovic - Plummer modul
"""
import numpy.random
import numpy as np
from math import pi

# Plummer modul
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