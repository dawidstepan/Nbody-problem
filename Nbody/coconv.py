import numpy as np

def kepToCart(kepler, m0):

    k2    = np.square(0.01720209895)
    igrad = 180/(np.pi)
    
    M        = kepler[5]/igrad
    ecentric = M
    pos      = np.zeros(3)
    vel      = np.zeros(3)
    
    acc = 1.
    e   = kepler[1]
    
    while(acc > 1e-13):
        ecentric =  e * np.sin(ecentric) + M
        acc = np.absolute(ecentric - e * np.sin(ecentric) - M)

    nu = 2* np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(ecentric/2))
    a  = kepler[0]
    r  = a * (1-e * np.cos(ecentric))
    mu = k2 * (m0 + kepler[6])
    h  = np.sqrt(mu * a * (1 - np.square(e)))
    p  = a * (1 - np.square(e))

    node = (kepler[4])/igrad
    per  = (kepler[3])/igrad
    inc  = kepler[2]/igrad
    
    pos[0] = r * (np.cos(node) * np.cos(per + nu) - np.sin(node) * np.sin(per + nu) * np.cos(inc))
    pos[1] = r * (np.sin(node) * np.cos(per + nu) + np.cos(node) * np.sin(per + nu) * np.cos(inc))
    pos[2] = r * (np.sin(inc) * np.sin(per + nu))

    vel[0] = (pos[0] * h * e * np.sin(nu)) / (r * p) - (h/r) * (np.cos(node) * np.sin(per + nu) + np.sin(node) * np.cos(per + nu) * np.cos(inc))
    vel[1] = (pos[1] * h * e * np.sin(nu)) / (r * p) - (h/r) * (np.sin(node) * np.sin(per + nu) - np.cos(node) * np.cos(per + nu) * np.cos(inc))
    vel[2] = (pos[2] * h * e * np.sin(nu)) / (r * p) + (h/r) * (np.sin(inc) * np.cos(per + nu))

    return pos, vel
    
    
def cartToKep(x1,v1,x0,v0,m0,m1):

    k2    = np.square(0.01720209895)
    igrad = 180/np.pi
    
    kepler = np.zeros(7)
    
    u = np.zeros((3,3))
    r = x1-x0
    v = v1-v0

    cappaq = m0 + m1

    l       = np.cross(r,v)
    rscal   = np.linalg.norm(r)
    l2      = np.dot(l,l)
    lprojxy = np.sqrt(np.square(l[0]) + np.square(l[1]))
    lrl     = np.cross(v,l)

    lrl = [lrl[i]/(k2*cappaq)-r[i]/rscal for i in range(len(lrl))]
    oM  = np.arctan2(l[0],-l[1]) * igrad
    
    if(oM < 0):
        oM = oM + 360
        
    inc = np.arctan2(lprojxy, l[2]) * igrad
    e   = np.linalg.norm(lrl)

    if(e < 1e-13):
        e  = 0
        e2 = 0
        om = 0
    else:
        e2 = np.square(e)

    a = l2/(cappaq * k2 * (1-e2))
    node = np.zeros(3)
    node[0] = -l[1] * l[2]
    node[1] =  l[0] * l[2]
    node[2] = 0
    nscal   = np.linalg.norm(node)

    if(inc < 1e-13):
        oM = 0
        om = np.arccos(lrl[0]/np.linalg.norm(lrl))
        if(lrl[1] < 0):
            om = 360 - om * igrad
        else:
            om = om * igrad
    else:
        h     = np.cross(l,node)
        hnorm = np.linalg.norm(h)
        om    = np.arctan2(np.dot(lrl,h) * nscal, np.dot(lrl,node) * hnorm) * igrad
        if(om < 0):
            om = om + 360

    if(e < 1.2e-13 and inc <= 1e-13):
        oM  = 0
        om  = 0
        mmm = np.arctan2(r[1],r[0]) * igrad
        if (mmm < 0):
            mmm = mmm + 360
    elif (e < 1.2e-13 and inc > 1e-13):
        h = np.cross(l,node)
        hnorm = np.linalg.norm(h)
        for j in range(len(node)):
            u[j][0] = node[j]/nscal
            u[j][1] = h[j]/hnorm
            u[j][2] = l[j]/np.sqrt(l2)

        ru  = np.linalg.solve(u,r)
        tAn = np.arctan2(ru[1],ru[0])
        mmm = tAn * igrad
        if (mmm < 0):
            mmm = mmm + 360
    elif(inc < 1e-13 and e > 1e-13):
        h = np.cross(l,lrl)
        hnorm = np.linalg.norm(h)
        tAn   = np.arctan2(np.dot(h,r) * e, np.dot(lrl,r) * hnorm)
        cosen = (e + np.cos(tAn))/(1+e*np.cos(tAn))
        sinen = np.sqrt(1 - e2) * np.sin(tAn)/(1 + e * np.cos(tAn))
        eanom = np.arctan2(sinen,cosen)

        mmm = (eanom - e * sinen) * igrad
        if(mmm<0):
            mmm = mmm + 360
    else:
        h     = np.cross(l,lrl)
        hnorm = np.linalg.norm(h)
        for j in range(len(lrl)):
            u[j][0] = lrl[j]/e
            u[j][1] = h[j]/hnorm
            u[j][2] = l[j]/np.sqrt(l2)

        ru    = np.linalg.solve(u,r)
        tAn   = np.arctan2(ru[1],ru[0])
        cosen = (e + np.cos(tAn))/(1 + e * np.cos(tAn))
        sinen = np.sqrt(1 - e2) * np.sin(tAn)/(1 + e * np.cos(tAn))
        eanom = np.arctan2(sinen, cosen)

        mmm = (eanom - e * sinen) * igrad
        if(mmm < 0):
            mmm = mmm + 360

    if (om >= 360):
        om = om - 360
    if (oM >= 360):
        oM = oM - 360
    if (mmm >= 360):
        mmm = mmm - 360

    kepler = a,e,inc,om,oM,mmm,m1

    return kepler