# PCA model with ROF version.
# E(L, S, T, {\alpha}_l) = \int \lambda|S| + \gamma\|\nabla T\| (L-\sum_l\alpha_l\Beta_l\)^2 dx
# implementation based on Primal-Dual Hybrid Gradient 
from __future__ import division

import sys
import time
import numpy as np


def grad(u):
    # calculate gradient 
    l, m, n = u.shape
    G = np.zeros((3, l, m, n), u.dtype)
    G[0, :, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
    G[1, :, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
    G[2, :-1, :, :] = u[1:, :, :] - u[:-1, :, :]
    return G

def div(u):
    # calculate divergence
    _, l, m, n = u.shape
    Px = u[0,:,:,:]
    Py = u[1,:,:,:]
    Pz = u[2,:,:,:]

    fx = np.zeros((l,m,n), u.dtype)
    fx[:,:,1:] = Px[:,:,1:] - Px[:,:,:-1]
    fx[:,:,0] = Px[:,:,0]
    fy = np.zeros((l,m,n), u.dtype)    
    fy[:,1:,:] = Py[:,1:,:] - Py[:,:-1,:]
    fy[:,0,:] = Py[:,0,:]
    fz = np.zeros((l,m,n), u.dtype)
    fz[1:,:,:] = Pz[1:,:,:] - Pz[:-1,:,:]
    fz[0,:,:] = Pz[0,:,:]

    return fx+fy+fz

def computeEnergy(D, T, gamma, Alpha, Beta):
    
    l, m, n = D.shape
    sum_alpha_beta = np.dot(Beta, Alpha)
  
    # \nabla TV term energy  
    GR = np.linalg.norm(grad(T), axis =0)
    ET = GR.sum()
    # L-Ba
    sparse = D.reshape(l*m*n, 1) - T.reshape(l*m*n,1) - sum_alpha_beta
    EL = np.square(sparse).sum()

    E = EL/2 + gamma * ET

    return EL,ET,E

def ProxG(t, a, tau, D, Beta):
    l,m,n = D.shape
    lmn = l*m*n
    dl = (D-t).reshape(lmn,1)
    delta = np.dot(np.transpose(Beta), dl)
    p_v1 = -tau*tau/(1+2*tau)/(1+tau) * np.dot(Beta, delta)
    p_v2 = -tau/(1+2*tau)*np.dot(Beta, a)
    p_v3 = tau/(1+tau)*dl
    p = (p_v1 + p_v2 + p_v3).reshape(l,m,n)
    pt = p + t
    pa = tau/(1+3*tau)*delta + (1+2*tau)/(1+3*tau)*a
  
    return pt, pa

def ProxFS(t, gamma):
    _,l,m,n = t.shape
    max_abs_t = np.maximum(1, np.linalg.norm(t, axis=0)/gamma)
    pt = np.zeros((3,l,m,n), t.dtype)
    pt[0,:,:,:] = np.divide(t[0,:,:,:], max_abs_t)
    pt[1,:,:,:] = np.divide(t[1,:,:,:], max_abs_t) 
    pt[2,:,:,:] = np.divide(t[2,:,:,:], max_abs_t)

    return pt

def decompose(D, Beta, gamma, verbose):
    print('start decomposing in CPU')
    print(str(gamma))
    l,m, n = D.shape
    _, k = Beta.shape

    tol = 0.1
    max_iter = 10000 

    tau = 0.1
    sigma = 1/(12*tau)

    x_t = np.zeros((l,m,n), D.dtype)
    x_a = np.zeros((k,1), D.dtype)
    y_t = np.zeros((3,l,m,n), D.dtype)
 
    EL,ET,Es = computeEnergy(D, x_t, gamma, x_a, Beta)
    print('Initial Energy: E = ' + str(Es) + ', EL=' + str(EL) + ', ET=' + str(ET))
    change = 10
    
    print_iters = 200
    if verbose == True:
        print_iters = 50 
    for i in range(max_iter):
        ks_yt = -div(y_t)
        xt_new, xa_new = ProxG(x_t - tau*ks_yt, x_a, tau, D, Beta)
        yt_new = ProxFS(y_t + sigma*grad(2*xt_new - x_t), gamma)
       
        x_t = xt_new
        x_a = xa_new
        y_t = yt_new
        
        EL,ET,E = computeEnergy(D, x_t, gamma, x_a, Beta)
        Es = np.append(Es, E)
        length = Es.shape[0]
        El5 = np.mean(Es[np.maximum(0,length-6):length-1])
        El5c = np.mean(Es[np.maximum(0,length-5):length])
        change = np.append(change, El5c - El5)
        if np.mod(i+1, print_iters) == 0:
            print('Iter ' + str(i+1) + ': E = ' + str(E) + '; EL=' + str(EL) + ', ET=' + str(ET) + ', aechg = ' + str(change[length-1]))
        if i >= 100 and np.max(np.abs(change[np.maximum(0, length-3):length])) < tol:
            print('Iter ' + str(i+1) + ': E = ' + str(E) + '; EL=' + str(EL) + ', ET=' + str(ET) + ', aechg = ' + str(change[length-1]))
            print('Converged after ' + str(i+1) + ' iterations.')
            break
    T = x_t
    Alpha = x_a
    L = D-T
    return (L, T, Alpha)

