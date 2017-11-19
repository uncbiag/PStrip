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

def computeEnergy(D, S, T, _Lambda, _gamma_c, Alpha, Beta):
    
    l, m, n = D.shape
    sum_alpha_beta = np.dot(Beta, Alpha);
  
    # \nabla TV term energy  
    GR = np.linalg.norm(grad(T), axis =0)
    ET = _gamma_c * GR.sum()
    # S term energy
    SP = _Lambda * np.abs(S)
    ES = SP.sum()
    # L-Ba
    sparse = D.reshape(l*m*n, 1) - S.reshape(l*m*n, 1) - T.reshape(l*m*n,1) - sum_alpha_beta
    EL = np.square(sparse).sum()

    E = EL/2 + ET + ES

    return EL,ES,ET,E 

def ProxG(s, t, a, tau, D, Beta):
    l,m,n = D.shape
    lmn = l*m*n
    dl = (D-s-t).reshape(lmn,1)
    delta = np.dot(np.transpose(Beta), dl)
    p_v1 = -tau*tau/(1+3*tau)/(1+2*tau) * np.dot(Beta, delta)
    p_v2 = -tau/(1+3*tau)*np.dot(Beta, a)
    p_v3 = tau/(1+2*tau)*dl
    p = (p_v1 + p_v2 + p_v3).reshape(l,m,n)
    ps = p + s
    pt = p + t
    pa = tau/(1+3*tau)*delta + (1+2*tau)/(1+3*tau)*a
  
    return ps, pt, pa

def ProxFS(s, t, _Lambda, _gamma_c):
    l,m,n = s.shape
    max_abs_s = np.maximum(1, np.abs(s)/_Lambda)
    max_abs_t = np.maximum(1, np.linalg.norm(t, axis=0)/_gamma_c)
    ps = np.divide(s, max_abs_s)
    pt = np.zeros((3,l,m,n), t.dtype)
    pt[0,:,:,:] = np.divide(t[0,:,:,:], max_abs_t)
    pt[1,:,:,:] = np.divide(t[1,:,:,:], max_abs_t) 
    pt[2,:,:,:] = np.divide(t[2,:,:,:], max_abs_t)

    return (ps, pt)

def decompose(D, Beta, _Lambda, _Gamma, _lambda_c, _gamma_c, verbose):
    print 'start decomposing in CPU'
    l,m, n = D.shape
    _, k = Beta.shape

    tol = 0.1
    max_iter = 10000 

    tau = 0.1
    sigma = 1/(13*tau)

    x_s = np.zeros((l,m,n), D.dtype)
    x_t = np.zeros((l,m,n), D.dtype)
    x_a = np.zeros((k,1), D.dtype)
    y_s = x_s
    y_t = np.zeros((3,l,m,n), D.dtype)
 
    _Lambda_out = np.where(_Lambda == 10)
    _Lambda_in = np.where(_Lambda == 0.001)
    _Gamma_out = np.where(_Gamma == 10)

   
    x_s[_Lambda_out] = 0
    x_s[_Lambda_in] = D[_Lambda_in]
 
    EL,ES,ET,Es = computeEnergy(D, x_s, x_t, _Lambda, _gamma_c, x_a, Beta) 
    print 'Initial Energy: E = ' + str(Es) + ', EL=' + str(EL) + ', ES=' + str(ES) + ', ET=' + str(ET)
    change = 10
    
    print_iters = 200
    if verbose == True:
        print_iters = 50 
    for i in range(max_iter):
        t_begin = time.clock()
        ks_yt = -div(y_t)
        ks_ys = y_s
        xs_new, xt_new, xa_new = ProxG(x_s - tau*ks_ys, x_t - tau*ks_yt, x_a, tau, D, Beta)

        xs_new[_Lambda_out] = 0
        xs_new[_Lambda_in] = D[_Lambda_in]
        xt_new[_Gamma_out] = 0

        ys_new, yt_new = ProxFS(y_s+sigma*(2*xs_new - x_s), y_t + sigma*grad(2*xt_new - x_t), _Lambda, _Gamma)
       
        x_s = xs_new
        x_t = xt_new
        x_a = xa_new
        y_s = ys_new
        y_t = yt_new
        
        EL,ES,ET,E = computeEnergy(D, x_s, x_t, _Lambda, _gamma_c, x_a, Beta)
        Es = np.append(Es, E)
        length = Es.shape[0]
        El5 = np.mean(Es[np.maximum(0,length-6):length-1])
        El5c = np.mean(Es[np.maximum(0,length-5):length])
        change = np.append(change, El5c - El5);
        if np.mod(i+1, print_iters) == 0:
            print 'Iter ' + str(i+1) + ': E = ' + str(E) + '; EL=' + str(EL) + ', ES=' + str(ES) + ', ET=' + str(ET) + ', aechg = ' + str(change[length-1]) 
        if i >= 100 and np.max(np.abs(change[np.maximum(0, length-3):length])) < tol:
            print 'Iter ' + str(i+1) + ': E = ' + str(E) + '; EL=' + str(EL) + ', ES=' + str(ES) + ', ET=' + str(ET) + ', aechg = ' + str(change[length-1]) 
            print 'Converged after ' + str(i+1) + ' iterations.'
            break
    S = x_s
    T = x_t
    Alpha = x_a
    L = D-S-T
    return (L, S, T, Alpha)

