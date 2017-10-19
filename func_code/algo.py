# PCA model with ROF version
# E(L, S, T, {\alpha}_l) = \int \gamma \|\nabla T| + \lambda\|S\|+ 1/2*(L-\sum_l\alpha_l\Beta_l\)^2 dx
from __future__ import division

import sys
import time
import numpy as np
import pycuda.autoinit 
import pycuda.driver as driver
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
import skcuda.linalg as sk_linalg
import skcuda.misc as sk_misc
import pycuda.cumath as cumath
import SimpleITK as sitk

from pycuda.elementwise import ElementwiseKernel
square_matrix = ElementwiseKernel(
    "float *x, float *z",
    "z[i] = x[i]*x[i]",
    "square_matrix")
sqrt_matrix = ElementwiseKernel(
    "float *x, float *y",
    "y[i] = sqrt(x[i])",
    "sqrt_matrix")

sum_three_matrix = ElementwiseKernel(
    "float *a, float *b, float *c, float *r, float alpha, float beta, float gamma",
    "r[i] = alpha*a[i] + beta*b[i] + gamma*c[i]",
    "sum_three_matrix")

sum_two_matrix = ElementwiseKernel(
    "float *a, float *b, float *r, float alpha, float beta",
    "r[i] = alpha*a[i] + beta*b[i]",
    "sum_two_matrix")
divide_matrix = ElementwiseKernel(
    "float *a, float *b, float *c",
    "c[i] = a[i]/b[i]",
    "divide_matrix")
multiply_matrix = ElementwiseKernel(
    "float *a, float *b, float *c",
    "c[i] = a[i]*b[i]",
    "multiply_matrix")

absolute_matrix = ElementwiseKernel(
    "float *a, float *b",
    "b[i] = abs(a[i])",
    "absolute_matrix")

assign_matrix = ElementwiseKernel(
    "float *a, float *_lambda, float *b",
    "if(_lambda[i]==10){a[i]=0;} else if(_lambda[i]==0.001f){a[i]=b[i];}",
    "assign_matrix")
kernel_code_template = """
__global__ void MatrixMulKernel(float *A, float *B, float *C)
{

  const uint wA = %(MATRIX_A_COL_SIZE)s;
  const uint wB = 1;

  // Block index
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;

  // Thread index
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  const uint aBegin = wA * %(BLOCK_ROW_SIZE)s * by;
  // Index of the last sub-matrix of A processed by the block
  const uint aEnd = aBegin + wA - 1;
  // Step size used to iterate through the sub-matrices of A
  const uint aStep = %(STEP_SIZE)s;

  // Index of the first sub-matrix of B processed by the block
  const uint bBegin = %(BLOCK_COL_SIZE)s * bx;
  // Step size used to iterate through the sub-matrices of B
  const uint bStep = %(STEP_SIZE)s * wB;

  // The element of the block sub-matrix that is computed
  // by the thread
  float Csub = 0;
  // Loop over all the sub-matrices of A and B required to
  // compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep)
    {
      // Shared memory for the sub-matrix of A
      __shared__ float As[%(BLOCK_ROW_SIZE)s][%(STEP_SIZE)s];
      // Shared memory for the sub-matrix of B
      __shared__ float Bs[%(STEP_SIZE)s][1];

      // Load the matrices from global memory to shared memory
      // each thread loads one element of each matrix
      As[ty][tx] = A[a + wA * ty + tx];
      if (ty == 0)
          Bs[tx][0] = B[b + wB * tx];
      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
      if (tx == 0)
          for (int k = 0; k < %(STEP_SIZE)s; ++k)
              Csub += As[ty][k] * Bs[k][tx];

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

  // Write the block sub-matrix to global memory;
  // each thread writes one element
  if (tx == 0){
    const uint c = wB * %(BLOCK_ROW_SIZE)s * by + %(BLOCK_COL_SIZE)s * bx;
    C[c + wB * ty + tx] = Csub;
  }
}
"""


def grad(u):
    l, m, n = u.shape
    G = gpuarray.zeros((3, l, m, n), u.dtype)
    ux1 = gpuarray.to_gpu(u[:,:,1:])
    ux0 = gpuarray.to_gpu(u[:,:,:-1])
    uy1 = gpuarray.to_gpu(u[:,1:,:])
    uy0 = gpuarray.to_gpu(u[:,:-1,:])
    uz1 = gpuarray.to_gpu(u[1:,:,:])
    uz0 = gpuarray.to_gpu(u[:-1,:,:])
    G[0, :, :, :-1] = ux1 - ux0
    G[1, :, :-1, :] = uy1 - uy0
    G[2, :-1, :, :] = uz1 - uz0
    return G

def div(u):
    _, l, m, n = u.shape
    
    Px = u[0,:,:,:]
    Py = u[1,:,:,:]
    Pz = u[2,:,:,:]

    fx = gpuarray.zeros((l,m,n), u.dtype)
    Px1 = gpuarray.to_gpu(Px[:,:,1:])
    Px0 = gpuarray.to_gpu(Px[:,:,:-1])
    fx[:,:,1:] = Px1 - Px0
    fx[:,:,0] = Px[:,:,0]

    fy = gpuarray.zeros((l,m,n), u.dtype)
    Py1 = gpuarray.to_gpu(Py[:,1:,:])
    Py0 = gpuarray.to_gpu(Py[:,:-1,:])
    fy[:,1:,:] = Py1 - Py0
    fy[:,0,:] = Py[:,0,:]

    fz = gpuarray.zeros((l,m,n), u.dtype)
    Pz1 = gpuarray.to_gpu(Pz[1:,:,:])
    Pz0 = gpuarray.to_gpu(Pz[:-1,:,:])
    fz[1:,:,:] = Pz1 - Pz0
    fz[0, :,:] = Pz[0,:,:]

    del Px1, Px0, Py1, Py0, Pz1, Pz0
    return fx + fy + fz

def computeEnergy(D_v, S,T, _Lambda, _gamma_c, Alpha, Beta):
    l, m, n = S.shape
    
    sum_alpha_beta = gpuarray.zeros_like(D_v)
    sk_linalg.dot(Beta, Alpha, out=sum_alpha_beta);
    
    GR = grad(T)
    square_matrix(GR,GR)
    G_norm = gpuarray.zeros_like(T)
    sum_three_matrix(GR[0,:,:,:], GR[1,:,:,:], GR[2,:,:,:], G_norm, 1.0, 1.0, 1.0)
    sqrt_matrix(G_norm, G_norm)
#    multiply_matrix(G_norm, _Gamma, G_norm)
    ET = _gamma_c*gpuarray.sum(G_norm)

    SP = gpuarray.zeros_like(S)
    absolute_matrix(S,SP)
    multiply_matrix(SP, _Lambda, SP)
    ES = gpuarray.sum(SP)
    
    sparse = D_v - S.reshape(l*m*n, 1) - T.reshape(l*m*n,1) -sum_alpha_beta
    square_matrix(sparse, sparse)
    EL = gpuarray.sum(sparse)

    E = 1/2*EL.get() + ES.get() + ET.get()

    return EL.get(), ES.get(), ET.get(), E


def ProxG(s, t, a, D_v, tau, Beta, BetaT, matrixmul, grid, block):
    l,m,n = s.shape
    lmn = l*m*n
    _, k = Beta.shape
    diff = D_v - s.reshape(lmn,1)-t.reshape(lmn,1)
    delta = gpuarray.zeros((k, 1), dtype=np.float32)
    matrixmul(BetaT, diff, delta, grid=grid, block=block)
    p_v1 = sk_linalg.dot(Beta, delta)
    p_v2 = sk_linalg.dot(Beta, a)
    p_v3 = gpuarray.empty_like(D_v)
    sum_three_matrix(p_v1, p_v2, diff, p_v3, -tau*tau/(1+2*tau)/(1+3*tau), -tau/(1+3*tau), tau/(1+2*tau))

    ps = p_v3.reshape(l,m,n) + s
    pt = p_v3.reshape(l,m,n) + t

    pa = gpuarray.empty_like(a)
    sum_two_matrix(delta, a, pa, tau/(1+3*tau), (1+2*tau)/(1+3*tau))

    return ps, pt, pa




def ProxFSs(s,t, _Lambda, _gamma_c):
    l,m,n = s.shape
    
    t2 = gpuarray.empty_like(t)
    square_matrix(t, t2)
    t_norm = gpuarray.empty((l,m,n), dtype=np.float32)
    sum_three_matrix(t2[0,:,:,:], t2[1,:,:,:], t2[2,:,:,], t_norm, 1.0, 1.0, 1.0)
    sqrt_matrix(t_norm, t_norm)
#    divide_matrix(t_norm, _Gamma, t_norm)
    max_abs_t = gpuarray.maximum(1, t_norm/_gamma_c)
    pt = gpuarray.zeros((3,l,m,n),t.dtype)
    divide_matrix(t[0,:,:,:], max_abs_t, pt[0,:,:,:])
    divide_matrix(t[1,:,:,:], max_abs_t, pt[1,:,:,:])
    divide_matrix(t[2,:,:,:], max_abs_t, pt[2,:,:,:])


    s_abs = gpuarray.empty((l,m,n), dtype=np.float32)   # \|s\|
    absolute_matrix(s, s_abs)
    divide_matrix(s_abs, _Lambda, s_abs)
    max_abs_s = gpuarray.maximum(1, s_abs)
    ps = gpuarray.zeros((l,m,n), s.dtype)
    divide_matrix(s,max_abs_s,ps)

    return ps,pt



def decompose(d, beta, betaT, _lambda, _gamma, _lambda_c, _gamma_c, tol=0.1):
    print 'start decomposing in GPU'
    sk_misc.init()
    D = gpuarray.to_gpu(d)
    Beta = gpuarray.to_gpu(beta)
    BetaT = gpuarray.to_gpu(betaT)
    l, m, n = D.shape
    step_size = np.maximum(m,n);
    lmn, k = Beta.shape
    _Lambda = gpuarray.to_gpu(_lambda)
    _Gamma = gpuarray.to_gpu(_gamma)

    kernel_code = kernel_code_template % {
        'MATRIX_A_COL_SIZE':lmn,
        'BLOCK_ROW_SIZE': 1,
        'BLOCK_COL_SIZE': 1,
        'STEP_SIZE': step_size,
    }         

    mod = compiler.SourceModule(kernel_code)
    matrixmul = mod.get_function("MatrixMulKernel")
    grid=(1,k)
    block=(step_size,1,1)

    D_v = D.reshape(lmn,1)
 
    tol = 0.1
    max_iter = 10000 

    tau = 0.1
    sigma = 1/(13*tau)
    x_s = gpuarray.zeros((l,m,n), D.dtype)
    x_t = gpuarray.zeros((l,m,n), D.dtype)
    x_a = gpuarray.zeros((k,1), D.dtype)
    y_t = gpuarray.zeros((3,l,m,n),D.dtype)
    y_s = x_s


    xs_new = gpuarray.zeros_like(x_s)
    xt_new = gpuarray.zeros_like(x_t)
    xa_new = gpuarray.zeros_like(x_a)
    ys_new = gpuarray.zeros_like(y_s)
    yt_new = gpuarray.zeros_like(y_t)
    assign_matrix(x_s, _Lambda, D)
 
    EL,ES, ET, Es = computeEnergy(D_v, x_s, x_t,  _Lambda, _gamma_c, x_a, Beta) 
    print 'Initial Energy: E = ' + str(Es) + ', EL=' + str(EL) + ', ES=' + str(ES) + ', ET=' + str(ET)
    change = 10;
    t0 = time.time()
    for i in range(max_iter):
        ks_yt = -div(y_t)
        ks_ys = y_s
        
        xs_new, xt_new, xa_new = ProxG(x_s - tau*ks_ys, x_t-tau*ks_yt, x_a, D_v, tau, Beta, BetaT, matrixmul, grid, block)
        assign_matrix(xs_new, _Lambda, D)
        assign_matrix(xt_new, _Gamma, D)

        ys_new,yt_new = ProxFSs(y_s + sigma*(2*xs_new - x_s), y_t + sigma*grad(2*xt_new-x_t), _Lambda, _gamma_c)
        

        x_s = xs_new
        x_t = xt_new
        x_a = xa_new
        y_s = ys_new
        y_t = yt_new

       

        EL,ES,ET,E = computeEnergy(D_v, x_s,x_t,  _Lambda, _gamma_c, x_a, Beta)
        Es = np.append(Es, E)
        length = Es.shape[0]
        El5 = np.mean(Es[np.maximum(0,length-6):length-1])
        El5c = np.mean(Es[np.maximum(0,length-5):length])
        change = np.append(change, El5c-El5);
        t1 = time.time() - t0
        
        if np.mod(i+1, 200) == 0:
            print 'Iter ' + str(i+1) + ': E = ' + str(E) + '; EL=' + str(EL) + ', ES=' + str(ES) + ', ET=' + str(ET) + ', aechg = ' + str(change[length-1])
        if i>= 100 and np.max(np.abs(change[np.maximum(0,length-3):length])) < tol:
            print 'Iter ' + str(i+1) + ': E = ' + str(E) + '; EL=' + str(EL) + ', ES=' + str(ES) + ', ET=' + str(ET) + ', aechg = ' + str(change[length-1])
            print 'Converged after ' + str(i+1) + ' iterations.'
            break
        
    S = x_s
    T = x_t
    Alpha = x_a
    L = D-S-T
    l = L.get()
    s = S.get()
    t = T.get()
    alpha = Alpha.get()
    
    sk_misc.shutdown()
    return (l, s, t, alpha)


