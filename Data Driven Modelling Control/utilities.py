import numpy as np 

import matplotlib.pyplot as plt 

import scipy as sp

# import kernels as ker

inds_cache = {}

def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype=bool)
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]


def fast_positive_definite_inverse(m):
    # fast way of computing the inverse of m if it is positive definite
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv 

def getRd1Rd2(X, y, m, nu): 

    Xy = np.hstack([X, np.array([y]).T])

    Rd = sp.linalg.qr(Xy, mode='r')[0]

    Rd1 = Rd[:, :nu*m]
    Rd2 = np.array([Rd[:, nu*m]]).T 
    return (Rd1, Rd2)


def vinv_mTC(alphaList, m, nu): 
    # To speed up MML evaluation, use the analytical inverse 
    R = np.zeros((m*nu, m*nu)) 

    for k in range(nu):
        Rk = np.zeros((m, m))
        alpha = alphaList[k]
        sqrta = np.sqrt(alpha)
        for i in range(m-1): 
            Rk[i,i] = (1+alpha) / (alpha**(i)*(1-alpha))
            Rk[i, i+1] = -sqrta/(alpha**(i+1/2)*(1-alpha))
            Rk[i+1, i] = -sqrta/(alpha**(i+1/2)*(1-alpha))
        Rk[0,0] = 1/(alpha*(1-alpha))
        Rk[m-1, m-1] = 1/(alpha**(m)*(1-alpha))
        R[k*m:(k+1)*m, k*m:(k+1)*m] = Rk
    return R

def getR1R2r(firorder, Rd1, Rd2, D, sigma2):


    QRMATRIX = np.block([[Rd1, Rd2], [np.sqrt(sigma2)*D, np.array([np.zeros(firorder)]).T]])
    
    R = sp.linalg.qr(QRMATRIX, mode='r')[0]
    R = R[:firorder+1, :firorder+1]
    r = R[-1, -1]
    R2 = R[:firorder, firorder]
    R1 = R[:firorder, :firorder]

    return (R1, R2, r)

def fastInvQuad(y, k): 
    # fast computation of quadratic form of inverse matrix 
    # returns y.T @ inv(k) @ y
    # k: square matrix of same size as y
    
    L  = sp.linalg.cholesky(k)  # upper triangular
    R, info  = sp.linalg.lapack.dtrtri(L) # inverse of upper triangular 
    
    return np.sum(np.square(R.T @ y))


def kernelMatrixTC_multi(alphaList, nu, m): 
    """ 
    m: lag length
    nu: number of inputs
    alphas: the different alphas, length nu, for each TC matrix
    """

    M = np.zeros((m*nu, m*nu))

    for i in range(nu):
        Mi = kernelMatrixTC(m, alphaList[i])

        M[m*i : m*(i+1), m*i : m*(i+1)] = Mi

    return M





# def kernelMatrixTC(m, alpha):
#     # construct matrix for TC kernel (1st order smoothness and stable)
#     P = np.zeros((m,m))
#     for i in range(m, 0, -1): 
#         P[:i, :i] = alpha**(i-1)
#     return P

def kernelMatrixTC(m, alpha):
    return np.fromfunction(lambda i, j: alpha**(np.maximum(i, j)), (m, m), dtype=int)





def block_diagonal_K(m, nu, kernel, alphaList): 

    """ 
    MISO FIR linear block kernel, where all inputs have same kernel, but different parameters
    """

    M = np.zeros((m*nu, m*nu))
    
    for i in range(nu):
        Mi = kernel(m, alphaList[:, i])

        M[m*i : m*(i+1), m*i : m*(i+1)] = Mi


    return M 


def kernelMatrixDI(m, alpha):
    # construct matrix for diagonal kernel (stable)
    P = np.diag([alpha**i for i in range(m)])
    return P



def plotMeanAnd2STDV(x, vlist, label="", c="k"):
    # Plots the mean and 2 standard deviations, from the "predict()" of the GPmodel class
    plt.plot(x, vlist[:,0], label=label, c=c)
    plt.fill_between(x, vlist[:,0] - 2*np.sqrt(vlist[:,1]), vlist[:,0] + 2*np.sqrt(vlist[:,1]), alpha=0.3, facecolor=c)



def steadyStateSignal(u, m): 
    return np.array([m*[u]])


def createInputMatrix(um, m, N): 
    # creates matrix Xm from the time series um
    Xt = np.zeros((N, m))
    for i in range(N): 
        Xt[i, :] = np.flip(um[i:i+m]) # um[m+i-1:i-1:-1]
    return Xt


def createMultiInputMatrix(nu, U, m, N): 
    # N: number of data points for training = number of observations - lag, m
    # m*nu: lag times number of inputs, that is the size of the models input
    Xt = np.zeros((N, m*nu))

    for j in range(nu): 
        Ui = createInputMatrix(U[:, j], m, N)
        Xt[:, j*m: (j+1)*m ] = Ui

    return Xt


def createStaticInputMatrix(us, m): 
    # creates matrix Xs from the steady state values us
    ns = np.size(us)
    Xs = np.zeros((ns, m))

    for i in range(ns): 
        Xs[i, :] = steadyStateSignal(us[i], m)
    return Xs 


def createARInputMatrix(ym, um, m, N):
    # constructs the hankel matrix of inputs and outputs for AR model
    # Equal number of lags for both y and u

    Xm = np.zeros((N, m+m))    
    for i in range(N): 
        Xm[i, :m] = um[m+i:i:-1]
        
        Xm[i, m:] = ym[m+i:i:-1]
    return Xm

def createStaticARInputMatrix(us, ys, m): 
    ns = np.size(us)
    Xs = np.zeros((ns, m+m))

    for i in range(ns): 
        Xs[i, :m] = steadyStateSignal(us[i], m)
        Xs[i, m:] = steadyStateSignal(ys[i], m)
    return Xs 


def generateKxx(X, kernel, ncol): 
    # Generates the square covariance matrix with input values X of the kernel 'kernel' (class)
    kxx = np.zeros((ncol, ncol))

    for i in range(ncol): 
        xi = X[i, :]
        for j in range(i, ncol):
            xj = X[j, :]
            elem = kernel.evaluate(xi, xj)
            kxx[i, j] = elem
            kxx[j, i] = elem
    return kxx


def constructCovarianceMatrix(x1, x2, kernel): 
    # creates the rectangular matrix of covariances between x1 and x2, with the covariance 'kernel' (class)
    # x1,x2: matrices of input signals
    
    n1 = np.shape(x1)[0]
    n2 = np.shape(x2)[0]

    kx1x2 = np.zeros((n1, n2))
    for i in range(n1): 
        xi = x1[i, :]
        for j in range(n2): 
            xj = x2[j, :]
            kx1x2[i,j] = kernel.evaluate(xi, xj)

    return kx1x2

def dof(kernel, hparams, X, nu, m): 

    sigma0 = hparams[0]
    rho0 = hparams[1]

    P = kernel.getKernelMatrix()
    H = X @ np.linalg.solve( P @ X.T @X + sigma0/rho0 * np.eye(nu*m), P @ X.T)
    return np.trace(H)


def loadData(filename, fra, til , ulist, m, stepsAhead=0 ): 

    normaddata = sp.io.loadmat(filename)


    inputs = normaddata["norm2_inputs"][fra:til, ulist]
    outputs = normaddata["norm2_outputs"][fra+stepsAhead:til+stepsAhead, :]

    nu = ulist.shape[0]
    X = createMultiInputMatrix(nu, inputs, m, til-fra-m)
    y = outputs[m:, 0]


    return (X, y, inputs)



def evalLogLikelihood_linear(hparams, X, y, nu, m, N): 

    sigma2 = hparams[0]
    rho = hparams[1]
    alphaList = hparams[2:]

    K = rho*kernelMatrixTC_multi(m, nu, alphaList) # generate the linear matrix
    
    xkx = np.dot(X, np.dot(K, X.T))     # costruct the covariance matrix, model , kernel_x covariance matrix 

    Z = xkx + sigma2*np.eye(N)      # covariance of output y        kernel_ym covariance matrix


    # print(xkx.shape)

    term1 = fastInvQuad(y, Z)
#    term1 = y.T @ np.linalg.solve(Z, y) 

    # term1 = 0 # ut.fastInvQuad(y, Z_eta)            # computes y.T Kinv y
    (sign, slogdetk) = np.linalg.slogdet(Z)   # precice way of computing the log det (k)
    val = term1 + slogdetk

    return val

     



def tuneGPkernel_MML(GP, y, X, nu, m, N): 

    sigma2 = 1
    rho_lin = 1
    alpha_nu = nu*[0.7]
    hparams0 = np.hstack((sigma2, rho_lin, alpha_nu[:]))

    alphaLB = 0.6
    optbounds = ((0.01, None),(0.01, None), (alphaLB, 0.9999), (alphaLB, 0.9999), (alphaLB, 0.9999), 
                 (alphaLB, 0.9999), (alphaLB, 0.9999), (alphaLB, 0.9999), (alphaLB, 0.9999), (alphaLB, 0.9999), (alphaLB, 0.9999), 
                 (alphaLB, 0.9999))
    args = (X, y, nu, m, N)
    
    #val = evalLogLikelihood_linear(hparams0, X, y, nu, m, N)
    #print(val)
    
    optimizationResult = sp.optimize.minimize(evalLogLikelihood_linear, hparams0, args=args, bounds=optbounds, options={'disp': True, 'maxiter': 2000})
    

    return optimizationResult



