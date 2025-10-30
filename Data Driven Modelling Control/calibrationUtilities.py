import numpy as np
import scipy as sp 
import utilities as ut
import torch
import matplotlib.pyplot as plt
#import tensorflow as tf

def d(x): 
    return 0.5*( -1-0.1*x**2)


def f(t, x, u):

    return d(x)*(x-u)


def g(x, c=0.1, bias=lambda x: 0*x): 
    return 2*c*x + 16*np.exp(x)/(1+np.exp(x)) + 8 + 2*4*2*8*(c - 0.05)**3*x + bias(x)

def pairwise_sq_dist(A, B):
    """
    Computes pairwise squared distances between rows of A and B.
    A, B: shape (m, d)
    Returns: matrix of shape (m, m), where result[i, j] = ||A[i] - B[j]||^2
    """
    A_sq = np.sum(A**2, axis=1).reshape(-1, 1)  # shape (m, 1)
    B_sq = np.sum(B**2, axis=1).reshape(1, -1)  # shape (1, m)
    inner_prod = A @ B.T                        # shape (m, m)
    return A_sq - 2 * inner_prod + B_sq


def calc_xBx(A, Xm):
    nm = Xm.shape[0]
    xBx = np.zeros((nm, nm))

    for i in range(nm-1): 

        elem = np.subtract(Xm[i+1:, :], Xm[i, :]).T

        norm = np.einsum('ij,ij->j', elem, A @ elem )

        xBx[i, i+1:] = norm

    return xBx + xBx.T


def calc_x1Bx2(A, X1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    xBx = np.zeros((n1, n2))
    #print(n1, n2)
    for i in range(n1): 

        elem = np.subtract(X1[i, :], X2[:, :]).T
        #print(elem.shape)

        norm = np.einsum('ij,ij->j', elem, A @ elem )

        xBx[i, :] = norm
        

    return xBx

def calc_x1x2(X1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    xBx = np.zeros((n1, n2))
    #print(n1, n2)
    for i in range(n1): 

        elem = np.subtract(X1[i, :], X2[:, :]).T
        #print(elem.shape)

        norm = np.einsum('ij,ij->j', elem, elem )

        xx[i, :] = norm
        

    return xx

def grad_tc(alpha, m):
    return np.fromfunction(lambda i, j: np.maximum(i, j) * alpha**(np.maximum(i, j) - 1), (m, m), dtype=int)

def grad_diagonal_increasing(alpha, m):
    return np.diag([i * alpha**(i - 1) if i > 0 else 0 for i in range(m)])

# Gradient of DI(alphaNL)
def grad_diagonal_geometric(C, alpha):
    m = C.shape[0]
    grad_diag = [i * alpha**(i - 1) if i > 0 else 0 for i in range(m)]
    return np.diag(grad_diag)


def K_linearnormalized(t1,t2, alpha=1): 

    return t1*alpha*t2/(0.5*t1**4*alpha**2 + 0.5*t2**4*alpha**2)**(1/2)


def normalizing_kernel(t1, t2, kernel, s=2):

    return kernel(t1,t2)/(0.5*kernel(t1,t1)**s + 0.5*kernel(t2,t2)**s)**(1/s)


def linear_kernel(t1,t2,alpha=1, c=0):
    return t1*t2*alpha + np.abs(c)


class GPFIR(): 
    def __init__(self, hpams, q, nkernel="DI", lkernel="TC"):
        self.firorder = q
        self.hpams = hpams
        
        if nkernel=="DI": 
            self.Mnonlin = ut.kernelMatrixDI(q, hpams["alphaNL"])
        if nkernel=="TC":
            self.Mnonlin = ut.kernelMatrixTC(q, hpams["alphaNL"])
        if lkernel=="DI": 
            self.Mlin = ut.kernelMatrixDI(q, hpams["alphaL"])
        if lkernel=="TC":
            self.Mlin = ut.kernelMatrixTC(q, hpams["alphaL"])


    def getHyperparameters(self):
        return self.hpams


    def setHyperparameters(self, hpams): 
        self.hpams = hpams
        self.Mlin = ut.kernelMatrixTC(self.firorder, hpams["alphaL"])
        self.Mnonlin = ut.kernelMatrixTC(self.firorder, hpams["alphaNL"])
    

    def setSpecificHpam(self, key, value): 
        self.hpams[key] = value

        if key=="alphaL" or key=="alphaNL": 
            self.Mlin = ut.kernelMatrixTC(self.firorder, self.hpams["alphaL"])
            self.Mnonlin = ut.kernelMatrixDI(self.firorder, self.hpams["alphaNL"])

        
    def updateModelParameter(self, hpams): 
        
        self.setHyperparameters(hpams)
        

    def Kx_steadystate(self, a, iAx, xBx, sumB, iBx): 

        Lterm = self.hpams["rhoL"] * a * iAx
        NLterm = self.hpams["rhoNL"]* np.exp(- (a**2 * sumB + xBx - 2*a*iBx)*self.hpams["lengthNL"])
        return Lterm * NLterm # ! * / +
    

    def getKsteadyState(self, a, xm): 

#        Kxms = np.zeros(nm)
        
        xBx = np.array([xm[i, :].T @ self.Mnonlin @ xm[i, :] for i in range(xm.shape[0])])
        iBx =   self.Mnonlin.sum(axis=0) @ xm.T
        iAx =  self.Mlin.sum(axis=0) @ xm.T
        sumB = self.Mnonlin.sum()

        na = a.shape[0]
        cov = np.zeros(na)
        kss = np.zeros((na, xm.shape[0]))
        for i in range(na): 
            kss[i, :] = self.Kx_steadystate(a[i], iAx, xBx, sumB, iBx)

            xs = np.array([xm.shape[1]*[a[i]]]).T

            cov[i] = self.Kx(xs, xs)

        return kss, cov



    def Kx(self, x1, x2):
        return self.hpams["rhoL"] * x1.T @self.Mlin @x2 * self.hpams["rhoNL"]* np.exp(-(x1 - x2).T @ self.Mnonlin@ (x1 - x2) *self.hpams["lengthNL"])
    # ! * / +

    def Kx_fast(self, x1, x2, xBx):
        return self.hpams["rhoL"] * x1.T @self.Mlin @x2 * self.hpams["rhoNL"]* np.exp(- xBx *self.hpams["lengthNL"])
    # ! * / +

    def Kt(self, t1mt2):
        
        # return normalizing_kernel(t1, t2, lambda x,y: linear_kernel(x,y,alpha=3000, c=1200), s=2)

        return self.hpams["rhoT"] * np.exp(- (t1mt2)**2 * self.hpams["alphaT"])


    def Kd(self, x1, x2):
        
        # Compute pairwise squared differences using broadcasting
        #x1Dx2 = (X_s[:, np.newaxis] - X_s[np.newaxis, :]) ** 2
    
        # Apply the exponential function element-wise
        #Kd = np.exp(-self.hpams["d"] * x1Dx2)
        x1x2 = pairwise_sq_dist(x1, x2)
        #x1x2 = np.subtract.outer(x1.T, x2.T)
        #Kd = np.exp(-self.hpams["d"]*x1x2**2)
        Kd = np.exp(-self.hpams["d"]*x1x2)
        return Kd, x1x2 # bias 


    def Ke(self, x1, x2):
        return self.hpams["sigma2"]*np.eye(x1.shape[0])
    

    def getKxms(self, x1, x2): 
        
        A = self.Mnonlin
        B = self.Mlin # !! Embed this into the GP?

        x1Bx2 = calc_x1Bx2(A, x1.T, x2.T)
        
        Kx1x2 = self.Kx_fast(x1, x2, x1Bx2) 

        return Kx1x2, x1Bx2
    
    def getKs(self, xsvec, thetas): 

        Ks = self.Kx(xsvec.T, xsvec.T)*self.Kt(np.subtract.outer(thetas, thetas))

        return Ks  


    def getKm(self, xm):
        
        xBx = calc_xBx(self.Mnonlin, xm)

        Kx = self.Kx_fast(xm.T, xm.T, xBx) 
        Kerr = self.Ke(xm, xm)
        return Kx, Kerr, xBx

    def getKxss(self, xs):
        
        xBx = calc_xBx(self.Mnonlin, xs)
        Kx_ss = self.Kx_fast(xs.T, xs.T, xBx) 
        return Kx_ss, xBx

    # The gradient computation
    # Functions
    # The Gradient of $\frac{\mathrm{d} K_{x}}{\mathrm{d} \alpha_{L}}$
    def grad_alpl(self, x1, x2):
        C = self.Mnonlin
        B = self.Mlin
        m = B.shape[0]
        x1Cx2 = calc_x1Bx2(C, x1.T, x2.T)  # scalar
        dB_dalpha = grad_tc(self.hpams["alphaL"], m)  # new implementation

        core = x1.T @ dB_dalpha @ x2
        return self.hpams["rhoL"] * core * self.hpams["rhoNL"] * np.exp(-x1Cx2 * self.hpams["lengthNL"])
    
    def grad_alpnl(self, x1, x2):
        C = self.Mnonlin  # TC kernel matrix
        B = self.Mlin # TC kernel matrix
        
        m = C.shape[0]
        x1Cx2 = calc_x1Bx2(C, x1.T, x2.T)  # scalar

        # Gradient of TC kernel matrix w.r.t. alphaNL
        dC_dalpha = grad_tc(self.hpams["alphaNL"], m)

        core = calc_x1Bx2(dC_dalpha, x1.T, x2.T)
        Kx = self.hpams["rhoL"] * (x1.T @ B @ x2) * np.exp(-x1Cx2 * self.hpams["lengthNL"])
        return -self.hpams["lengthNL"] * core * Kx
        
    # The Gradient of $\frac{\mathrm{d} K_{x}}{\mathrm{d} \rho _{L}}$
    def grad_rhol(self, x1, x2):
        A = self.Mnonlin
        x1Bx2 = calc_x1Bx2(A, x1.T, x2.T)
        return x1.T @self.Mlin @x2 * self.hpams["rhoNL"]* np.exp(-x1Bx2 *self.hpams["lengthNL"])

    # The Gradient of $\frac{\mathrm{d} K_{x}}{\mathrm{d} \length _{NL}}$
    def grad_lnl(self, x1, x2):
        A = self.Mnonlin
        x1Bx2 = calc_x1Bx2(A, x1.T, x2.T)
        return self.hpams["rhoL"] * x1.T @self.Mlin @x2 * self.hpams["rhoNL"]* x1Bx2 * np.exp(-x1Bx2 *self.hpams["lengthNL"])

    # The Gradient of $\frac{\mathrm{d} K_{t}}{\mathrm{d} \alpha _{t}}$
    def grad_alpt(self, t1, t2):
        t1mt2 = np.subtract.outer(t1, t2)
        return -self.hpams["rhoT"] * (t1mt2)**2 * np.exp(- (t1mt2)**2 * self.hpams["alphaT"])

    # The Gradient of $\frac{\mathrm{d} K_{b}}{\mathrm{d} d}$
    def grad_kbd(self, x1, x2):
        x1x2 = pairwise_sq_dist(x1, x2)
        return -x1x2*np.exp(-self.hpams["d"]* x1x2)

def forwardEuler(ut, f, x0, t0, tend, n): 
    xt = np.zeros(n)
    dt = (tend - t0)/n

    xt[0] = x0
    t = t0
    for i in range(n-1):
               
        xt[i+1] = xt[i] + dt*f(t, xt[i], ut[i])

        t = t0+dt 
    return xt


def createInputMatrix(um, m, N): 
    # creates matrix Xm from the time series um
    Xt = np.zeros((N, m))
    for i in range(N): 
        Xt[i, :] = np.flip(um[i:i+m]) # um[m+i-1:i-1:-1]
    return Xt


def simulateSystem(u, t0, tend, N, x0=1): 

    
    xtot = forwardEuler(u, f, x0, t0, tend, N)
    ytot = g(xtot) 

    return ytot


def rbf_kernel(x1, x2, variance = 1):
    return np.exp(-1 * ((x1-x2) ** 2) / (2*variance))


def gram_matrix(xs, variance=1):
    return [[rbf_kernel(x1,x2, variance) for x2 in xs] for x1 in xs]


def sampleGPinput(t0, tend, dt, var=1):
    np.random.seed(42)
    xs = np.arange(t0, tend, dt)
    mean = [0 for x in xs]
    gram = gram_matrix(xs, variance=var)

    plt_vals = []
    for i in range(0, 1):
        ys = 5*np.random.multivariate_normal(mean, gram)
        plt_vals.extend([xs, ys, "k"])
    return ys


#def kernelMatrixTC(m, alpha):
#    # construct matrix for TC kernel (1st order smoothness and stable)
#    P = np.zeros((m,m))
#    for i in range(m, 0, -1): 
#        P[:i, :i] = alpha**(i)
#    return P

def kernelMatrixTC(m, alpha):
    return np.fromfunction(lambda i, j: alpha**(np.maximum(i, j)), (m, m), dtype=int)

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

def filterTCmatrix(alpha, n):
    F = np.zeros((n,n))
    for i in range(n): 
        if i<n-1: 
            F[i,i] = np.sqrt(1/(alpha**(i+1)*(1-alpha)))

            F[i,i+1] = -np.sqrt(1/(alpha**(i+1)*(1-alpha)))
            #F[i,i+1] = -np.sqrt(1/(alpha**(i+1)*(1-alpha)))


        elif i==n-1: 
            # print(i)
            F[i,i] = np.sqrt(1/(alpha**(i+1)))   # !!! 


    return F


def filterTC_multi(alphas, n):
    nu = np.shape(alphas)[0]

    Fm = np.zeros((nu*n, nu*n))

    for i in range(nu):
        
        Fm[i*n:(i+1)*n, i*n:(i+1)*n] = filterTCmatrix(alphas[i], n)

    return Fm


def invFilterTC_multi(alphas, n): 

    nu = np.shape(alphas)[0]

    Finv = np.zeros((nu*n, nu*n))

    for i in range(nu):
        Finv[i*n:(i+1)*n, i*n:(i+1)*n] = np.linalg.inv(filterTCmatrix(alphas[i], n), )

    return Finv



def getR1R2rDsim(eta, firorder, Rd1, Rd2): 
    
    alpha = eta[0] # TC parameter
    lmbda = eta[1] # Kernel coefficient
    sigma2 = eta[2] # White noise variance

    gamma = sigma2/lmbda 

    ninput = int(Rd1.shape[1]/ firorder)

    F = torch.tensor(filterTC_multi(ninput*[alpha],  firorder), device='cpu')
    
    line1 = torch.hstack((Rd1, Rd2))
    line2 = torch.hstack((np.sqrt(gamma)*F, torch.zeros((ninput*firorder, Rd2.shape[1]))))
    QRMATRIX =     torch.vstack((line1, line2))

    nump = Rd1.shape[1]
    numy = Rd2.shape[1]
    R = torch.linalg.qr(QRMATRIX, mode='r')[1]
    
    r = R[-1, -1]
    R2 = R[:nump, -numy:]
    R1 = R[:nump, :nump]

    return (R1, R2, r, F)


def mml_QR_fast(eta, firorder, Rd1, Rd2): 
    
    (R1, R2, r, F) = getR1R2rDsim(eta, firorder, Rd1, Rd2)
    sigma2 = eta[2]
    res = r**2 / sigma2 + (-firorder)*np.log(sigma2) + torch.linalg.slogdet(R1)[1]**2
    return res


def train_QR_fast(X, y, eta, firorder):

    Xy = torch.tensor(np.hstack((X, y)), device='cpu')
    Q, R = torch.linalg.qr(Xy, mode='reduced')

    nx = X.shape[1]
    Rd1 = R[:, :nx]
    Rd2 = R[:, nx:]

    (R1, R2, r, F) = getR1R2rDsim(eta, firorder, Rd1, Rd2)
    
    return torch.linalg.solve_triangular(R1, R2, upper=True)


def getCov(K_cov, K_rep, cholK): 
    
    cholKinvK_rep = torch.linalg.solve_triangular(cholK, K_rep.T, upper=False)

    return K_cov - cholKinvK_rep.numpy().T @ cholKinvK_rep.numpy()


def comp_chol_K(Kt, cholA, Kms, KssKt): 
    
    KmsKt = Kms * Kt

    cholAinvB = torch.linalg.tsolve_triangular(cholA, KmsKt)

    DD = cholAinvB.numpy().T  @ cholAinvB.numpy()
    schurElement = KssKt - DD

    cholSchur = np.linalg.cholesky(schurElement)

    cholK = np.vstack((np.hstack((cholA, np.zeros_like(cholAinvB.numpy()))), np.hstack((cholAinvB.numpy().T, cholSchur))))

    return cholK


def theta_log_likelihood(gpmodel, ts, y, cholA, Kms, KssKt, interval): 

    k = y.shape[0]
    khalflog2pi = k/2*1.8378770664093453
    theta_list = interval
    L = []
    for i in range(theta_list.shape[0]):

        t1mt2 = np.subtract.outer(theta_list[i], ts)
        Kt = gpmodel.Kt(t1mt2)

        cholK = comp_chol_K(Kt, cholA, Kms, KssKt)

        qf = torch.linalg.solve_triangular(cholK, y, upper=False) # half of the quadratic form
        
        logdetK = np.sum(np.log(np.diag(cholK))) # 0.5*log det cholK **2 = log det cholK 

        elem = -0.5*qf.numpy().T @ qf.numpy() - logdetK - khalflog2pi
        
        L.append(elem[0,0])

    return np.array(L), theta_list


def calibrate_df_model_kt(gpmodel, thetas, xss_matrix, yss, X_train, y_train, cholA, interval): 
    # Use when changes has been made to Ds, and not Dm. 
    
    t1mt2 = np.subtract.outer(thetas, thetas)
    Kt = gpmodel.Kt(t1mt2)
    Kss, Keee = gpmodel.getKm(xss_matrix) 
    
    Kms = gpmodel.getKxms(X_train.T, xss_matrix.T)


    L, tl = theta_log_likelihood(gpmodel, thetas, np.hstack((y_train, yss)).reshape(y_train.shape[0] + yss.shape[0], 1), cholA, Kms, Kss*Kt, interval)

    return L, tl 



def plot_likelihood(L, tl, gpmodel, trueVal = 0):

    hpams = gpmodel.getHyperparameters() 
    alphaT = hpams["alphaT"]

    theta = tl[np.argmax(L)]
    print(theta, " is the ML estimate, using alphaT={}".format(alphaT))
    
    plt.figure()
    plt.title("Log likelihood of theta, alphaT={}".format(alphaT))
    plt.plot(tl, (L), '*-')
    plt.vlines(trueVal, (L).min(), (L).max(), 'r', label="True parameter")
    plt.grid()
    plt.legend()

    return theta

def prepDs(thetass_list, xss_list, X_train, xSYS, wsys, q, gpmodel): 

    yss, xss_matrix = sample_simulator(xss_list, thetass_list, xSYS, wsys, q, gpmodel)
    Kt = gpmodel.Kt(np.subtract.outer(thetass_list, thetass_list))

    Kss, _, _ = gpmodel.getKm(xss_matrix)  
    Kms = gpmodel.getKxms(X_train.T, xss_matrix.T) # Kx( Xm, Xs)
    
    return yss, xss_matrix, Kt*Kss, Kms


def sys_ss_output(u, xSYS, wsys, gpmodel): 

    ss_rep, cov = gpmodel.getKsteadyState(u, xSYS)   # returns the representers and the covariance of the steady states in alist 

    return ss_rep @ wsys

def sample_simulator(xss, tss, xSYS, wsys, firorder, gpmodel): 

    yss = sys_ss_output(xss, xSYS, wsys, gpmodel) + tss

    xss_matrix = np.repeat(xss.reshape(xss.shape[0], 1), firorder, axis=1)

    return yss, xss_matrix

