import numpy as np
from numpy import linalg as la
import scipy as sp
import calibrationUtilities as cut

def sure(y,k,e):
    # SURE estimator
    jitter = 1e-2
    n = k.shape[0]
    #h = k @ la.inv(k+e+(jitter*np.eye(n)))
    h = np.linalg.solve(k + e + jitter*np.eye(k.shape[0]), k).T
    r = y - (h @ y)
    #L_1 = r.T @ la.inv(e) @ r
    #L_2 = 2*np.trace(la.inv(e) @ h)
    L_1 = r.T @ np.linalg.solve(e, r)
    L_2 = 2 * np.trace(np.linalg.solve(e, h))
    L = L_1+L_2-n
    return L.item()
    
# The Gradient of SURE estimator s
# with respect to H
def grad_sh(y,e,h):
    r = y - (h @ y)
    #Einv = la.inv(e)
    #grad = -2 * (Einv @ (r @ y.T)) + 2 * Einv
    grad = -2 * np.linalg.solve(e, r @ y.T) + 2 * np.linalg.solve(e, np.eye(e.shape[0]))
    #grad = ((-2 *la.inv(e) @ r @ np.transpose(y)) + (2* la.inv(e)))
    #print(Einv.shape, r.shape, y.shape)
    #print((r @ y.T).shape)
    return grad

# The gradient of H with
# respect to K
def grad_hk(h,k,e):
    #m = la.inv(k+e)
    m = np.linalg.solve(k + e, np.eye(k.shape[0]))
    # Compute dH/dK
    grad = m - (h @ m)
    return grad

# The gradient of H with
# respect to E
def grad_he(h,k,e):
    #m = la.inv(k+e)
    m = np.linalg.solve(k + e, np.eye(k.shape[0]))
    # Compute dH/dE
    grad = - (h @ m)
    return grad


# Adaptive Gradient Descent Algorithms
def f_grad_eta(hpams, pam):
    
    # Initialization
    xm = pam["xm"]
    xs = pam["xs"]
    y = pam["y"]
    q = pam["q"]
    thetas = pam["thetas"]
    theta = pam["theta"]
    jitter = 1e-2
    

    # Hyperparameter list
    gpmodel = cut.GPFIR(hpams, q, nkernel="TC", lkernel="TC")

    # Computing Gradient for theta[0]
    
    # Scaling K matrices
    Kx, Ke, _ = gpmodel.getKm(xm.T)
    Kms, _ = gpmodel.getKxms(xm, xs)
    Ksm, _ = gpmodel.getKxms(xs, xm) 
    Ks = gpmodel.getKs(xs.T, thetas)
    Kb, _ = gpmodel.Kd(xs.T, xs.T)
    
    t1mt2 = np.subtract.outer(thetas, thetas)
    t1mts2 = np.subtract.outer(theta, thetas)
    ts1mt2 = np.subtract.outer(thetas, theta)
    Kt = gpmodel.Kt(t1mt2)
    Kts = gpmodel.Kt(t1mts2)
    Kst = gpmodel.Kt(ts1mt2)

    K_11 = Kx
    K_12 = Kms * Kts
    K_21 = Ksm * Kst
    K_22 = Ks
    k = np.block([[K_11, K_12], [K_21, K_22]])

    n = k.shape[0]

    E_11 = Ke
    E_12 = np.zeros((Ke.shape[0], Kb.shape[1]))
    E_21 = np.zeros((Kb.shape[0], Ke.shape[1]))
    E_22 = Kb
    e = np.block([[E_11, E_12], [E_21, E_22]])
    
    #h = k @ la.inv(k+e+(jitter*np.eye(n)))
    h = np.linalg.solve(k + e + jitter*np.eye(k.shape[0]), k).T
    
    # Computing the Gradient

    # Computing Gradient for eta[0]
    gt1_11 = gpmodel.grad_alpl(xm, xm)
    gt1_12 = gpmodel.grad_alpl(xm, xs) * Kts
    gt1_21 = gpmodel.grad_alpl(xs, xm) * Kst
    gt1_22 = gpmodel.grad_alpl(xs, xs) * Kt
    grad_alpl = np.block([[gt1_11, gt1_12], [gt1_21, gt1_22]])
    #gr_al = np.trace(np.dot(grad_sh(y,e,h), np.dot(grad_hk(h,k,e), grad_alpl)))
    gr_al = np.trace(grad_sh(y,e,h).T @ grad_hk(h,k,e) @ grad_alpl)

    # Computing Gradient for eta[1]

    # Computing the Gradient
    gt2_11 = gpmodel.grad_alpnl(xm, xm)
    gt2_12 = gpmodel.grad_alpnl(xm, xs) * Kts
    gt2_21 = gpmodel.grad_alpnl(xs, xm) * Kst
    gt2_22 = gpmodel.grad_alpnl(xs, xs) * Kt
    grad_alpnl = np.block([[gt2_11, gt2_12], [gt2_21, gt2_22]])
    #gr_anl = np.trace(np.dot(grad_sh(y,e,h), np.dot(grad_hk(h,k,e), grad_alpnl)))
    gr_anl = np.trace(grad_sh(y,e,h).T @ grad_hk(h,k,e) @ grad_alpnl)

    # Computing Gradient for eta[2]
    
    # Computing the Gradient
    gt3_11 = gpmodel.grad_rhol(xm, xm)
    gt3_12 = gpmodel.grad_rhol(xm, xs) * Kts
    gt3_21 = gpmodel.grad_rhol(xs, xm) * Kst
    gt3_22 = gpmodel.grad_rhol(xs, xs) * Kt
    grad_rhol = np.block([[gt3_11, gt3_12], [gt3_21, gt3_22]])
    #gr_rl = np.trace(np.dot(grad_sh(y,e,h), np.dot(grad_hk(h,k,e), grad_rhol)))
    gr_rl = np.trace(grad_sh(y,e,h).T @ grad_hk(h,k,e) @ grad_rhol)


    # Computing the Gradient for eta[3]
    
    # Computing the Gradient
    gt4_11 = gpmodel.grad_lnl(xm, xm)
    gt4_12 = gpmodel.grad_lnl(xm, xs) * Kts
    gt4_21 = gpmodel.grad_lnl(xs, xm) * Kst
    gt4_22 = gpmodel.grad_lnl(xs, xs) * Kt
    grad_lnl = np.block([[gt4_11, gt4_12], [gt4_21, gt4_22]])
    #gr_lnl = np.trace(np.dot(grad_sh(y,e,h), np.dot(grad_hk(h,k,e), grad_lnl)))
    gr_lnl = np.trace(grad_sh(y,e,h).T @ grad_hk(h,k,e) @ grad_lnl)

    # Update the Gradient of eta
    gr_eta = np.array([gr_al, gr_anl, gr_rl, gr_lnl])
    
    return gr_eta

def f_grad_at(hpams, pam):
    # Initialization
    xm = pam["xm"]
    xs = pam["xs"]
    y = pam["y"]
    q = pam["q"]
    thetas = pam["thetas"]
    theta = pam["theta"]
    jitter = 1e-2
     
    # Scaling K matrices
    gpmodel = cut.GPFIR(hpams, q, nkernel="TC", lkernel="TC")
    Kx, Ke, _ = gpmodel.getKm(xm.T)
    Kms, _ = gpmodel.getKxms(xm, xs)
    Ksm, _ = gpmodel.getKxms(xs, xm) 
    Ks = gpmodel.getKs(xs.T, thetas)
    Kb, _ = gpmodel.Kd(xs.T, xs.T)
    
    t1mt2 = np.subtract.outer(thetas, thetas)
    t1mts2 = np.subtract.outer(theta, thetas)
    ts1mt2 = np.subtract.outer(thetas, theta)
    Kt = gpmodel.Kt(t1mt2)
    Kts = gpmodel.Kt(t1mts2)
    Kst = gpmodel.Kt(ts1mt2)

    K_11 = Kx + Ke
    K_12 = Kms * Kts
    K_21 = Ksm * Kst
    K_22 = Ks + Kb
    k = np.block([[K_11, K_12], [K_21, K_22]])

    n = k.shape[0]

    E_11 = Ke
    E_12 = np.zeros((Ke.shape[0], Kb.shape[1]))
    E_21 = np.zeros((Kb.shape[0], Ke.shape[1]))
    E_22 = Kb
    e = np.block([[E_11, E_12], [E_21, E_22]])
    
    #h = k @ la.inv(k+e+(jitter*np.eye(n)))
    h = np.linalg.solve(k + e + jitter*np.eye(k.shape[0]), k).T
    
    # Computing the Gradient
    gt5_11 = np.zeros((Kx.shape[0], Kx.shape[0]))
    gt5_12 = Kms * gpmodel.grad_alpt(theta, thetas) 
    gt5_21 = Ksm * gpmodel.grad_alpt(thetas, theta) 
    gt5_22 = Ks * gpmodel.grad_alpt(thetas, thetas) / Kt
    grad_alpt = np.block([[gt5_11, gt5_12], [gt5_21, gt5_22]])
    gr_at = np.trace(grad_sh(y,e,h).T @ grad_hk(h,k,e) @ grad_alpt)
    
    return gr_at

def f_grad_nu(hpams, pam):
    
    # Initialization
    xm = pam["xm"]
    xs = pam["xs"]
    y = pam["y"]
    q = pam["q"]
    thetas = pam["thetas"]
    theta = pam["theta"]
    jitter = 1e-2
      
    # Scaling K matrices
    gpmodel = cut.GPFIR(hpams, q, nkernel="TC", lkernel="TC")
    Kx, Ke, _ = gpmodel.getKm(xm.T)
    Kms, _ = gpmodel.getKxms(xm, xs)
    Ksm, _ = gpmodel.getKxms(xs, xm) 
    Ks = gpmodel.getKs(xs.T, thetas)
    Kb, _ = gpmodel.Kd(xs.T, xs.T)
    
    t1mt2 = np.subtract.outer(thetas, thetas)
    t1mts2 = np.subtract.outer(theta, thetas)
    ts1mt2 = np.subtract.outer(thetas, theta)
    Kt = gpmodel.Kt(t1mt2)
    Kts = gpmodel.Kt(t1mts2)
    Kst = gpmodel.Kt(ts1mt2)

    K_11 = Kx + Ke
    K_12 = Kms * Kts
    K_21 = Ksm * Kst
    K_22 = Ks + Kb
    k = np.block([[K_11, K_12], [K_21, K_22]])

    n = k.shape[0]

    E_11 = Ke
    E_12 = np.zeros((Ke.shape[0], Kb.shape[1]))
    E_21 = np.zeros((Kb.shape[0], Ke.shape[1]))
    E_22 = Kb
    e = np.block([[E_11, E_12], [E_21, E_22]])
    
    #h = k @ la.inv(k+e+(jitter*np.eye(n)))
    h = np.linalg.solve(k + e + jitter*np.eye(k.shape[0]), k).T
    
    # Computing the Gradient
    gt6_11 = np.eye(Kx.shape[0])
    gt6_12 = np.zeros((Kms.shape[0], Kms.shape[1]))
    gt6_21 = np.zeros((Ksm.shape[0], Ksm.shape[1]))
    gt6_22 = np.zeros((Ks.shape[0], Ks.shape[1]))
    grad_nu = np.block([[gt6_11, gt6_12], [gt6_21, gt6_22]])
    gr_nu = np.trace(grad_sh(y,e,h).T @ grad_he(h,k,e) @ grad_nu)
    
    return gr_nu

def f_grad_d(hpams, pam):
    # Initialization
    xm = pam["xm"]
    xs = pam["xs"]
    y = pam["y"]
    q = pam["q"]
    thetas = pam["thetas"]
    theta = pam["theta"]
    jitter = 1e-2

    # Scaling K matrices
    gpmodel = cut.GPFIR(hpams, q, nkernel="TC", lkernel="TC")
    Kx, Ke, _ = gpmodel.getKm(xm.T)
    Kms, _ = gpmodel.getKxms(xm, xs)
    Ksm, _ = gpmodel.getKxms(xs, xm) 
    Ks = gpmodel.getKs(xs.T, thetas)
    Kb, _ = gpmodel.Kd(xs.T, xs.T)
    
    t1mt2 = np.subtract.outer(thetas, thetas)
    t1mts2 = np.subtract.outer(theta, thetas)
    ts1mt2 = np.subtract.outer(thetas, theta)
    Kt = gpmodel.Kt(t1mt2)
    Kts = gpmodel.Kt(t1mts2)
    Kst = gpmodel.Kt(ts1mt2)

    K_11 = Kx + Ke
    K_12 = Kms * Kts
    K_21 = Ksm * Kst
    K_22 = Ks + Kb
    k = np.block([[K_11, K_12], [K_21, K_22]])

    n = k.shape[0]

    E_11 = Ke
    E_12 = np.zeros((Ke.shape[0], Kb.shape[1]))
    E_21 = np.zeros((Kb.shape[0], Ke.shape[1]))
    E_22 = Kb
    e = np.block([[E_11, E_12], [E_21, E_22]])
    
    #h = k @ la.inv(k+e+(jitter*np.eye(n)))
    h = np.linalg.solve(k + e + jitter*np.eye(k.shape[0]), k).T
    
    # Computing the Gradient
    gt7_11 = np.zeros((Kx.shape[0], Kx.shape[0]))
    gt7_12 = np.zeros((Kms.shape[0], Kms.shape[1]))
    gt7_21 = np.zeros((Ksm.shape[0], Ksm.shape[1]))
    gt7_22 = gpmodel.grad_kbd(xs.T, xs.T) 
    grad_kbd = np.block([[gt7_11, gt7_12], [gt7_21, gt7_22]])
    gr_d = np.trace(grad_sh(y,e,h).T @ grad_he(h,k,e) @ grad_kbd)
    
    return gr_d

