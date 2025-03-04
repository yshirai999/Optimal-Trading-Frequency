import numpy as np
import scipy as sp

def BSprice(S: list[float], k0: np.ndarray, k1: np.ndarray, r: float, T: float, sigma: float) -> list[np.ndarray]:

    d1 = ( np.log(k0/S[0]) + (r+0.5*sigma[0]*sigma[0]*np.sqrt(T)) ) / sigma[0]*np.sqrt(T)
    d2 = d1 - sigma[0]*np.sqrt(T)
    C0 = sp.stats.norm.cdf(d1)*S[0]-sp.stats.norm.cdf(d2)*k0*np.exp(-r*T)
    P0 = C0 - S[0] + k0*np.exp(-r*T)

    d1 = ( np.log(k1/S[1]) + (r+0.5*sigma[1]*sigma[1]*np.sqrt(T)) ) / sigma[1]*np.sqrt(T)
    d2 = d1 - sigma[1]*np.sqrt(T)
    C1 = sp.stats.norm.cdf(d1)*S[1]-sp.stats.norm.cdf(d2)*k1*np.exp(-r*T) 
    P1 = C1 - S[1] + k1*np.exp(-r*T)
    return [C0, P0, C1, P1]

def BGprice(S: list[float], k0: np.ndarray, k1: np.ndarray, r: float, T: float, \
            alpha: float, lam: float, eta: float, N: int, \
             bp: float, cp: float, bn: float, cn: float) -> list[np.ndarray]:
    beta = np.log(S[0])-lam*N/2
    k = beta+(np.cumsum(np.ones(N,1))-1)*lam
    u = (np.cumsum(np.ones(N,1))-1)*eta
    w = np.ones(N,1)*eta
    w[0] = w[0]/2
    x = np.exp(-1j*beta*u)*Psi_BG(u,bp,cp,bn,cn,T,r,alpha,S[0])*w
    Call = np.real((np.exp(-alpha*k)/np.pi)*sp.fft(x,N))
    kk = np.log(k0)
    C0 = np.interp(k,Call,kk)
    P0 = C0 - S[0] + k0*np.exp(-r*T)

    beta = np.log(S[1])-lam*N/2
    k = beta+(np.cumsum(np.ones(N,1))-1)*lam
    u = (np.cumsum(np.ones(N,1))-1)*eta
    w = np.ones(N,1)*eta
    w[0] = w[0]/2
    x = np.exp(-1j*beta*u)*Psi_BG(u,bp,cp,bn,cn,T,r,alpha,S[0])*w
    Call = np.real((np.exp(-alpha*k)/np.pi)*sp.fft(x,N))
    kk = np.log(k1)
    C1 = np.interp(k,Call,kk)
    P1 = C1 - S[1] + k1*np.exp(-r*T)

    return [C0, P0, C1, P1]

def Phi_BG(u: float, r: float, k: np.ndarray, T: float, bp: float, cp: float, bn: float, cn: float) -> list[float]:
    d0 = 1j*u*(np.log(k)+T*(r+cp*np.log(1-bp)+cn*np.log(1+bn)))
    Phi = np.exp(d0)*((1-1j*u*bp)^(-T*cp))*((1+1j*u*bn)^(-T*cn))
    return Phi

def Psi_BG(u: float, r: float, k: np.ndarray, T: float, alpha: float, bp: float, cp: float, bn: float, cn: float) -> list[float]: 
    Psi = (np.exp(-r*T)/((alpha+1j*u)/(alpha+1j*u+1)))/Phi_BG(u-(alpha+1)*1j,bp,cp,bn,cn,T,k,r)
    return Psi