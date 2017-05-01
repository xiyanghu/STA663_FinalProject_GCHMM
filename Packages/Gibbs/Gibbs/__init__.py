import numpy as np
from scipy import stats
import scipy.io
from matplotlib import pyplot as plt
import time
import pstats

def Gibbs_GCHMM(G, Ytrue, mask, T, 
                prior={'ap':2, 'bp':5, 'aa':2, 'ba':5, 'ab':2, 'bb':5, 'ar':2, 'br':5, 'a1':5, 'b1':2, 'a0':2, 'b0':5}):

    N, _, D = G.shape
    _, S, _ = Ytrue.shape

    X = np.zeros((N,D+1))
    R = np.zeros((N,D))

    ap = prior['ap']; bp = prior['bp']
    aa = prior['aa']; ba = prior['ba']
    ab = prior['ab']; bb = prior['bb']
    ar = prior['ar']; br = prior['br']
    a1 = prior['a1']; b1 = prior['b1']
    a0 = prior['a0']; b0 = prior['b0']
    
    xi = stats.beta.rvs(ap, bp, size=1)
    alpha = stats.beta.rvs(aa,ba,size=1)
    beta = stats.beta.rvs(ab,bb,size=1)
    gamma = stats.beta.rvs(ar,br,size=1)
    theta1 = stats.beta.rvs(a1,b1,size=(1,S)) 
    theta0 = stats.beta.rvs(a0,b0,size=(1,S)) 
    
    B = T/2 # Burn-in from Iteration B
    Xbi = X # Burn-in for X
    Ybi = np.zeros((Ytrue.shape))
    parabi = np.zeros((1,2*S+4)) # Burn-in for all parameters
    NPI = np.zeros((N,D)) # Num. of previous infection

    for t in range(T):
        th1 = np.repeat(np.repeat(theta1.reshape((1,S,1)), N, axis=0), D, axis=2)
        th0 = np.repeat(np.repeat(theta0.reshape((1,S,1)), N, axis=0), D, axis=2)
        Ym = mask * (stats.bernoulli.rvs(th1, size=Ytrue.shape) * (X[:,1:] == 1).reshape(N, 1 ,D) + 
                stats.bernoulli.rvs(th0, size=Ytrue.shape) * (X[:,1:] == 0).reshape(N, 1, D))
        Y = Ym + (1 - mask) * Ytrue

        # Update the initial X, root
        NPI[:,0] = NumPreInf(X[:,0],G[:,:,0]);
        p1 = xi*(gamma**np.array(X[:,1]==0)*(1-gamma)**np.array(X[:,1]))
        p0 = (1-xi)*(1-(1-alpha)*(1-beta)**NPI[:,0])**X[:,1]*((1-alpha)*(1-beta)**NPI[:,0])**(X[:,1]==0)
        p = p1 / (p0+p1)
        X[:,0] = 0+(np.random.rand(N,)<=p)
    
        # Update intermediate X
        for i in range(1,D):
            NPI[:,i-1] = NumPreInf(X[:,i-1],G[:,:,i-1])
            NPI[:,i] = NumPreInf(X[:,i],G[:,:,i])
            tmp1 = np.exp(Y[:,:,i-1] @ np.log(theta1.T))*np.exp((1-Y[:,:,i-1]) @ np.log(1-theta1.T))
            p1 = gamma**(X[:,i+1]==0)*(1-gamma)**(X[:,i-1]+X[:,i+1])*(1-(1-alpha)*(1-beta)**NPI[:,i-1])**(X[:,i-1]==0) * tmp1.reshape((N,))
            tmp0 = np.exp(Y[:,:,i-1] @ np.log(theta0.T))*np.exp((1-Y[:,:,i-1]) @ np.log(1-theta0.T))
            p0 = gamma**X[:,i-1]*(1-(1-alpha)*(1-beta)**NPI[:,i])**X[:,i+1]*(1-alpha)**((X[:,i-1]==0)+(X[:,i+1]==0))*(1-beta)**(NPI[:,i-1]*(X[:,i-1]==0)+NPI[:,i]*(X[:,i+1]==0))*tmp0.reshape((N,))
            p = p1 / (p0+p1)
            X[:,i] = 0+(np.random.rand(N,)<=p)
        
        # Updata last X
        NPI[:,D-1] = NumPreInf(X[:,D],G[:,:,D-1])
        tmp1 = np.exp(Y[:,:,D-1] @ np.log(theta1.T))* np.exp((1-Y[:,:,D-1]) @ np.log(1-theta1.T))
        p1 = (1-gamma)**X[:,D-1]*(1-(1-alpha)*(1-beta)**NPI[:,D-1])**(X[:,D-1]==0)*tmp1.reshape((N,))
        tmp0 = np.exp(Y[:,:,D-1] @ np.log(theta0.T))*np.exp((1-Y[:,:,D-1]) @ np.log(1-theta0.T))
        p0 = gamma**X[:,D-1]*((1-alpha)*(1-beta)**NPI[:,D-1])**(X[:,D-1]==0)*tmp0.reshape((N,))
        p = p1/(p0+p1)
        X[:,D] = 0+(np.random.rand(N,)<=p)
        
        p = alpha / (alpha+beta * NPI)
        tmp = 2 - (np.random.rand(N,D) <= p)
        R = (X[:,0:D]==0)*X[:,1:]*tmp
        
        # Update parameters
        xi = stats.beta.rvs(ap+sum(X[:,0]),bp+N-sum(X[:,0]),size=1)
        gamma = stats.beta.rvs(ar+np.sum(X[:,0:D]*(X[:,1:]==0)),br+np.sum(X[:,0:D]*X[:,1:]))
        alpha = stats.beta.rvs(aa+np.sum(R==1),ba+ np.sum((X[:,0:D]==0)*(X[:,1:]==0))+np.sum(R==2))
        beta = stats.beta.rvs(ab+np.sum(R>1),bb+np.sum(NPI*((X[:,0:D]==0)^(R>1))))
    
        temp = np.transpose(np.repeat(np.expand_dims(X[:,1:], axis=2),S, axis = 2), axes = [0, 2, 1])
        theta1 = stats.beta.rvs(a1 + np.sum(Y*temp, axis=(0,2)), b1 + np.sum((1-Y)*temp, axis=(0,2)), size = S).reshape((1,S))
        theta0 = stats.beta.rvs(a0 + np.sum(Y*(temp==0), axis=(0,2)), b0 + np.sum((1-Y)*(temp==0), axis=(0,2)), size = S).reshape((1,S))
        
        # Burn-in
        if t>B:
            Xbi = Xbi + X
            Ybi = Ybi + Ym
            parabi = parabi + np.c_[xi,alpha,beta,gamma,theta1,theta0]
    # prediction
    Xpred = Xbi/(T-B)
    Ympred = Ybi/(T-B)
    parapred = parabi/(T-B)
    return [Xpred, Ympred, parapred]

def NumPreInf(Xt, Gt):
    return ((Gt + Gt.T) > 0) @ Xt


