import sys
import numpy as np

def CalcDeltas(S, I, D, beta, gamma, dRate):
    betaSI = beta * S[:, None] * I[:, None]
    outS = np.sum(betaSI, axis=1)
    inI  = np.sum(betaSI, axis=0)
    newD = dRate * gamma * I
    newS = (1.0 - dRate) * gamma * I
    dS   = -outS + newS
    dI   = inI - newD - newS
    dD   = newD
    dOut = inI

    return [dS, dI, dD, dOut]

def RunSpatialSIDS(T, startS, startI, startD, beta, gamma, dRate):
    N = len(startS)
    assert(len(startI) == N)
    assert(len(startD) == N)
    assert(beta.shape == (N, N))

    S   = np.empty((N, T))
    I   = np.empty((N, T))
    D   = np.empty((N, T))
    out = np.zeros((N, T))
    S[:, 0] = startS
    I[:, 0] = startI
    D[:, 0] = startD

    for t in range(1, T):
        [dS, dI, dD, dOut] = \
            CalcDeltas(S[:, t-1], I[:, t-1], D[:, t-1], beta, gamma, dRate)
        S[:, t] = S[:, t-1] + dS
        I[:, t] = I[:, t-1] + dI
        D[:, t] = D[:, t-1] + dD
        out[:, t] = out[:, t-1] + dOut
    
    return [S, I, D, out]