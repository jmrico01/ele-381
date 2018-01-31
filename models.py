import sys
import numpy as np

def PrintProgress(count, total, status=""):
    barLen = 60
    filledLen = int(barLen * count / total)
    bar = "=" * filledLen + "-" * (barLen - filledLen)
    percent = round(100.0 * count / float(total), 1)

    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percent, "%", status))
    sys.stdout.flush()

def ResampleData(src, dst):
    xSrc = np.linspace(0.0, 1.0, len(src))
    xDst = np.linspace(0.0, 1.0, len(dst))
    resampled = np.interp(xDst, xSrc, src)
    assert(len(resampled) == len(dst))
    return resampled

def LerpF(a, b, t):
    return a * (1 - t) + b * t

def GetDataValue(t, T, data):
    T_DATA = len(data)
    dataIndFloat = (float(t) / (T - 1)) * (T_DATA - 1)
    dataIndMin = int(np.floor(dataIndFloat))
    dataIndMax = min(int(np.ceil(dataIndFloat)), T_DATA - 1)
    lerpT = dataIndFloat - int(dataIndFloat)
    return LerpF(data[dataIndMin], data[dataIndMax], lerpT)

def CalcDeltasSIR(S, I, R, params):
    # Takes in coefficients and the S, I, R of the previous time step
    # Returns deltas: [dS, dI, dR, dOut]
    newInfected = params[0] * S * I
    newRecovered = params[1] * I
    dS = -newInfected
    dI = newInfected - newRecovered
    dR = newRecovered
    # we're only recording the number of new cases of I
    dOut = newInfected

    return [dS, dI, dR, dOut]

def CalcDeltasSIDS(S, I, R, params):
    # Takes in coefficients and the S, I, R of the previous time step
    # Returns deltas: [dS, dI, dR, dOut]
    newInfected = params[0] * S * I
    newSusceptible = (1.0 - params[2]) * params[1] * I
    dead = params[2] * params[1] * I
    dS = -newInfected + newSusceptible
    dI = newInfected - dead - newSusceptible
    dR = dead
    # we're only recording the number of new cases of I
    dOut = newInfected

    return [dS, dI, dR, dOut]

def BatchModelSingle(T, modelInd, startS, startI, startR,
    data, params, progress = True):
    # Run SIR on every combination of vectors beta & gamma passed in.
    # This version doesn't output the full time series, but an error matrix.
    N = len(params[0])
    assert(len(params[1]) == N)
    beta = np.repeat(params[0].reshape(N, 1), N, axis=1)
    gamma = np.repeat(params[1].reshape(1, N), N, axis=0)
    assert(beta.shape == (N, N))
    assert(gamma.shape == (N, N))

    # Keep track of a single SIR time step for all beta*gamma coefficients
    S = np.full((N, N), startS, dtype=np.float64)
    I = np.full((N, N), startI, dtype=np.float64)
    R = np.full((N, N), startR, dtype=np.float64)
    out = np.full((N, N), 0, dtype=np.float64)
    errors = np.full((N, N), 0, dtype=np.float64)
    for t in range(1, T):
        if progress:
            PrintProgress(t, T - 1)

        # This will change depending on the model
        if modelInd == 0:
            [dS, dI, dR, dOut] = CalcDeltasSIR(S, I, R,
                [beta, gamma])
        elif modelInd == 1:
            [dS, dI, dR, dOut] = CalcDeltasSIDS(S, I, R,
                [beta, gamma, params[2]])
        else:
            print("Unsupported model")
            return
        S   += dS
        I   += dI
        R   += dR
        out += dOut

        dataVal = GetDataValue(t, T - 1, data)
        diff = out - dataVal
        errors += (diff * diff) / T
    
    if progress:
        print("")
    return np.sqrt(errors) / np.average(data)

def RunModelSingle(T, modelInd, startS, startI, startR, params):
    # Run a single instance of SIR for the given parameters.
    # Output S, I, R, and the output modeled data
    S = [startS] * T
    I = [startI] * T
    R = [startR] * T
    out = [0] * T
    for t in range(1, T):
        # This will change depending on the model
        dS = 0
        dI = 0
        dR = 0
        dOut = 0
        if modelInd == 0:
            [dS, dI, dR, dOut] = CalcDeltasSIR(S[t-1], I[t-1], R[t-1], params)
        elif modelInd == 1:
            [dS, dI, dR, dOut] = CalcDeltasSIDS(S[t-1], I[t-1], R[t-1], params)
        else:
            print("ERROR: unsupported model")
            return

        S[t]   = S[t-1]   + dS
        I[t]   = I[t-1]   + dI
        R[t]   = R[t-1]   + dR
        out[t] = out[t-1] + dOut
    
    return [S, I, R, out]

def CalcErrorSingle(model, data):
    modelResample = ResampleData(model, data)
    #diff = np.divide(modelResample - data, data, where=data!=0)
    diff = modelResample - data
    return np.sqrt(np.sum(diff * diff) / len(diff)) / np.average(data)

def CalcErrorSingleFromParams(T, modelInd,
startS, startI, startR, params, data):
    [_, _, _, out] = RunModelSingle(T, modelInd,
        startS, startI, startR, params)
    return CalcErrorSingle(out, data)

def CalcDeltasSpatial(S, I, D, beta, gamma, dRate):
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

def RunModelSpatial(T, startS, startI, startD, beta, gamma, dRate):
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
        [dS, dI, dD, dOut] = CalcDeltasSpatial(
            S[:, t-1], I[:, t-1], D[:, t-1], beta, gamma, dRate)
        S[:, t] = S[:, t-1] + dS
        I[:, t] = I[:, t-1] + dI
        D[:, t] = D[:, t-1] + dD
        out[:, t] = out[:, t-1] + dOut
    
    return [S, I, D, out]

def CalcErrorSpatialFromParams(T, startS, startI, startD,
beta, gamma, dRate, bestParamsSingle, data, countries):
    errors = {
        "spatial": [],
        "single": []
    }
    [S, I, R, out] = RunModelSpatial(T, startS, startI, startD,
        beta, gamma, dRate)
        
    N = len(startS)
    for i in range(N):
        errors["spatial"].append(round(CalcErrorSingle(
            out[i], data[countries[i]]), 6))

        [_, _, _, outSingle] = RunModelSingle(T, 1,
            startS[i], startI[i], startD[i],
            bestParamsSingle[countries[i]][1])
        errors["single"].append(round(CalcErrorSingle(
            outSingle, data[countries[i]]), 6))
    
    return errors