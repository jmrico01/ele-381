import sys
import numpy as np

def PrintProgress(count, total, status=""):
    barLen = 60
    filledLen = int(barLen * count / total)
    bar = "=" * filledLen + "-" * (barLen - filledLen)
    percent = round(100.0 * count / float(total), 1)

    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percent, "%", status))
    sys.stdout.flush()

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

def LerpF(a, b, t):
    return a * (1 - t) + b * t

def GetDataValue(t, T, data):
    T_DATA = len(data)
    dataIndFloat = (float(t) / (T - 1)) * (T_DATA - 1)
    dataIndMin = int(np.floor(dataIndFloat))
    dataIndMax = min(int(np.ceil(dataIndFloat)), T_DATA - 1)
    lerpT = dataIndFloat - int(dataIndFloat)
    return LerpF(data[dataIndMin], data[dataIndMax], lerpT)


def BatchSIR(T, modelInd, startS, startI, startR,
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

def RunModel(T, modelInd, startS, startI, startR, params):
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

def ResampleData(src, dst):
    xSrc = np.linspace(0.0, 1.0, len(src))
    xDst = np.linspace(0.0, 1.0, len(dst))
    resampled = np.interp(xDst, xSrc, src)
    assert(len(resampled) == len(dst))
    return resampled

def CalcError(model, data):
    modelResample = ResampleData(model, data)
    #diff = np.divide(modelResample - data, data, where=data!=0)
    diff = modelResample - data
    return np.sqrt(np.sum(diff * diff) / len(diff)) / np.average(data)

def CalcErrorFromParams(T, modelInd, startS, startI, startR, params, data):
    [_, _, _, out] = RunModel(T, modelInd,
        startS, startI, startR, params)
    return CalcError(out, data)
