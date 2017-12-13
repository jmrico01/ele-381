import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import read_data

# optimize beta, gamma for both SIR SIS/D, give loss
# graphs:
#   model vs real data

# default:
# heat

# -- Import data
dataFilePath = "ebola_data_db_format.csv"
countries = [
    "Sierra Leone",
    "Guinea",
    "Liberia"
]
categories = [
    read_data.CAT_CONF
]
[data, n, dateStart, dateEnd] = read_data.ReadData(dataFilePath,
    countries, categories, True, 5, False)
#print("----- Data Imported -----")
#print("Data points: " + str(n))
#print("Start date:  " + str(dateStart))
#print("End date:    " + str(dateEnd))
#print("")

startData = {
    "Sierra Leone": [
        7e6,
        200,
        0
    ],
    "Guinea": [
        11.8e6,
        250,
        0
    ],
    "Liberia": [
        4.39e6,
        100,
        0
    ]
}

def PrintProgress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
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
        if dataVal != 0:
            diff /= dataVal
        errors += (diff * diff) / T
    
    if progress:
        print("")
    return np.sqrt(errors)

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
    diff = np.divide(modelResample - data, data, where=data!=0)
    return np.sqrt(np.sum(diff * diff) / len(diff))

def CalcCoefValues(mid, orderOfMag, n):
    exp = 8.0
    values = np.linspace(0.0, 1.0, n)**exp
    minVal = mid * 10.0**orderOfMag
    maxVal = mid / 10.0**orderOfMag
    values = values * (maxVal - minVal) + minVal
    return values

def Present():
    T = 10000
    country = "Guinea"
    modelInd = 1
    bestData = {
        "Sierra Leone": [
            [
                1.5932992983e-09,
                0.0114221593
            ],
            [
                1.59937781919e-09,
                0.01142412728,
                0.71
            ],
        ],
        "Guinea": [
            [
                3.54610225193e-10,
                0.00458655495867,
            ],
            [
                3.97879216751e-10,
                0.0051461993053,
                0.71
            ],
        ],
        "Liberia": [
            [
                1.59937781919e-09,
                0.01142412728
            ],
            [
                1.59937781919e-09,
                0.01142412728,
                0.71
            ],
        ]
    }

    [S, I, R, out] = RunModel(T, modelInd,
        startData[country][0], startData[country][1], startData[country][2],
        bestData[country][modelInd])
    print("Error: " + str(CalcError(out, data[country])))
    out = ResampleData(out, data[country])
    lineModel, = plt.plot(out, label="Model")
    lineData,  = plt.plot(data[country], label="Data")
    plotTitle = country
    if modelInd == 0:
        plotTitle += " - SIR Model"
    elif modelInd == 1:
        plotTitle += " - SIDS Model"
    plt.title(plotTitle)
    plt.xlabel("Time (days since 08/29/2014)")
    plt.ylabel("Cumulative number of Ebola cases")
    plt.legend(handles=[lineModel, lineData])
    plt.show()

def Optimize():
    # BEST SO FAR
    # Beta: 1.59937781919e-09
    # Gamma: 0.01142412728
    # > Error: 0.322179010692

    supervised = True

    print("----- Optimizing SIR Parameters -----")
    country = "Liberia"
    modelInd = 1
    T = 10000
    startS = startData[country][0] # population of Sierra Leone
    startI = startData[country][1] # TODO guess this in a better way
    startR = startData[country][2]

    paramIters = 200
    betaMid = 1e-8
    #betaRangeInitial = 1
    betaRange = 1 # in orders of magnitude

    gammaMid = 0.1
    #gammaRangeInitial = 0.
    gammaRange = 0.1

    deadRate = 0.71

    nextBacktrack = 1
    dRange = 0.5

    minError = 1.0
    minParams = [-1, -1]
    while minError > 0.1: # Probably not feasible
        beta = CalcCoefValues(betaMid, betaRange, paramIters)
        gamma = CalcCoefValues(gammaMid, gammaRange, paramIters)
        print("GREEDY SEARCH ITERATION")
        print("Beta range:  " + str(np.min(beta)) + " - " + str(np.max(beta)))
        print("Gamma range: " + str(np.min(gamma)) + " - " + str(np.max(gamma)))
        params = [beta, gamma]
        if modelInd == 1:
            params.append(deadRate)
        errors = BatchSIR(T, modelInd, startS, startI, startR,
            data[country], params, True)

        [minBetaInd, minGammaInd] = np.unravel_index(np.argmin(errors),
            (paramIters, paramIters))
        iterMinError = errors[minBetaInd, minGammaInd]
        if iterMinError < minError:
            errDiff = minError - iterMinError
            minError = iterMinError
            betaMid = beta[minBetaInd]
            gammaMid = gamma[minGammaInd]
            minParams = [betaMid, gammaMid]
            if (errDiff < 0.0000000000001):
                print("ERROR IMPROVED MARGINALLY. RESETTING RANGE...")
                betaRange = betaRangeInitial
                gammaRange = gammaRangeInitial
            else:
                print("ERROR IMPROVED")
                betaRange *= dRange
                gammaRange *= dRange
            
            print("Beta: " + str(betaMid))
            print("Gamma: " + str(gammaMid))
            print("> Error: " + str(iterMinError))
            params = [betaMid, gammaMid]
            if modelInd == 1:
                params.append(deadRate)
            [_, _, _, out] = RunModel(T, modelInd,
                startS, startI, startR, params)
            print("> Real Error: " + str(CalcError(out, data[country])))
            if supervised:
                outResample = ResampleData(out, data[country])
                plt.plot(outResample)
                plt.plot(data[country])
                plt.show()
        else:
            # Unused. This doesn't work very well.
            print("ERROR WORSENED (" + str(iterMinError) + "), BACKTRACKING")
            if nextBacktrack == 0:
                betaRange /= dRange
            elif nextBacktrack == 1:
                gammaRange /= dRange
            nextBacktrack = (nextBacktrack + 1) % 2
    
    #print("-- Done --")
    #print("beta:  " + str(minParams[0]))
    #print("gamma: " + str(minParams[1]))
    #print("error:  " + str(error))

Present()
#Optimize()
exit()

T = 10000
model = 1
bestTestStart = [
    12e6,
    1800,
    0
]
bestTestParams = [
    [
        5e-10,
        0.002
    ],
    [
        5e-10,
        0.002,
        0.71
    ]
]
sliderRange = 2.0

[S, I, R, out] = RunModel(T, model,
    bestTestStart[0], bestTestStart[1], bestTestStart[2],
    bestTestParams[model])

fig, ax = plt.subplots()
plt.subplots_adjust(left = 0.35, bottom = 0.35)
t = np.linspace(0.0, 1.0, num = T, endpoint = True)
lineS, = plt.plot(t, S, lw=2.0, color='orange', label="Susceptible")
lineI, = plt.plot(t, I, lw=2.0, color='red', label="Infected")
lineR, = plt.plot(t, R, lw=2.0, color='blue', label="Recovered")
plt.xlabel("Time (normalized 0 to 1)")
plt.ylabel("Number of people")
plt.legend(handles=[lineS, lineI, lineR])
#lineOut, = plt.plot(t, output, color="red")

axisColor = 'lightgoldenrodyellow'
axisBeta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axisColor)
axisGamma = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axisColor)
sliderBeta = Slider(axisBeta, 'beta',
    bestTestParams[model][0] / sliderRange,
    bestTestParams[model][0] * sliderRange,
    valinit = bestTestParams[model][0])
sliderGamma = Slider(axisGamma, 'gamma',
    bestTestParams[model][1] / sliderRange,
    bestTestParams[model][1] * sliderRange,
    valinit = bestTestParams[model][1])

def UpdatePlot(val):
    beta = sliderBeta.val
    gamma = sliderGamma.val
    bestTestParams[model][0] = beta
    bestTestParams[model][1] = gamma
    [S, I, R, output] = RunModel(T, model,
        bestTestStart[0], bestTestStart[1], bestTestStart[2],
        bestTestParams[model])
    lineS.set_ydata(S)
    lineI.set_ydata(I)
    lineR.set_ydata(R)
    #lineOut.set_ydata(output)
    fig.canvas.draw_idle()

sliderBeta.on_changed(UpdatePlot)
sliderGamma.on_changed(UpdatePlot)

axisPrint = plt.axes([0.8, 0.025, 0.1, 0.04])
buttonError = Button(axisPrint, 'Calc Error', color=axisColor, hovercolor='0.975')

def PlotCalcError(event):
    pass
    #beta = sliderBeta.val
    #r0 = sliderR0.val
    #gamma = beta * startS / r0
    #[_, _, _, output] = RunSIR(beta, gamma, startS, startI, startR, T)
    #print("Error: " + str(CalcError(output, data["Sierra Leone"])))
buttonError.on_clicked(PlotCalcError)

plt.show()