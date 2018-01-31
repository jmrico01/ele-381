import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import get_data
import models

# -- Import data
dataFilePath = "ebola_data_db_format.csv"
countries = [
    "Sierra Leone",
    "Guinea",
    "Liberia"
]
categories = [
    get_data.CAT_CONFIRMED
]
[data, n, dateStart, dateEnd] = get_data.ReadData(dataFilePath,
    countries, categories, True, 5, False)
#print("----- Data Imported -----")
#print("Data points: " + str(n))
#print("Start date:  " + str(dateStart))
#print("End date:    " + str(dateEnd))
#print("")

country = get_data.COUNTRY_TOTAL
modelInd = 1

def Present(T, modelInd, startS, startI, startR, params, data, country):
    [S, I, R, out] = models.RunModelSingle(T, modelInd,
        startS, startI, startR, params[modelInd])
    error = round(models.CalcErrorSingle(out, data), 6)
    #print("Error: " + str(error))
    out = models.ResampleData(out, data)
    lineModel, = plt.plot(out, label="Model")
    lineData,  = plt.plot(data, label="Data")
    plotTitle = country
    if modelInd == 0:
        plotTitle += " - SIR Model"
    elif modelInd == 1:
        plotTitle += " - SIDS Model"
    plotTitle += " ( error: " + str(100.0 * error) + "% )"
    plt.title(plotTitle)
    plt.xlabel("Time (days since 08/29/2014)")
    plt.ylabel("Cumulative number of Ebola cases")
    plt.legend(handles=[lineModel, lineData])
    plt.show()

def CalcCoefValues(mid, orderOfMag, n):
    exp = 2.0
    values = np.linspace(0.0, 1.0, n)**exp
    minVal = mid * 10.0**orderOfMag
    maxVal = mid / 10.0**orderOfMag
    values = values * (maxVal - minVal) + minVal
    return values

def Optimize():
    supervised = True

    print("----- Optimizing SIR Parameters -----")
    startS = get_data.startData[country][0]
    startI = get_data.startData[country][1]
    startR = get_data.startData[country][2]

    paramIters = 100
    betaMid = get_data.bestParamsSingle[country][modelInd][0]
    #betaRangeInitial = 1
    betaRange = 0.1 # in orders of magnitude

    gammaMid = get_data.bestParamsSingle[country][modelInd][1]
    #gammaRangeInitial = 0.
    gammaRange = 0.1

    nextBacktrack = 1
    dRange = 0.5

    params = [betaMid, gammaMid]
    if modelInd == 1:
        params.append(get_data.FATALITY_RATE)

    minError = models.CalcErrorSingleFromParams(
        get_data.T, modelInd, startS, startI, startR, params,
        data[country]
    )
    minParams = [-1, -1]
    print("Initial error: " + str(minError))
    while minError > 0.01: # Probably not feasible, doesn't matter
        beta = CalcCoefValues(betaMid, betaRange, paramIters)
        gamma = CalcCoefValues(gammaMid, gammaRange, paramIters)
        print("GREEDY SEARCH ITERATION")
        print("Beta range:  " + str(np.min(beta)) + " - " + str(np.max(beta)))
        print("Gamma range: " + str(np.min(gamma)) + " - " + str(np.max(gamma)))
        params = [beta, gamma]
        if modelInd == 1:
            params.append(get_data.FATALITY_RATE)
        errors = models.BatchSIR(get_data.T, modelInd, startS, startI, startR,
            data[country], params, True)

        [minBetaInd, minGammaInd] = np.unravel_index(np.argmin(errors),
            (paramIters, paramIters))
        iterMinError = errors[minBetaInd, minGammaInd]
        plotCurve = False
        if iterMinError < minError:
            errDiff = minError - iterMinError
            minError = iterMinError
            betaMid = beta[minBetaInd]
            gammaMid = gamma[minGammaInd]
            minParams = [betaMid, gammaMid]
            print("ERROR IMPROVED")
            betaRange *= dRange
            gammaRange *= dRange
            
            print("Beta: " + str(betaMid))
            print("Gamma: " + str(gammaMid))
            print("> Error: " + str(iterMinError))
            params = [beta[minBetaInd], gamma[minGammaInd]]
            if modelInd == 1:
                params.append(get_data.FATALITY_RATE)
            print("> Real Error: " + str(models.CalcErrorSingleFromParams(
                get_data.T, modelInd, startS, startI, startR, params,
                data[country]
            )))
            plotCurve = True
        else:
            # Unused. This doesn't work very well.
            print("ERROR WORSENED (" + str(iterMinError) + "), BACKTRACKING")
            if nextBacktrack == 0:
                betaRange /= dRange
            elif nextBacktrack == 1:
                gammaRange /= dRange
            nextBacktrack = (nextBacktrack + 1) % 2
            plotCurve = True

        if plotCurve:
            params = [beta[minBetaInd], gamma[minGammaInd]]
            if modelInd == 1:
                params.append(get_data.FATALITY_RATE)
            [_, _, _, out] = models.RunModelSingle(get_data.T, modelInd,
                startS, startI, startR, params)
            if supervised:
                outResample = models.ResampleData(out, data[country])
                plt.plot(outResample)
                plt.plot(data[country])
                plt.show()
    
    #print("-- Done --")
    #print("beta:  " + str(minParams[0]))
    #print("gamma: " + str(minParams[1]))
    #print("error:  " + str(error))

if len(sys.argv) == 2:
    if sys.argv[1] == "opt":
        Optimize()
        exit()
    elif sys.argv[1] == "present":
        Present(get_data.T, modelInd,
            get_data.startData[country][0],
            get_data.startData[country][1],
            get_data.startData[country][2],
            get_data.bestParamsSingle[country], data[country], country)
        exit()
    elif sys.argv[1] == "show":
        # Pass through to model demo below
        pass
    elif sys.argv[1] == "random":
        [_, _, _, out0] = models.RunModelSingle(get_data.T, 0,
            get_data.startData[get_data.COUNTRY_TOTAL][0],
            get_data.startData[get_data.COUNTRY_TOTAL][1],
            get_data.startData[get_data.COUNTRY_TOTAL][2],
            get_data.bestParamsSingle[get_data.COUNTRY_TOTAL][0])
        [_, _, _, out1] = models.RunModelSingle(get_data.T, 1,
            get_data.startData[get_data.COUNTRY_TOTAL][0],
            get_data.startData[get_data.COUNTRY_TOTAL][1],
            get_data.startData[get_data.COUNTRY_TOTAL][2],
            get_data.bestParamsSingle[get_data.COUNTRY_TOTAL][1])
        out0 = models.ResampleData(out0, data[get_data.COUNTRY_TOTAL])
        out1 = models.ResampleData(out1, data[get_data.COUNTRY_TOTAL])
        lineM0, = plt.plot(out0, label="SIR")
        lineM1, = plt.plot(out1, label="SIDS")
        lineData,  = plt.plot(data[get_data.COUNTRY_TOTAL], label="Data")
        plt.title("Model Predictions")
        plt.xlabel("Time (days since 08/29/2014)")
        plt.ylabel("Cumulative number of Ebola cases")
        plt.legend(handles=[lineM0, lineM1, lineData])
        plt.show()
        exit()
    else:
        print("Unrecognized arguments")
        exit()
elif len(sys.argv) == 1:
    Present()
    exit()
else:
    print("Unrecognized arguments")
    exit()

testT = 10000
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
        get_data.FATALITY_RATE
    ]
]
sliderRange = 2.0

[S, I, R, out] = models.RunModelSingle(testT, model,
    bestTestStart[0], bestTestStart[1], bestTestStart[2],
    bestTestParams[model])

fig, ax = plt.subplots()
plt.subplots_adjust(left = 0.1, bottom = 0.35)
t = np.linspace(0.0, 1.0, num = testT, endpoint = True)
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
    [S, I, R, output] = models.RunModelSingle(testT, model,
        bestTestStart[0], bestTestStart[1], bestTestStart[2],
        bestTestParams[model])
    lineS.set_ydata(S)
    lineI.set_ydata(I)
    lineR.set_ydata(R)
    #lineOut.set_ydata(output)
    fig.canvas.draw_idle()

sliderBeta.on_changed(UpdatePlot)
sliderGamma.on_changed(UpdatePlot)

#axisPrint = plt.axes([0.8, 0.025, 0.1, 0.04])
#buttonError = Button(axisPrint, 'Calc Error', color=axisColor, hovercolor='0.975')

def PlotCalcError(event):
    pass
    #beta = sliderBeta.val
    #r0 = sliderR0.val
    #gamma = beta * startS / r0
    #[_, _, _, output] = RunSIR(beta, gamma, startS, startI, startR, testT)
    #print("Error: " + str(CalcErrorSingle(output, data["Sierra Leone"])))
#buttonError.on_clicked(PlotCalcError)

plt.show()