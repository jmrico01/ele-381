import sys
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons

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
print("----- Data Imported -----")
print("Data points: " + str(n))
print("Start date:  " + str(dateStart))
print("End date:    " + str(dateEnd))
print("")

def Present(T, startS, startI, startD, beta, gamma, dRate, data):
    [S, I, R, out] = models.RunModelSpatial(T, startS, startI, startD,
        beta, gamma, dRate)
    
    errors = models.CalcErrorSpatialFromParams(T, startS, startI, startD,
        beta, gamma, dRate, get_data.bestParamsSingle, data, countries)

    FIG_DPI = 100
    N = len(startS)
    fig, axes = plt.subplots(1, N)
    fig.set_dpi(FIG_DPI)
    fig.set_size_inches(1600 / FIG_DPI, 800 / FIG_DPI)
    for i in range(N):
        outSpatialResample = models.ResampleData(out[i], data[countries[i]])
        lineSpatial, = axes[i].plot(outSpatialResample,
            label="Spatial ( " + str(100.0 * errors["spatial"][i]) + "% )")

        [_, _, _, outSingle] = models.RunModelSingle(get_data.T, 1,
            startS[i], startI[i], startD[i],
            get_data.bestParamsSingle[countries[i]][1])
        outSingleResample = models.ResampleData(outSingle, data[countries[i]])
        lineSingle, = axes[i].plot(outSingleResample,
            label="Single ( " + str(100.0 * errors["single"][i]) + "% )")

        lineData, = axes[i].plot(data[countries[i]], label="Data")
        axes[i].legend(handles=[lineSpatial, lineSingle, lineData])

        axes[i].set_title(countries[i])
        axes[i].set_xlabel("Time (days since 08/29/2014)")
        axes[i].set_ylabel("Cumulative number of Ebola cases")
    
    plt.show()

def Optimize():
    T = get_data.T
    N = len(countries)
    startS = np.empty((N,))
    startI = np.empty((N,))
    startD = np.zeros((N,))
    totalPop = np.empty((N,))
    for i in range(N):
        startS[i] = get_data.startData[countries[i]][0]
        startI[i] = get_data.startData[countries[i]][1]
        startD[i] = get_data.startData[countries[i]][2]
    totalPop = startS + startI + startD

    betaStart = np.array(get_data.bestParamsSpatial[0])
    gammaStart = np.array(get_data.bestParamsSpatial[1])
    dRate = get_data.FATALITY_RATE

    errors = models.CalcErrorSpatialFromParams(T, startS, startI, startD,
        betaStart, gammaStart, dRate, get_data.bestParamsSingle,
        data, countries)
    print(errors)

    while np.sum(errors["single"]) <= np.sum(errors["spatial"]):
        beta = np.copy(betaStart)
        gamma = np.copy(gammaStart)
        for i in range(N):
            betaModifier = np.random.uniform(-0.005, 0.005)
            beta[i, i] += beta[i, i] * betaModifier
            betaSpreadFrac = np.random.uniform(0.0, 0.005)
            betaSpread = np.random.uniform(0.0, 1.0, N)
            betaSpread[i] = 0.0
            betaSpread *= beta[i, i] * betaSpreadFrac / np.sum(betaSpread)
            beta[i, i] -= beta[i, i] * betaSpreadFrac
            beta[i, :] += betaSpread

            gammaModifier = np.random.uniform(-0.01, 0.01)
            gamma[i] += gamma[i] * gammaModifier

        #print(beta)
        #print(gamma)
        errors = models.CalcErrorSpatialFromParams(T, startS, startI, startD,
            beta, gamma, dRate, get_data.bestParamsSingle,
            data, countries)
        print(np.sum(errors["spatial"]) - np.sum(errors["single"]))
        #break

    print(beta)
    print(gamma)
    print(errors)

if len(sys.argv) == 2:
    if sys.argv[1] == "opt":
        Optimize()
        exit()
    elif sys.argv[1] == "present":
        N = len(countries)
        startS = np.empty((N,))
        startI = np.empty((N,))
        startD = np.zeros((N,))
        totalPop = np.empty((N,))
        for i in range(N):
            startS[i] = get_data.startData[countries[i]][0]
            startI[i] = get_data.startData[countries[i]][1]
            startD[i] = get_data.startData[countries[i]][2]
        totalPop = startS + startI + startD

        beta = np.array(get_data.bestParamsSpatial[0])
        gamma = np.array(get_data.bestParamsSpatial[1])
        dRate = get_data.FATALITY_RATE
        print(beta)
        print(gamma)

        Present(get_data.T, startS, startI, startD, beta, gamma, dRate, data)
        exit()
    elif sys.argv[1] == "show":
        # Pass through to model demo below
        pass
    elif sys.argv[1] == "random":
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

# ----- Visualization test -----
if len(sys.argv) != 2:
    print("Expected 1 argument")
    exit()

N = int(sys.argv[1]) # Number of locations

startS = np.empty((N,))
startI = np.empty((N,))
startD = np.zeros((N,))
totalPop = np.empty((N,))

beta = np.empty((N, N))
gamma = np.empty((N,))
dRate = get_data.FATALITY_RATE

startSAvg = 8e6
startSRange = -0.8
startS = np.random.normal(startSAvg,
    startSAvg * 10**startSRange, startS.shape)
startIAvg = 1000
startIRange = -0.5
startI = np.random.normal(startIAvg,
    startIAvg * 10**startIRange, startI.shape)
totalPop = startS + startI
betaAvg = 1e-10
betaRange = -1.0
gammaAvg = 2e-4
gammaRange = -1.5
# Randomize beta, gamma
beta = np.random.normal(betaAvg,
    betaAvg * 10**betaRange, beta.shape)
gamma = np.random.normal(gammaAvg,
    gammaAvg * 10**gammaRange, gamma.shape)

print("----- Running Spatial SIDS -----")
#print(startS[0])
#print(startI[0])
[S, I, D, out] = RunSpatialSIDS(get_data.T,
    startS, startI, startD, beta, gamma, dRate)

#print(S[0, :])
if False:
    t = np.linspace(0.0, 1.0, get_data.T)
    outS = np.sum(S, axis=0)
    outI = np.sum(I, axis=0)
    outD = np.sum(D, axis=0)
    lineS, = plt.plot(t, outS, color="orange", label="Susceptible")
    lineI, = plt.plot(t, outI, color="red", label="Infected")
    lineD, = plt.plot(t, outD, color="black", label="Dead")
    plt.xlabel("Time (normalized 0 to 1)")
    plt.ylabel("Number of people")
    plt.legend(handles=[lineS, lineI, lineD])
    plt.show()

fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()
plt.subplots_adjust(left = 0.1, bottom = 0.2)

def PlotCircles(St, It, Dt):
    gridN = int(np.ceil(np.sqrt(N)))
    circleRadius = 1.0 / gridN / 4.0
    margin = 0.2
    ax.clear()
    for loc in range(N):
        (j, i) = np.unravel_index(loc, (gridN, gridN))
        posX = margin + (1.0 - margin * 2.0) * (float(i) / (gridN-1))
        posY = margin + (1.0 - margin * 2.0) * (float(j) / (gridN-1))
        fracS = St[loc] / totalPop[0]
        fracI = It[loc] / totalPop[0]
        fracD = Dt[loc] / totalPop[0]
        circleS = fracS * circleRadius
        circleI = circleS + fracI * circleRadius
        circleD = circleI + fracD * circleRadius
        circle = plt.Circle((posX, posY), circleD, color='black')
        ax.add_artist(circle)
        circle = plt.Circle((posX, posY), circleI, color='red')
        ax.add_artist(circle)
        circle = plt.Circle((posX, posY), circleS, color='green')
        ax.add_artist(circle)

PlotCircles(S[:, 0], I[:, 0], D[:, 0])

axisColor = 'lightgoldenrodyellow'
axisTime = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axisColor)
sliderTime = Slider(axisTime, 'Time', 0.0, 1.0, 0.0)

def UpdatePlot(val):
    time = sliderTime.val
    t = min(int(time * get_data.T), get_data.T-1)
    PlotCircles(S[:, t], I[:, t], D[:, t])

sliderTime.on_changed(UpdatePlot)

plt.show()