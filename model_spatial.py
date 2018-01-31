import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import read_data

# -- Import data
dataFilePath = "ebola_data_db_format.csv"
countries = [
    "Sierra Leone",
    "Guinea",
    "Liberia"
]
N = len(countries)
categories = [
    read_data.CAT_CONF
]
[data, n, dateStart, dateEnd] = read_data.ReadData(dataFilePath,
    countries, categories, True, 5, False)
print("----- Data Imported -----")
print("Data points: " + str(n))
print("Start date:  " + str(dateStart))
print("End date:    " + str(dateEnd))
print("")

startData = [
    [ # "Sierra Leone": [
        7e6,
        200,
        0
    ],
    [ # "Guinea": [
        11.8e6,
        250,
        0
    ],
    [ # "Liberia": [
        4.39e6,
        100,
        0
    ]
]
bestData = [
    [ # "Sierra Leone": [
        1.59937781919e-09,
        0.01142412728
    ],
    [ # "Guinea": [
        3.97879216751e-10,
        0.0051461993053
    ],
    [ # "Liberia": [
        1.59937781919e-09,
        0.01142412728
    ]
]

T = 10000

startS = np.empty((N,))
startI = np.empty((N,))
startD = np.zeros((N,))
totalPop = np.empty((N,))
for i in range(N):
    startS[i] = startData[i][0]
    startI[i] = startData[i][1]
    startD[i] = startData[i][2]
totalPop = startS + startI + startD

beta = np.zeros((N, N))
gamma = np.zeros((N,))
dRate = 0.71
for i in range(N):
    beta[i][i] = bestData[i][0]
    gamma[i] = bestData[i][1]

for i in range(N):
    for j in range(N):
        if i == j:
            continue

        beta[i][j] = beta[i][i] * 10**(-1)

print(beta)
print(gamma)

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

[S, I, D, out] = RunSpatialSIDS(T,
    startS, startI, startD, beta, gamma, dRate)


t = np.linspace(0.0, 1.0, T)
#outS = S[0]#np.sum(S, axis=0)
outI = np.sum(I, axis=0)
#outD = D[0]#np.sum(D, axis=0)
#lineS, = plt.plot(t, outS, color="orange", label="Susceptible")
lineI, = plt.plot(t, outI, color="red", label="Infected")
#lineD, = plt.plot(t, outD, color="black", label="Dead")
plt.xlabel("Time (normalized 0 to 1)")
plt.ylabel("Number of people")
#plt.legend(handles=[lineS, lineI, lineD])
plt.show()

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
dRate = 0.71

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
[S, I, D, out] = RunSpatialSIDS(T,
    startS, startI, startD, beta, gamma, dRate)

#print(S[0, :])
if False:
    t = np.linspace(0.0, 1.0, T)
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
    t = min(int(time * T), T-1)
    PlotCircles(S[:, t], I[:, t], D[:, t])

sliderTime.on_changed(UpdatePlot)

plt.show()